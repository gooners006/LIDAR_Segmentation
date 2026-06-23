"""Analyze HDBSCAN clustering quality and geometric filter rejection.

Two analyses:
1. Geometric filter ablation — which filter rejects the most GT-matching clusters
2. Merge/split analysis — how GT cars distribute across HDBSCAN clusters
"""

import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline import PIPELINE_CONFIG, cluster_objects, remove_ground, _center_height_above_ground

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THING_CLASSES_SUPPORTED = {10, 252}


def diagnose_geometric_rejection(cluster_pcd, ground_plane, config):
    """Return the name of the first geometric filter that rejects this cluster,
    or None if it passes all filters."""
    n_pts = len(cluster_pcd.points)
    if n_pts < config["min_points_in_cluster"]:
        return "min_points"

    bbox = cluster_pcd.get_oriented_bounding_box()
    volume = bbox.volume()
    if volume < config["min_volume"]:
        return "min_volume"
    if volume > config["max_volume"]:
        return "max_volume"

    min_dim, med_dim, max_dim = np.sort(bbox.extent)
    if max_dim > config["max_dim_length"]:
        return "max_dim_length"
    if max_dim < config["min_max_dim"]:
        return "min_max_dim"
    if med_dim < config["min_med_dim"]:
        return "min_med_dim"

    height = _center_height_above_ground(bbox.get_center(), ground_plane)
    if height > config["max_center_height_above_ground"]:
        return "max_center_height"

    pts = np.asarray(cluster_pcd.points)
    height_span = pts[:, 2].max() - pts[:, 2].min()
    if height_span > config["max_height_span"]:
        return "max_height_span"

    aspect = max_dim / min_dim if min_dim > 1e-6 else float("inf")
    if aspect > config["max_aspect_max_min"]:
        return "max_aspect"

    return None


def process_frame(bin_path, label_path, cfg):
    """Run pipeline through clustering, return per-cluster and per-GT analysis."""

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem = raw_labels & 0xFFFF
    inst = raw_labels >> 16

    z_mask = xyz[:, 2] > cfg["z_threshold"]
    xyz_filtered = xyz[z_mask]
    sem_filtered = sem[z_mask]
    inst_filtered = inst[z_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
    pcd_denoised, ind_denoise = pcd.remove_statistical_outlier(
        nb_neighbors=cfg["denoise_nb_neighbors"],
        std_ratio=cfg["denoise_std_ratio"],
    )
    sem_denoised = sem_filtered[ind_denoise]
    inst_denoised = inst_filtered[ind_denoise]

    pcd_down = pcd_denoised.voxel_down_sample(voxel_size=cfg["voxel_size"])
    xyz_down = np.asarray(pcd_down.points)
    tree = cKDTree(np.asarray(pcd_denoised.points))
    _, nn_idx = tree.query(xyz_down)
    sem_down = sem_denoised[nn_idx]
    inst_down = inst_denoised[nn_idx]

    ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)

    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[ground_inliers] = False
    sem_obj = sem_down[object_mask]
    inst_obj = inst_down[object_mask]

    cluster_labels = cluster_objects(objects_pcd)
    obj_points = np.asarray(objects_pcd.points)

    # --- Build GT instance info ---
    thing_mask = np.isin(sem_obj, list(THING_CLASSES_SUPPORTED))
    gt_instances = np.unique(inst_obj[thing_mask])
    gt_instances = gt_instances[gt_instances > 0]

    gt_point_masks = {}
    for gi in gt_instances:
        m = (inst_obj == gi) & thing_mask
        if m.sum() >= 10:
            gt_point_masks[gi] = m

    # ===== Analysis 1: Geometric filter ablation =====
    filter_rejections_gt = Counter()  # filter_name -> count of GT-matching clusters killed
    filter_rejections_all = Counter()  # filter_name -> count of all clusters killed
    n_clusters_total = 0
    n_clusters_passed = 0

    unique_labels = set(cluster_labels.tolist()) - {-1}
    n_clusters_total = len(unique_labels)

    for cl in unique_labels:
        cl_mask = cluster_labels == cl
        cluster_pcd = objects_pcd.select_by_index(np.where(cl_mask)[0])
        rejection = diagnose_geometric_rejection(cluster_pcd, ground_plane, cfg)

        if rejection is None:
            n_clusters_passed += 1
            continue

        filter_rejections_all[rejection] += 1

        # Does this cluster overlap with any GT car?
        for gi, gt_m in gt_point_masks.items():
            overlap = (cl_mask & gt_m).sum()
            if overlap >= 5:
                filter_rejections_gt[rejection] += 1
                break

    # ===== Analysis 2: Merge/split =====
    gt_analysis = []
    for gi, gt_m in gt_point_masks.items():
        gt_n_pts = gt_m.sum()
        gt_cluster_labels = cluster_labels[gt_m]

        label_counts = Counter(gt_cluster_labels.tolist())
        noise_pts = label_counts.pop(-1, 0)
        noise_frac = noise_pts / gt_n_pts

        if not label_counts:
            gt_analysis.append({
                "gt_id": gi, "n_pts": gt_n_pts,
                "status": "all_noise", "noise_frac": noise_frac,
                "n_clusters": 0, "dominant_frac": 0.0,
            })
            continue

        dominant_label = max(label_counts, key=label_counts.get)
        dominant_pts = label_counts[dominant_label]
        dominant_frac = dominant_pts / gt_n_pts
        n_clusters = len(label_counts)

        gt_analysis.append({
            "gt_id": gi, "n_pts": gt_n_pts,
            "status": "ok" if n_clusters == 1 and noise_frac < 0.3 else
                      "split" if n_clusters > 1 else "noisy",
            "noise_frac": noise_frac,
            "n_clusters": n_clusters,
            "dominant_frac": dominant_frac,
            "dominant_label": dominant_label,
        })

    # Check for merges: multiple GT instances sharing the same dominant cluster
    label_to_gts = defaultdict(list)
    for g in gt_analysis:
        if "dominant_label" in g:
            label_to_gts[g["dominant_label"]].append(g["gt_id"])

    merged_gt_ids = set()
    for label, gts in label_to_gts.items():
        if len(gts) > 1:
            merged_gt_ids.update(gts)

    for g in gt_analysis:
        if g["gt_id"] in merged_gt_ids and g["status"] != "all_noise":
            g["merged"] = True
        else:
            g["merged"] = False

    return {
        "n_clusters_total": n_clusters_total,
        "n_clusters_passed": n_clusters_passed,
        "filter_rejections_gt": filter_rejections_gt,
        "filter_rejections_all": filter_rejections_all,
        "gt_analysis": gt_analysis,
        "n_noise_points": int((cluster_labels == -1).sum()),
        "n_object_points": len(obj_points),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="00")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    cfg = PIPELINE_CONFIG
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:args.frames]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[:args.frames]
    n_frames = len(bin_paths)

    print(f"Analyzing {n_frames} frames (seq {args.seq})...")
    print("=" * 70)

    # Accumulators
    total_filter_rej_gt = Counter()
    total_filter_rej_all = Counter()
    total_clusters = 0
    total_passed = 0
    all_gt = []
    total_noise_pts = 0
    total_obj_pts = 0

    for i in range(n_frames):
        result = process_frame(bin_paths[i], label_paths[i], cfg)
        total_clusters += result["n_clusters_total"]
        total_passed += result["n_clusters_passed"]
        total_filter_rej_gt += result["filter_rejections_gt"]
        total_filter_rej_all += result["filter_rejections_all"]
        all_gt.extend(result["gt_analysis"])
        total_noise_pts += result["n_noise_points"]
        total_obj_pts += result["n_object_points"]

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"  Frame {i:3d} done ({len(result['gt_analysis'])} GT cars)")

    # ===== Report: Geometric Filter Ablation =====
    print("\n" + "=" * 70)
    print("GEOMETRIC FILTER ABLATION")
    print("=" * 70)
    print(f"Total HDBSCAN clusters: {total_clusters}")
    print(f"Passed all filters:     {total_passed} ({100*total_passed/total_clusters:.1f}%)")
    print(f"Rejected:               {total_clusters - total_passed}")
    print(f"HDBSCAN noise points:   {total_noise_pts}/{total_obj_pts} "
          f"({100*total_noise_pts/total_obj_pts:.1f}%)")

    print(f"\nAll rejected clusters by filter (first failing check):")
    for f, c in total_filter_rej_all.most_common():
        print(f"  {f:25s}: {c:5d}")

    print(f"\nGT-matching clusters rejected by filter:")
    total_gt_killed = sum(total_filter_rej_gt.values())
    print(f"  Total GT-matching clusters killed: {total_gt_killed}")
    for f, c in total_filter_rej_gt.most_common():
        print(f"  {f:25s}: {c:5d} ({100*c/total_gt_killed:.1f}%)")

    # ===== Report: Merge/Split Analysis =====
    print("\n" + "=" * 70)
    print("HDBSCAN MERGE/SPLIT ANALYSIS")
    print("=" * 70)
    n_gt = len(all_gt)
    print(f"Total GT car instances (>=10 pts): {n_gt}")

    status_counts = Counter(g["status"] for g in all_gt)
    for s in ["ok", "split", "noisy", "all_noise"]:
        c = status_counts.get(s, 0)
        print(f"  {s:12s}: {c:5d} ({100*c/n_gt:.1f}%)")

    n_merged = sum(1 for g in all_gt if g["merged"])
    print(f"  {'merged':12s}: {n_merged:5d} ({100*n_merged/n_gt:.1f}%)")

    # Split details
    split_gt = [g for g in all_gt if g["status"] == "split"]
    if split_gt:
        n_clusters_dist = [g["n_clusters"] for g in split_gt]
        dom_fracs = [g["dominant_frac"] for g in split_gt]
        print(f"\nSplit GT cars detail:")
        print(f"  Mean clusters per split GT: {np.mean(n_clusters_dist):.1f}")
        print(f"  Mean dominant cluster fraction: {np.mean(dom_fracs):.2f}")
        print(f"  Median dominant cluster fraction: {np.median(dom_fracs):.2f}")

    # Merged details
    merged_gt = [g for g in all_gt if g["merged"]]
    if merged_gt:
        print(f"\nMerged GT cars detail:")
        print(f"  {n_merged} GT instances share a cluster with another GT car")

    # Noise analysis
    noise_fracs = [g["noise_frac"] for g in all_gt]
    print(f"\nNoise fraction per GT car:")
    print(f"  Mean:   {np.mean(noise_fracs):.2f}")
    print(f"  Median: {np.median(noise_fracs):.2f}")
    print(f"  >50% noise: {sum(1 for f in noise_fracs if f > 0.5)} GT cars")

    # Point-size distribution for different statuses
    print(f"\nGT car point counts by status:")
    for status in ["ok", "split", "all_noise"]:
        pts = [g["n_pts"] for g in all_gt if g["status"] == status]
        if pts:
            print(f"  {status:12s}: median={np.median(pts):.0f}, "
                  f"mean={np.mean(pts):.0f}, min={min(pts)}, max={max(pts)}")

    # Combined: what fraction of GT cars are "recoverable" (clean single cluster)?
    clean = sum(1 for g in all_gt if g["status"] == "ok" and not g["merged"])
    print(f"\nRecoverable GT cars (single cluster, not merged, <30% noise): "
          f"{clean}/{n_gt} ({100*clean/n_gt:.1f}%)")
    print(f"Lost to merge:  {n_merged}")
    lost_split = sum(1 for g in all_gt if g["status"] == "split" and not g["merged"])
    print(f"Lost to split:  {lost_split}")
    lost_noise = sum(1 for g in all_gt if g["status"] in ('all_noise', 'noisy'))
    print(f"Lost to noise:  {lost_noise}")
