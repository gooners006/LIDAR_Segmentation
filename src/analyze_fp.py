"""False-positive analysis: log per-cluster features to CSV for filter tuning.

Runs pipeline stages 1-5 on SemanticKITTI frames, matches detections to GT,
and writes one row per detected cluster with geometric features and match status.

Usage:
    python src/analyze_fp.py --seq 00 --frames 100
    python src/analyze_fp.py --seq 00 --frames 100 --iou-threshold 0.3
"""

import argparse
import csv
import glob
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from evaluate import THING_CLASSES_ALL, compute_iou
from pipeline import (
    PIPELINE_CONFIG,
    cluster_objects,
    filter_clusters,
    remove_ground,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEMANTICKITTI_NAMES = {
    0: "unlabeled", 1: "outlier",
    10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
    16: "on-rails", 18: "truck", 20: "other-vehicle",
    30: "person", 31: "bicyclist", 32: "motorcyclist",
    40: "road", 44: "parking", 48: "sidewalk",
    49: "other-ground", 50: "building", 51: "fence",
    52: "other-structure", 60: "lane-marking", 70: "vegetation",
    71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
    99: "other-object",
    252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
    255: "moving-motorcyclist", 256: "moving-on-rails",
    257: "moving-bus", 258: "moving-truck", 259: "moving-other-vehicle",
}

CSV_COLUMNS = [
    "frame",
    "cluster_label",
    "is_tp",
    "matched_gt_id",
    "best_iou",
    "semantic_majority_label",
    "semantic_majority_name",
    "point_count",
    "extent_min",
    "extent_med",
    "extent_max",
    "volume",
    "aspect_max_min",
    "aspect_med_min",
    "height_span",
    "height_above_ground",
    "distance_to_ego",
    "cluster_density",
]


def analyze_frame(bin_path: str, label_path: str, frame_idx: int,
                  iou_threshold: float) -> list[dict]:
    """Run the pipeline on one frame and extract per-cluster diagnostic features.

    Replicates stages 1–5 with GT label propagation, matches detections
    to GT instances via greedy IoU, then computes geometric features for
    each detected cluster (TP and FP alike).

    Args:
        bin_path: Path to the Velodyne ``.bin`` file.
        label_path: Path to the SemanticKITTI ``.label`` file.
        frame_idx: Frame number (stored in output rows).
        iou_threshold: Minimum IoU for a detection–GT match.

    Returns:
        List of dicts (one per detected cluster) with columns defined
        by ``CSV_COLUMNS``.
    """
    cfg = PIPELINE_CONFIG

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem = raw_labels & 0xFFFF
    inst = raw_labels >> 16

    # Preprocessing with label propagation (mirrors evaluate.py)
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
    xyz_denoised = np.asarray(pcd_denoised.points)

    pcd_down = pcd_denoised.voxel_down_sample(voxel_size=cfg["voxel_size"])
    xyz_down = np.asarray(pcd_down.points)

    tree = cKDTree(xyz_denoised)
    _, nn_idx = tree.query(xyz_down)
    sem_down = sem_denoised[nn_idx]
    inst_down = inst_denoised[nn_idx]

    # Pipeline stages 3-5
    ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)

    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[ground_inliers] = False

    sem_obj = sem_down[object_mask]
    inst_obj = inst_down[object_mask]

    cluster_labels = cluster_objects(objects_pcd)
    clusters = filter_clusters(objects_pcd, cluster_labels, ground_plane)
    obj_points = np.asarray(objects_pcd.points)

    # Build GT masks
    thing_mask = np.isin(sem_obj, list(THING_CLASSES_ALL))
    gt_instances = np.unique(inst_obj[thing_mask])
    gt_instances = gt_instances[gt_instances > 0]
    gt_masks = {}
    for gi in gt_instances:
        m = (inst_obj == gi) & thing_mask
        if m.sum() >= 10:
            gt_masks[gi] = m

    # Build detection masks and compute all pairwise IoUs
    det_masks = {}
    for bbox, cl in clusters:
        det_masks[cl] = (cluster_labels == cl)

    # Greedy matching (same logic as evaluate.py)
    pairs = []
    for det_id, det_m in det_masks.items():
        for gt_id, gt_m in gt_masks.items():
            iou = compute_iou(det_m, gt_m)
            if iou > 0:
                pairs.append((iou, det_id, gt_id))
    pairs.sort(reverse=True)

    matched_det = set()
    matched_gt = set()
    det_to_gt = {}
    det_to_iou = {}

    for iou, det_id, gt_id in pairs:
        if det_id in matched_det or gt_id in matched_gt:
            continue
        if iou >= iou_threshold:
            matched_det.add(det_id)
            matched_gt.add(gt_id)
            det_to_gt[det_id] = gt_id
            det_to_iou[det_id] = iou

    # Best IoU for unmatched detections (for diagnostics)
    best_iou_any = {}
    for det_id in det_masks:
        best = 0.0
        for gt_id, gt_m in gt_masks.items():
            iou = compute_iou(det_masks[det_id], gt_m)
            if iou > best:
                best = iou
        best_iou_any[det_id] = best

    # Ground plane for height computation
    if ground_plane is not None:
        gp_a, gp_b, gp_c, gp_d = ground_plane
        gp_norm = np.sqrt(gp_a**2 + gp_b**2 + gp_c**2)

    rows = []
    for bbox, cl in clusters:
        mask = cluster_labels == cl
        cluster_pts = obj_points[mask]
        cluster_sem = sem_obj[mask]
        n_pts = int(mask.sum())

        # Semantic majority
        if n_pts > 0:
            labels, counts = np.unique(cluster_sem, return_counts=True)
            maj_label = int(labels[counts.argmax()])
        else:
            maj_label = 0

        # OBB extents
        extent = np.sort(bbox.extent)
        ext_min, ext_med, ext_max = extent[0], extent[1], extent[2]
        volume = bbox.volume()

        # Aspect ratios
        aspect_max_min = ext_max / ext_min if ext_min > 1e-6 else 0.0
        aspect_med_min = ext_med / ext_min if ext_min > 1e-6 else 0.0

        # Height span
        if n_pts > 0:
            height_span = float(cluster_pts[:, 2].max() - cluster_pts[:, 2].min())
        else:
            height_span = 0.0

        # Height above ground
        center = bbox.get_center()
        if ground_plane is not None:
            height_above = (gp_a * center[0] + gp_b * center[1]
                            + gp_c * center[2] + gp_d) / gp_norm
        else:
            height_above = center[2]

        # Distance to ego (origin in local frame)
        dist_ego = float(np.linalg.norm(center))

        # Density
        density = n_pts / volume if volume > 1e-6 else 0.0

        is_tp = cl in det_to_gt
        rows.append({
            "frame": frame_idx,
            "cluster_label": cl,
            "is_tp": int(is_tp),
            "matched_gt_id": det_to_gt.get(cl, -1),
            "best_iou": round(det_to_iou[cl] if is_tp else best_iou_any.get(cl, 0.0), 4),
            "semantic_majority_label": maj_label,
            "semantic_majority_name": SEMANTICKITTI_NAMES.get(maj_label, f"id_{maj_label}"),
            "point_count": n_pts,
            "extent_min": round(ext_min, 4),
            "extent_med": round(ext_med, 4),
            "extent_max": round(ext_max, 4),
            "volume": round(volume, 4),
            "aspect_max_min": round(aspect_max_min, 4),
            "aspect_med_min": round(aspect_med_min, 4),
            "height_span": round(height_span, 4),
            "height_above_ground": round(float(height_above), 4),
            "distance_to_ego": round(dist_ego, 4),
            "cluster_density": round(density, 2),
        })

    return rows


def main():
    """Run FP analysis across a sequence and write a diagnostic CSV.

    For each frame, extracts per-cluster features and match status,
    then prints a summary of TP/FP distributions by semantic class and
    geometric properties.
    """
    parser = argparse.ArgumentParser(description="False-positive analysis")
    parser.add_argument("--seq", default="00", help="Sequence ID")
    parser.add_argument("--frames", type=int, default=100, help="Max frames")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: output/eval_fp_analysis.csv)")
    args = parser.parse_args()

    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:args.frames]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[:args.frames]

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "output", f"eval_fp_analysis_{args.seq}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_rows = []
    n_frames = len(bin_paths)
    print(f"Analyzing {n_frames} frames (seq {args.seq}, IoU threshold {args.iou_threshold})...")

    for i in range(n_frames):
        rows = analyze_frame(bin_paths[i], label_paths[i], i, args.iou_threshold)
        tp = sum(1 for r in rows if r["is_tp"])
        fp = len(rows) - tp
        all_rows.extend(rows)
        print(f"  Frame {i:3d}: {len(rows):3d} detections ({tp} TP, {fp} FP)")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    total = len(all_rows)
    total_tp = sum(1 for r in all_rows if r["is_tp"])
    total_fp = total - total_tp

    print(f"\nWrote {total} rows to {out_path}")
    print(f"  TP: {total_tp}  FP: {total_fp}  Precision: {total_tp/total:.3f}" if total else "  No detections")

    tp_rows = [r for r in all_rows if r["is_tp"]]
    fp_rows = [r for r in all_rows if not r["is_tp"]]

    def print_distribution(label: str, rows: list[dict]):
        if not rows:
            print(f"\n  {label}: no samples")
            return
        n = len(rows)

        sem_counts: dict[str, int] = {}
        for r in rows:
            name = r["semantic_majority_name"]
            sem_counts[name] = sem_counts.get(name, 0) + 1
        print(f"\n  {label} breakdown by semantic class ({n} total):")
        for name, count in sorted(sem_counts.items(), key=lambda x: -x[1]):
            print(f"    {name:20s}: {count:4d} ({100*count/n:.1f}%)")

        for feat in ["point_count", "volume", "distance_to_ego",
                      "extent_max", "height_span", "cluster_density"]:
            vals = [r[feat] for r in rows]
            print(f"  {label} {feat:18s}: median={np.median(vals):8.2f}  "
                  f"p25={np.percentile(vals,25):8.2f}  "
                  f"p75={np.percentile(vals,75):8.2f}")

    print_distribution("TP", tp_rows)
    print_distribution("FP", fp_rows)


if __name__ == "__main__":
    main()
