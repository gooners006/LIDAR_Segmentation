"""Explore two strategies for improving recall by addressing HDBSCAN splitting.

1. Post-clustering merge: measure fragment distances to see if merging is feasible
   without accidentally merging distinct cars.
2. Adaptive HDBSCAN: measure distance-to-sensor for split vs ok cars, and test
   the effect of increasing min_cluster_size.
"""

import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline import PIPELINE_CONFIG, cluster_objects, remove_ground, _cluster_hdbscan

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THING_CLASSES = {10, 252}


def process_frame(bin_path, label_path, cfg):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem = raw_labels & 0xFFFF
    inst = raw_labels >> 16

    z_mask = xyz[:, 2] > cfg["z_threshold"]
    xyz_f = xyz[z_mask]
    sem_f = sem[z_mask]
    inst_f = inst[z_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_f)
    pcd_dn, ind = pcd.remove_statistical_outlier(
        nb_neighbors=cfg["denoise_nb_neighbors"],
        std_ratio=cfg["denoise_std_ratio"],
    )
    sem_dn = sem_f[ind]
    inst_dn = inst_f[ind]

    pcd_down = pcd_dn.voxel_down_sample(voxel_size=cfg["voxel_size"])
    xyz_down = np.asarray(pcd_down.points)
    tree = cKDTree(np.asarray(pcd_dn.points))
    _, nn_idx = tree.query(xyz_down)
    sem_down = sem_dn[nn_idx]
    inst_down = inst_dn[nn_idx]

    ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)
    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[ground_inliers] = False
    sem_obj = sem_down[object_mask]
    inst_obj = inst_down[object_mask]
    obj_points = np.asarray(objects_pcd.points)

    cluster_labels = cluster_objects(objects_pcd, cfg)

    thing_mask = np.isin(sem_obj, list(THING_CLASSES))
    gt_instances = np.unique(inst_obj[thing_mask])
    gt_instances = gt_instances[gt_instances > 0]

    gt_info = {}
    for gi in gt_instances:
        m = (inst_obj == gi) & thing_mask
        if m.sum() >= 10:
            gt_info[gi] = m

    results = []
    for gi, gt_mask in gt_info.items():
        gt_pts = obj_points[gt_mask]
        gt_centroid = gt_pts.mean(axis=0)
        gt_range = np.linalg.norm(gt_centroid[:2])
        gt_n_pts = gt_mask.sum()

        cl_labels = cluster_labels[gt_mask]
        label_counts = Counter(cl_labels.tolist())
        label_counts.pop(-1, None)

        if not label_counts:
            results.append({
                "gt_id": gi, "n_pts": gt_n_pts, "range": gt_range,
                "status": "all_noise", "n_clusters": 0,
                "fragment_centroids": [], "fragment_sizes": [],
            })
            continue

        n_clusters = len(label_counts)
        status = "ok" if n_clusters == 1 else "split"

        fragment_centroids = []
        fragment_sizes = []
        for cl, count in label_counts.items():
            cl_mask = cluster_labels == cl
            cl_pts = obj_points[cl_mask]
            fragment_centroids.append(cl_pts.mean(axis=0))
            fragment_sizes.append(count)

        results.append({
            "gt_id": gi, "n_pts": gt_n_pts, "range": gt_range,
            "status": status, "n_clusters": n_clusters,
            "fragment_centroids": fragment_centroids,
            "fragment_sizes": fragment_sizes,
        })

    # Also compute inter-car distances for distinct GT cars that are NOT split
    # (to understand the danger zone for merging)
    ok_cars = [r for r in results if r["status"] == "ok"]
    inter_car_dists = []
    for i in range(len(ok_cars)):
        for j in range(i + 1, len(ok_cars)):
            c1 = ok_cars[i]["fragment_centroids"][0]
            c2 = ok_cars[j]["fragment_centroids"][0]
            inter_car_dists.append(np.linalg.norm(c1 - c2))

    return results, inter_car_dists


def run_hdbscan_sweep(bin_paths, label_paths, cfg, min_cluster_sizes):
    """Test different min_cluster_size values and count split/ok/merged."""
    from pipeline import remove_ground, _cluster_hdbscan

    all_results = {mcs: {"ok": 0, "split": 0, "total": 0} for mcs in min_cluster_sizes}

    for fi in range(len(bin_paths)):
        points = np.fromfile(bin_paths[fi], dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        raw_labels = np.fromfile(label_paths[fi], dtype=np.uint32)
        sem = raw_labels & 0xFFFF
        inst = raw_labels >> 16

        z_mask = xyz[:, 2] > cfg["z_threshold"]
        xyz_f = xyz[z_mask]
        sem_f = sem[z_mask]
        inst_f = inst[z_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_f)
        pcd_dn, ind = pcd.remove_statistical_outlier(
            nb_neighbors=cfg["denoise_nb_neighbors"],
            std_ratio=cfg["denoise_std_ratio"],
        )
        sem_dn = sem_f[ind]
        inst_dn = inst_f[ind]

        pcd_down = pcd_dn.voxel_down_sample(voxel_size=cfg["voxel_size"])
        xyz_down = np.asarray(pcd_down.points)
        tree = cKDTree(np.asarray(pcd_dn.points))
        _, nn_idx = tree.query(xyz_down)
        sem_down = sem_dn[nn_idx]
        inst_down = inst_dn[nn_idx]

        ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)
        object_mask = np.ones(len(xyz_down), dtype=bool)
        object_mask[ground_inliers] = False
        sem_obj = sem_down[object_mask]
        inst_obj = inst_down[object_mask]
        obj_points = np.asarray(objects_pcd.points)

        thing_mask = np.isin(sem_obj, list(THING_CLASSES))
        gt_instances = np.unique(inst_obj[thing_mask])
        gt_instances = gt_instances[gt_instances > 0]
        gt_info = {}
        for gi in gt_instances:
            m = (inst_obj == gi) & thing_mask
            if m.sum() >= 10:
                gt_info[gi] = m

        for mcs in min_cluster_sizes:
            test_cfg = dict(cfg)
            test_cfg["hdbscan_min_cluster_size"] = mcs
            cl = cluster_objects(objects_pcd, test_cfg)

            for gi, gt_mask in gt_info.items():
                cl_labels = cl[gt_mask]
                label_counts = Counter(cl_labels.tolist())
                label_counts.pop(-1, None)
                n_cl = len(label_counts)
                all_results[mcs]["total"] += 1
                if n_cl == 1:
                    all_results[mcs]["ok"] += 1
                elif n_cl > 1:
                    all_results[mcs]["split"] += 1

        if (fi + 1) % 10 == 0 or fi == len(bin_paths) - 1:
            print(f"  Frame {fi + 1}/{len(bin_paths)}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="00")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    cfg = dict(PIPELINE_CONFIG)
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:args.frames]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[:args.frames]
    n_frames = len(bin_paths)

    # ========== PART 1: Fragment distance analysis ==========
    print(f"=== Part 1: Fragment distance analysis (seq {args.seq}, {n_frames} frames) ===")
    all_gt = []
    all_inter_car = []

    for i in range(n_frames):
        gt_results, inter_dists = process_frame(bin_paths[i], label_paths[i], cfg)
        all_gt.extend(gt_results)
        all_inter_car.extend(inter_dists)
        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"  Frame {i + 1}/{n_frames}")

    split_cars = [g for g in all_gt if g["status"] == "split"]
    ok_cars = [g for g in all_gt if g["status"] == "ok"]

    # For split cars: compute max distance between fragment centroids
    frag_max_dists = []
    frag_mean_dists = []
    for car in split_cars:
        centroids = car["fragment_centroids"]
        if len(centroids) < 2:
            continue
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        frag_max_dists.append(max(dists))
        frag_mean_dists.append(np.mean(dists))

    print(f"\nSplit GT cars: {len(split_cars)}")
    print(f"Ok GT cars: {len(ok_cars)}")

    if frag_max_dists:
        print(f"\nFragment centroid distances (split cars):")
        print(f"  Max intra-car distance:")
        for p in [25, 50, 75, 90, 95]:
            print(f"    P{p}: {np.percentile(frag_max_dists, p):.2f} m")
        print(f"  Mean intra-car distance:")
        for p in [25, 50, 75, 90, 95]:
            print(f"    P{p}: {np.percentile(frag_mean_dists, p):.2f} m")

    if all_inter_car:
        print(f"\nInter-car distances (distinct ok cars, same frame):")
        near_inter = [d for d in all_inter_car if d < 20]
        for p in [5, 10, 25, 50]:
            print(f"    P{p}: {np.percentile(near_inter, p):.2f} m")
        print(f"  Cars within 2m: {sum(1 for d in all_inter_car if d < 2)}/{len(all_inter_car)}")
        print(f"  Cars within 3m: {sum(1 for d in all_inter_car if d < 3)}/{len(all_inter_car)}")
        print(f"  Cars within 5m: {sum(1 for d in all_inter_car if d < 5)}/{len(all_inter_car)}")

    # Distance-to-sensor analysis
    split_ranges = [g["range"] for g in split_cars]
    ok_ranges = [g["range"] for g in ok_cars]

    print(f"\n=== Part 2: Distance-to-sensor (range) analysis ===")
    print(f"\nSplit cars range (m):")
    for p in [25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(split_ranges, p):.1f}")
    print(f"OK cars range (m):")
    for p in [25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(ok_ranges, p):.1f}")

    # Range buckets
    buckets = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
    print(f"\nSplit rate by range bucket:")
    print(f"  {'Range':>10s}  {'Split':>6s}  {'OK':>6s}  {'Total':>6s}  {'Split%':>7s}")
    for lo, hi in buckets:
        s = sum(1 for g in all_gt if g["status"] == "split" and lo <= g["range"] < hi)
        o = sum(1 for g in all_gt if g["status"] == "ok" and lo <= g["range"] < hi)
        t = s + o
        pct = 100 * s / t if t > 0 else 0
        print(f"  {lo:>3d}-{hi:<3d} m    {s:>5d}   {o:>5d}   {t:>5d}   {pct:>6.1f}%")

    # ========== PART 3: min_cluster_size sweep ==========
    print(f"\n=== Part 3: min_cluster_size sweep ===")
    mcs_values = [10, 15, 20, 30, 50, 75, 100]
    sweep_results = run_hdbscan_sweep(bin_paths, label_paths, cfg, mcs_values)

    print(f"\n  {'MCS':>5s}  {'OK':>6s}  {'Split':>6s}  {'Total':>6s}  {'OK%':>6s}  {'Split%':>7s}")
    for mcs in mcs_values:
        r = sweep_results[mcs]
        t = r["total"]
        ok_pct = 100 * r["ok"] / t if t > 0 else 0
        sp_pct = 100 * r["split"] / t if t > 0 else 0
        print(f"  {mcs:>5d}  {r['ok']:>6d}  {r['split']:>6d}  {t:>6d}  {ok_pct:>5.1f}%  {sp_pct:>6.1f}%")
