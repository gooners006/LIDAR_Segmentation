"""Per-stage inference-time benchmark of the full pipeline (advisor request).

Mirrors the production path (evaluate.py get_frame_detections + tracker) stage
by stage on seq 08, plus PCN completion timing on car-classified clusters.
Wall-clock via time.perf_counter; GPU transfers back to CPU act as sync points,
so classifier/completion timings include the full round trip.

Usage:
    .venv\\Scripts\\python.exe scratchpad/timing_benchmark.py --seq 08 --frames 100
"""

import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from classifier import classify_clusters_batch, load_classifier
from completion import PointCloudCompleter
from pipeline import (
    PIPELINE_CONFIG,
    cluster_objects,
    filter_clusters,
    load_calib,
    load_poses,
    remove_ground,
)
from tracker import CentroidTracker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = [
    "load",
    "z_filter",
    "denoise",
    "voxel_downsample",
    "ransac_ground",
    "hdbscan",
    "geometric_filter",
    "classifier",
    "tracker",
]


class _CentroidProxy:
    def __init__(self, center):
        self._center = center

    def get_center(self):
        return self._center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1,
                    help="Sample every Nth frame (spread over the sequence "
                         "instead of the contiguous start)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Frames excluded from stats (JIT/CUDA warmup)")
    ap.add_argument("--tag", default="",
                    help="Suffix appended to output JSON name (avoids clobbering)")
    ap.add_argument("--voxel-before-denoise", action="store_true",
                    help="Tier-2: voxel-downsample before statistical denoise "
                         "(denoise runs on the ~10x smaller cloud)")
    ap.add_argument("--ransac-iterations", type=int, default=None,
                    help="Tier-2: override PIPELINE_CONFIG ransac_iterations")
    ap.add_argument("--cluster-voxel-size", type=float, default=None,
                    help="Tier-3: cluster on a coarser voxel grid, propagate "
                         "labels back by nearest neighbour")
    args = ap.parse_args()

    cfg = PIPELINE_CONFIG
    if args.voxel_before_denoise:
        cfg["voxel_before_denoise"] = True
    if args.ransac_iterations is not None:
        cfg["ransac_iterations"] = args.ransac_iterations
    if args.cluster_voxel_size is not None:
        cfg["cluster_voxel_size"] = args.cluster_voxel_size
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
    bin_paths = bin_paths[:: args.stride][: args.frames]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, cls_bbox_stats = load_classifier(
        os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth"), device)
    completer = PointCloudCompleter(
        model_path=os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth"))

    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    calib = load_calib(os.path.join(seq_dir, "calib.txt"))
    Tr = calib["Tr"]
    tracker = CentroidTracker(
        max_distance=cfg["tracker_max_distance"],
        max_disappeared=cfg["tracker_max_disappeared"],
    )

    frame_times = {s: [] for s in STAGES}
    n_clusters_classified = []
    completion_times = []      # per successful completion (gate passed)
    completion_gate_times = [] # per gated (rejected) attempt
    per_frame_total = []

    for i, bp in enumerate(bin_paths):
        t = {}

        t0 = time.perf_counter()
        points = np.fromfile(bp, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        t["load"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        xyz_f = xyz[xyz[:, 2] > cfg["z_threshold"]]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_f)
        t["z_filter"] = time.perf_counter() - t0

        if cfg.get("voxel_before_denoise", False):
            # Tier-2 order: voxel first, denoise the smaller cloud.
            t0 = time.perf_counter()
            pcd_vx = pcd.voxel_down_sample(voxel_size=cfg["voxel_size"])
            t["voxel_downsample"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            pcd_down, _ = pcd_vx.remove_statistical_outlier(
                nb_neighbors=cfg["denoise_nb_neighbors"],
                std_ratio=cfg["denoise_std_ratio"])
            t["denoise"] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            pcd_dn, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg["denoise_nb_neighbors"],
                std_ratio=cfg["denoise_std_ratio"])
            t["denoise"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            pcd_down = pcd_dn.voxel_down_sample(voxel_size=cfg["voxel_size"])
            t["voxel_downsample"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, objects_pcd, ground_plane, _ = remove_ground(pcd_down)
        t["ransac_ground"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cluster_labels = cluster_objects(objects_pcd)
        t["hdbscan"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        clusters = filter_clusters(objects_pcd, cluster_labels, ground_plane)
        t["geometric_filter"] = time.perf_counter() - t0

        obj_points = np.asarray(objects_pcd.points)
        car_clusters = []
        bboxes = [bbox for bbox, _ in clusters]
        t0 = time.perf_counter()
        cluster_points_list = [obj_points[cluster_labels == cl] for _, cl in clusters]
        results = classify_clusters_batch(cluster_points_list, cls_model, device,
                                          cls_bbox_stats, unknown_threshold=0.50)
        t["classifier"] = time.perf_counter() - t0
        for cpts, res in zip(cluster_points_list, results):
            if res.label == "car":
                car_clusters.append(cpts)
        n_clusters_classified.append(len(clusters))

        frame_id = int(os.path.basename(bp)[:6])
        t0 = time.perf_counter()
        T_total = poses[frame_id] @ Tr
        R, tvec = T_total[:3, :3], T_total[:3, 3]
        proxies = [_CentroidProxy(R @ np.asarray(b.get_center()) + tvec) for b in bboxes]
        tracker.update(proxies)
        t["tracker"] = time.perf_counter() - t0

        if i >= args.warmup:
            for s in STAGES:
                frame_times[s].append(t[s])
            per_frame_total.append(sum(t.values()))

        # Completion timing (separate accounting: per-object, not per-frame)
        for cpts in car_clusters:
            t0 = time.perf_counter()
            _, skip = completer.complete(cpts, "car")
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                (completion_gate_times if skip else completion_times).append(dt)

        if (i + 1) % 20 == 0:
            print(f"  frame {i + 1}/{len(bin_paths)}", flush=True)

        if (i + 1) % 250 == 0:
            ck_dir = os.path.join(PROJECT_ROOT, "output", "experiments", "timing")
            os.makedirs(ck_dir, exist_ok=True)
            with open(os.path.join(
                    ck_dir, f"timing_seq{args.seq}_partial.json"), "w") as f:
                json.dump({"frames_done": i + 1,
                           "stages_s": {s: v for s, v in frame_times.items()},
                           "completion_s": completion_times,
                           "gated_s": completion_gate_times}, f)

    def ms(x):
        return float(np.mean(x) * 1000), float(np.median(x) * 1000)

    print("\n=== Hardware ===")
    print("CPU:", platform.processor())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    n = len(per_frame_total)
    print(f"\n=== Detection pipeline, per frame (seq {args.seq}, n={n} frames, "
          f"{args.warmup} warmup excluded) ===")
    print(f"{'stage':20s} {'mean ms':>9s} {'median ms':>10s} {'share':>7s}")
    total_mean = np.mean(per_frame_total) * 1000
    results = {}
    for s in STAGES:
        m, med = ms(frame_times[s])
        results[s] = {"mean_ms": m, "median_ms": med}
        print(f"{s:20s} {m:9.1f} {med:10.1f} {m / total_mean:6.1%}")
    print(f"{'TOTAL':20s} {total_mean:9.1f} "
          f"{np.median(per_frame_total) * 1000:10.1f}")
    print(f"Effective rate: {1000 / total_mean:.2f} frames/s "
          f"(sensor rate: 10 frames/s)")
    print(f"Mean clusters classified per frame: {np.mean(n_clusters_classified):.1f}")

    print("\n=== Completion (per car-classified cluster) ===")
    if completion_times:
        m, med = ms(completion_times)
        print(f"gate passed + PCN:   mean {m:.1f} ms  median {med:.1f} ms  "
              f"(n={len(completion_times)})")
    if completion_gate_times:
        m, med = ms(completion_gate_times)
        print(f"gate rejected:       mean {m:.1f} ms  median {med:.1f} ms  "
              f"(n={len(completion_gate_times)})")

    out = {
        "seq": args.seq, "frames": n, "warmup": args.warmup,
        "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stages_ms": results,
        "total_mean_ms": total_mean,
        "total_median_ms": float(np.median(per_frame_total) * 1000),
        "mean_clusters_per_frame": float(np.mean(n_clusters_classified)),
        "completion_ms": {
            "completed_mean": float(np.mean(completion_times) * 1000) if completion_times else None,
            "completed_median": float(np.median(completion_times) * 1000) if completion_times else None,
            "completed_n": len(completion_times),
            "gated_mean": float(np.mean(completion_gate_times) * 1000) if completion_gate_times else None,
            "gated_n": len(completion_gate_times),
        },
    }
    out_dir = os.path.join(PROJECT_ROOT, "output", "experiments", "timing")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_stride{args.stride}" if args.stride > 1 else ""
    if args.stride == 1 and n > 100:
        suffix = f"_full_n{n}"
    if args.tag:
        suffix += f"_{args.tag}"
    out_path = os.path.join(out_dir, f"timing_seq{args.seq}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
