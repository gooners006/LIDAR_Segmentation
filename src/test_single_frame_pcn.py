"""Quick test: PCN completion on single-frame clusters vs accumulated tracks.

Runs a few frames, picks the best single-frame cluster per track,
completes it with PCN, and renders side-by-side comparisons.
"""

import os
import sys
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from completion import PointCloudCompleter
from classifier import classify_cluster, load_classifier
from pipeline import (
    PIPELINE_CONFIG,
    cluster_objects,
    filter_clusters,
    load_calib,
    load_poses,
    preprocess_frame,
    remove_ground,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_comparison(partials, titles, suptitle, output_path):
    n = len(partials)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), subplot_kw={"projection": "3d"})
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=12)

    colors = ["#4da6ff", "#00e64d", "#ff6b6b"]
    for ax, (pts, label), color in zip(axes, zip(partials, titles), colors):
        show_pts = pts
        if len(pts) > 5000:
            idx = np.random.choice(len(pts), 5000, replace=False)
            show_pts = pts[idx]
        ax.scatter(show_pts[:, 0], show_pts[:, 1], show_pts[:, 2],
                   s=0.5, c=color, alpha=0.7)
        ax.set_title(f"{label} ({len(pts)} pts)", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcn-ckpt", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "pcn_best.pth"))
    parser.add_argument("--seq", type=str, default="00")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    seq = args.seq
    n_frames = args.frames
    pcn_ckpt = args.pcn_ckpt
    cls_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_best.pth")
    ckpt_name = os.path.splitext(os.path.basename(pcn_ckpt))[0]
    output_dir = os.path.join(PROJECT_ROOT, "output", f"single_frame_pcn_{ckpt_name}")
    os.makedirs(output_dir, exist_ok=True)

    completer = PointCloudCompleter(model_path=pcn_ckpt, seed=0)
    print(f"PCN loaded: {completer.is_loaded}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, cls_bbox_stats = load_classifier(cls_ckpt, device)

    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:n_frames]

    # Collect single-frame clusters with their class and point count
    best_clusters = {}  # class -> (points_local, n_pts, frame_idx)

    print(f"Scanning {len(bin_paths)} frames for good single-frame clusters...")
    for frame_idx, bin_path in enumerate(bin_paths):
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        pcd_down = preprocess_frame(xyz)
        ground_pcd, objects_pcd, ground_plane, _ = remove_ground(pcd_down)
        labels = cluster_objects(objects_pcd)
        clusters = filter_clusters(objects_pcd, labels, ground_plane)

        obj_pts = np.asarray(objects_pcd.points)
        for bbox, cluster_label in clusters:
            mask = labels == cluster_label
            cluster_points = obj_pts[mask]

            result = classify_cluster(
                cluster_points, cls_model, device, cls_bbox_stats,
                unknown_threshold=0.50,
            )
            if result.label == "not-car":
                continue

            n_pts = len(cluster_points)
            key = result.label
            if key not in best_clusters or n_pts > best_clusters[key][1]:
                best_clusters[key] = (cluster_points.copy(), n_pts, frame_idx)

    print(f"\nBest single-frame clusters found:")
    for cls, (pts, n, fidx) in sorted(best_clusters.items()):
        print(f"  {cls}: {n} pts (frame {fidx})")

    # Complete each and render
    n_rendered = 0
    for cls, (pts, n_pts, frame_idx) in sorted(best_clusters.items()):
        # Center for visualization
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        completed, skip_reason = completer.complete(pts, cls)
        if skip_reason:
            print(f"  {cls}: skipped ({skip_reason})")
            continue

        completed_centered = completed - centroid

        suptitle = f"Single-frame PCN | {cls} | frame {frame_idx} | {n_pts} input pts"
        out_path = os.path.join(output_dir, f"{cls}_frame{frame_idx}.png")

        render_comparison(
            [pts_centered, completed_centered],
            ["Raw single frame", "PCN completed"],
            suptitle,
            out_path,
        )
        print(f"  Saved: {out_path}")
        n_rendered += 1

    print(f"\n{n_rendered} comparisons saved to {output_dir}")


if __name__ == "__main__":
    main()
