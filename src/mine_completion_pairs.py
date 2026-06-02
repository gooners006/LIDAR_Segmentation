"""Mine sparse/dense completion pairs using GT SemanticKITTI labels.

Runs the pipeline with tracking (like main.py) but uses GT semantic labels
(like mine_stage_b.py) instead of the learned classifier to determine cluster
class. This eliminates classifier noise from the mined pairs.

Usage:
    python src/mine_completion_pairs.py --seq 00 01 02 --frames 99999
    python src/mine_completion_pairs.py --seq 08 --frames 99999 --output data/real_pairs/val
"""

import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

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

SEMKITTI_TO_CLASS = {
    10: "car", 252: "car",
    13: "bus", 257: "bus",
    15: "motorcycle", 255: "motorcycle",
}

VALID_CLASSES = {"car", "bus", "motorcycle"}


def classify_cluster_gt(sem_labels: np.ndarray, purity_threshold: float):
    """Determine class from GT semantic labels by majority vote."""
    unique, counts = np.unique(sem_labels, return_counts=True)
    total = counts.sum()

    class_votes = {}
    for label, count in zip(unique, counts):
        cls = SEMKITTI_TO_CLASS.get(int(label), "unknown")
        class_votes[cls] = class_votes.get(cls, 0) + int(count)

    majority_class = max(class_votes, key=class_votes.get)
    purity = class_votes[majority_class] / total

    if majority_class == "unknown" or purity < purity_threshold:
        return None, purity
    return majority_class, purity


def process_frame(bin_path, label_path, purity_threshold):
    """Run pipeline stages 1-5 with GT label propagation.

    Returns (objects_pcd, cluster_list, cluster_gt_classes) where
    cluster_list is [(bbox, cluster_label), ...] and cluster_gt_classes
    is the GT class for each (None if rejected).
    """
    cfg = PIPELINE_CONFIG

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem = raw_labels & 0xFFFF

    # Stage 1: Z-filter
    z_mask = xyz[:, 2] > cfg["z_threshold"]
    xyz_filtered = xyz[z_mask]
    sem_filtered = sem[z_mask]

    if len(xyz_filtered) == 0:
        return None, [], []

    # Stage 2: Denoise
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
    pcd_denoised, ind_denoise = pcd.remove_statistical_outlier(
        nb_neighbors=cfg["denoise_nb_neighbors"],
        std_ratio=cfg["denoise_std_ratio"],
    )
    sem_denoised = sem_filtered[ind_denoise]
    xyz_denoised = np.asarray(pcd_denoised.points)

    if len(xyz_denoised) == 0:
        return None, [], []

    # Stage 3: Downsample
    pcd_down = pcd_denoised.voxel_down_sample(voxel_size=cfg["voxel_size"])
    xyz_down = np.asarray(pcd_down.points)

    if len(xyz_down) == 0:
        return None, [], []

    # Propagate labels via KD-tree
    tree = cKDTree(xyz_denoised)
    _, nn_idx = tree.query(xyz_down)
    sem_down = sem_denoised[nn_idx]

    # Stages 4-5: Ground removal, clustering, filtering
    ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)

    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[ground_inliers] = False
    sem_obj = sem_down[object_mask]

    if len(sem_obj) == 0:
        return objects_pcd, [], []

    cluster_labels = cluster_objects(objects_pcd)
    clusters = filter_clusters(objects_pcd, cluster_labels, ground_plane)

    cluster_gt_classes = []
    for bbox, cl in clusters:
        mask = cluster_labels == cl
        cluster_sem = sem_obj[mask]
        cls_name, _ = classify_cluster_gt(cluster_sem, purity_threshold)
        cluster_gt_classes.append(cls_name)

    return objects_pcd, clusters, cluster_gt_classes, cluster_labels


def mine_sequence(seq, n_frames, output_dir, purity_threshold, min_track_length):
    """Mine completion pairs from one sequence using GT labels."""
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))

    n = min(n_frames, len(bin_paths), len(label_paths))
    bin_paths = bin_paths[:n]
    label_paths = label_paths[:n]

    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    calib = load_calib(os.path.join(seq_dir, "calib.txt"))
    Tr = calib["Tr"]

    tracker = CentroidTracker(
        max_distance=PIPELINE_CONFIG["tracker_max_distance"],
        max_disappeared=PIPELINE_CONFIG["tracker_max_disappeared"],
    )

    track_points: dict[int, list[np.ndarray]] = {}
    track_gt_classes: dict[int, list[str]] = {}
    track_frames: dict[int, list[int]] = {}

    print(f"  Processing {n} frames...")
    for frame_idx in range(n):
        result = process_frame(
            bin_paths[frame_idx], label_paths[frame_idx], purity_threshold
        )
        if result[0] is None:
            continue

        objects_pcd, clusters, cluster_gt_classes, cluster_labels = result

        # Transform to global frame
        T_total = poses[frame_idx] @ Tr
        R_total = T_total[:3, :3]
        t_total = T_total[:3, 3]

        bboxes = []
        bbox_cluster_labels = []
        bbox_gt_classes = []
        for (bbox, cl), gt_cls in zip(clusters, cluster_gt_classes):
            bbox.rotate(R_total, center=np.zeros(3))
            bbox.translate(t_total)
            bboxes.append(bbox)
            bbox_cluster_labels.append(cl)
            bbox_gt_classes.append(gt_cls)

        objects_pcd.transform(T_total)

        # Tracking
        tracked_objects, assignments = tracker.update(bboxes)

        obj_pts = np.asarray(objects_pcd.points)
        for bbox_idx, track_id in assignments.items():
            gt_cls = bbox_gt_classes[bbox_idx]
            if gt_cls is None:
                continue

            cl = bbox_cluster_labels[bbox_idx]
            cluster_pts = obj_pts[cluster_labels == cl]

            if track_id not in track_points:
                track_points[track_id] = []
                track_gt_classes[track_id] = []
                track_frames[track_id] = []

            track_points[track_id].append(cluster_pts)
            track_gt_classes[track_id].append(gt_cls)
            track_frames[track_id].append(frame_idx)

        if (frame_idx + 1) % 100 == 0 or frame_idx == n - 1:
            print(f"    Frame {frame_idx}: {len(tracker.objects)} active tracks")

    # Save pairs
    n_tracks = 0
    n_pairs = 0
    class_counts: dict[str, int] = {}

    for track_id, pts_list in track_points.items():
        if len(track_frames[track_id]) < min_track_length:
            continue

        # Majority vote on GT classes
        from collections import Counter
        votes = Counter(track_gt_classes[track_id])
        track_cls = votes.most_common(1)[0][0]

        cls_dir = os.path.join(output_dir, track_cls)
        os.makedirs(cls_dir, exist_ok=True)

        tag = f"s{seq}_{track_id:04d}"
        dense = np.vstack(pts_list).astype(np.float32)
        np.save(os.path.join(cls_dir, f"dense_{tag}.npy"), dense)

        frames = track_frames[track_id]
        for frame_pts, fidx in zip(pts_list, frames):
            sparse = frame_pts.astype(np.float32)
            np.save(os.path.join(cls_dir, f"sparse_{tag}_f{fidx:04d}.npy"), sparse)
            n_pairs += 1

        n_tracks += 1
        class_counts[track_cls] = class_counts.get(track_cls, 0) + 1

    return n_tracks, n_pairs, class_counts


def main():
    parser = argparse.ArgumentParser(
        description="Mine completion pairs using GT SemanticKITTI labels")
    parser.add_argument("--seq", nargs="+", default=["00"],
                        help="Sequence IDs")
    parser.add_argument("--frames", type=int, default=99999,
                        help="Max frames per sequence")
    parser.add_argument("--output", type=str, default="data/real_pairs/train",
                        help="Output directory")
    parser.add_argument("--purity", type=float, default=0.75,
                        help="GT label purity threshold per cluster")
    parser.add_argument("--min-track-length", type=int, default=3,
                        help="Minimum frames per track")
    args = parser.parse_args()

    total_tracks = 0
    total_pairs = 0
    total_class_counts: dict[str, int] = {}

    for seq in args.seq:
        print(f"\n=== Sequence {seq} ===")
        n_tracks, n_pairs, class_counts = mine_sequence(
            seq, args.frames, args.output, args.purity, args.min_track_length
        )
        total_tracks += n_tracks
        total_pairs += n_pairs
        for cls, cnt in class_counts.items():
            total_class_counts[cls] = total_class_counts.get(cls, 0) + cnt
        print(f"  => {n_tracks} tracks, {n_pairs} pairs | {class_counts}")

    print(f"\n{'='*60}")
    print(f"Total: {total_tracks} tracks, {total_pairs} pairs")
    for cls, cnt in sorted(total_class_counts.items()):
        print(f"  {cls}: {cnt} tracks")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
