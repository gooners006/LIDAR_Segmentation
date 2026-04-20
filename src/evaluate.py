"""Evaluate detection (segmentation) quality against SemanticKITTI ground truth labels.

Compares HDBSCAN clusters against GT instance segmentation using point-level IoU.
Prints per-frame and aggregate precision, recall, and F1.
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline import PIPELINE_CONFIG, cluster_objects, filter_clusters, remove_ground

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SemanticKITTI "thing" classes (objects with instance labels)
THING_CLASSES = {
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    252, 253, 254, 255, 256, 257, 258, 259,
}


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return intersection / union if union > 0 else 0.0


def match_detections_to_gt(det_masks: dict, gt_masks: dict, iou_thresh: float):
    """Greedy IoU matching. Returns: tp, fp, fn, list of match IoUs."""
    matched_gt: set = set()
    matched_det: set = set()
    match_ious: list[float] = []

    pairs = []
    for det_id, det_m in det_masks.items():
        for gt_id, gt_m in gt_masks.items():
            iou = compute_iou(det_m, gt_m)
            if iou >= iou_thresh:
                pairs.append((iou, det_id, gt_id))

    pairs.sort(reverse=True)

    for iou, det_id, gt_id in pairs:
        if det_id in matched_det or gt_id in matched_gt:
            continue
        matched_det.add(det_id)
        matched_gt.add(gt_id)
        match_ious.append(iou)

    tp = len(match_ious)
    fp = len(det_masks) - tp
    fn = len(gt_masks) - tp

    return tp, fp, fn, match_ious


def evaluate_frame(bin_path: str, label_path: str, iou_threshold: float):
    """Run the detection pipeline on one frame and compare to GT."""
    cfg = PIPELINE_CONFIG

    # --- 1. Load data ---
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem = raw_labels & 0xFFFF
    inst = raw_labels >> 16

    # --- 2. Preprocessing with label propagation ---
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

    # --- 3-5. Shared pipeline: ground removal, clustering, filtering ---
    ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)

    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[ground_inliers] = False

    sem_obj = sem_down[object_mask]
    inst_obj = inst_down[object_mask]

    cluster_labels = cluster_objects(objects_pcd)
    clusters = filter_clusters(objects_pcd, cluster_labels, ground_plane)

    # --- 6. Prepare masks for evaluation ---
    valid_cluster_labels = [cl for _, cl in clusters]
    det_masks = {cl: (cluster_labels == cl) for cl in valid_cluster_labels}

    thing_mask = np.isin(sem_obj, list(THING_CLASSES))
    gt_instances = np.unique(inst_obj[thing_mask])
    gt_instances = gt_instances[gt_instances > 0]

    gt_masks = {}
    for gi in gt_instances:
        m = (inst_obj == gi) & thing_mask
        if m.sum() >= 10:
            gt_masks[gi] = m

    # --- 7. Match & evaluate ---
    return match_detections_to_gt(det_masks, gt_masks, iou_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detection against GT")
    parser.add_argument("--seq", default="00", help="Sequence ID")
    parser.add_argument("--frames", type=int, default=100, help="Max frames")
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3, help="IoU threshold"
    )
    args = parser.parse_args()

    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[
        : args.frames
    ]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[
        : args.frames
    ]

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious: list[float] = []

    num_frames = len(bin_paths)
    print(f"Evaluating detection on {num_frames} frames (sequence {args.seq})...")
    print(f"IoU threshold: {args.iou_threshold}")
    print("-" * 80)

    for i in range(num_frames):
        tp, fp, fn, ious = evaluate_frame(
            bin_paths[i], label_paths[i], args.iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        iou_str = f"{np.mean(ious):.2f}" if ious else "N/A "
        print(
            f"Frame {i:3d}: TP={tp:2d}  FP={fp:2d}  FN={fn:2d}  "
            f"Prec={prec:.2f}  Rec={rec:.2f}  F1={f1:.2f}  meanIoU={iou_str}"
        )

    print("-" * 80)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mean_iou = np.mean(all_ious) if all_ious else 0

    print(f"AGGREGATE over {num_frames} frames:")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Mean IoU:  {mean_iou:.3f}")
