"""Evaluate detection (segmentation) quality against SemanticKITTI ground truth labels.

Compares DBSCAN clusters against GT instance segmentation using point-level IoU.
Prints per-frame and aggregate precision, recall, and F1.
"""

import glob
import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---
CONFIG = {
    "seq": "00",
    "max_frames": 100,
    "z_threshold": -2.0,
    "voxel_size": 0.05,
    "ransac_distance_threshold": 0.2,
    "ransac_n": 3,
    "ransac_num_iterations": 1000,
    "dbscan_eps": 0.5,
    "dbscan_min_points": 10,
    "iou_threshold": 0.3,
    # Bounding Box Heuristics
    "min_points_in_cluster": 15,
    "min_volume": 0.2,
    "max_volume": 100.0,
    "max_dim_length": 8.0,
    "min_max_dim": 0.5,
    "min_med_dim": 0.2,
    "max_height_above_gnd": 3.0,
}

# SemanticKITTI "thing" classes (objects with instance labels)
THING_CLASSES = {
    10,
    11,
    13,
    15,
    16,
    18,
    20,
    30,
    31,
    32,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
}

# --- File Paths ---
SEQ_DIR = os.path.join(PROJECT_ROOT, f"dataset/sequences/{CONFIG['seq']}")
bin_paths = sorted(glob.glob(os.path.join(SEQ_DIR, "velodyne/*.bin")))[
    : CONFIG["max_frames"]
]
label_paths = sorted(glob.glob(os.path.join(SEQ_DIR, "labels/*.label")))[
    : CONFIG["max_frames"]
]


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two boolean masks over the same point set."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return intersection / union if union > 0 else 0.0


def match_detections_to_gt(det_masks: dict, gt_masks: dict, iou_thresh: float):
    """Greedy IoU matching between detection masks and ground truth masks.

    Returns: tp, fp, fn, list of match IoUs.
    """
    matched_gt = set()
    matched_det = set()
    match_ious = []

    # Calculate all IoUs above threshold
    pairs = []
    for det_id, det_m in det_masks.items():
        for gt_id, gt_m in gt_masks.items():
            iou = compute_iou(det_m, gt_m)
            if iou >= iou_thresh:
                pairs.append((iou, det_id, gt_id))

    # Greedy match: highest IoU first
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


def evaluate_frame(frame_idx: int):
    """Run the detection pipeline on one frame and compare to GT."""

    # --- 1. Load Data ---
    points = np.fromfile(bin_paths[frame_idx], dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    raw_labels = np.fromfile(label_paths[frame_idx], dtype=np.uint32)
    sem = raw_labels & 0xFFFF
    inst = raw_labels >> 16

    # --- 2. Z-Filter ---
    z_mask = xyz[:, 2] > CONFIG["z_threshold"]
    xyz_filtered = xyz[z_mask]
    sem_filtered = sem[z_mask]
    inst_filtered = inst[z_mask]

    # --- 3. Denoise & Downsample ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
    pcd_denoised, ind_denoise = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    # Propagate labels through denoising
    sem_denoised = sem_filtered[ind_denoise]
    inst_denoised = inst_filtered[ind_denoise]
    xyz_denoised = np.asarray(pcd_denoised.points)

    pcd_down = pcd_denoised.voxel_down_sample(voxel_size=CONFIG["voxel_size"])
    xyz_down = np.asarray(pcd_down.points)

    # Propagate labels to new voxel centroids via Nearest Neighbor
    tree = cKDTree(xyz_denoised)
    _, nn_idx = tree.query(xyz_down)
    sem_down = sem_denoised[nn_idx]
    inst_down = inst_denoised[nn_idx]

    # --- 4. Ground Segmentation (Local Frame) ---
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=CONFIG["ransac_distance_threshold"],
        ransac_n=CONFIG["ransac_n"],
        num_iterations=CONFIG["ransac_num_iterations"],
    )
    [a, b, c, d] = plane_model
    ground_normal_len = np.sqrt(a**2 + b**2 + c**2)

    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[inliers] = False
    objects_pcd = pcd_down.select_by_index(np.where(object_mask)[0])

    # Labels exclusively for non-ground points
    sem_obj = sem_down[object_mask]
    inst_obj = inst_down[object_mask]

    # --- 5. Clustering ---
    cluster_labels = np.array(
        objects_pcd.cluster_dbscan(
            eps=CONFIG["dbscan_eps"],
            min_points=CONFIG["dbscan_min_points"],
            print_progress=False,
        )
    )

    # --- 6. Bounding Box Filtering ---
    valid_cluster_labels = []

    for label in np.unique(cluster_labels):
        if label == -1:
            continue

        cluster_idx = np.asarray(cluster_labels == label).nonzero()[0]
        cluster_pcd = objects_pcd.select_by_index(cluster_idx)

        if len(cluster_pcd.points) < CONFIG["min_points_in_cluster"]:
            continue

        bbox = cluster_pcd.get_oriented_bounding_box()
        volume = bbox.volume()
        if volume < CONFIG["min_volume"] or volume > CONFIG["max_volume"]:
            continue

        sorted_ext = np.sort(bbox.extent)
        if sorted_ext[2] > CONFIG["max_dim_length"]:
            continue
        if (
            sorted_ext[2] < CONFIG["min_max_dim"]
            or sorted_ext[1] < CONFIG["min_med_dim"]
        ):
            continue

        center = bbox.get_center()
        height_above_ground = (
            a * center[0] + b * center[1] + c * center[2] + d
        ) / ground_normal_len
        if height_above_ground > CONFIG["max_height_above_gnd"]:
            continue

        valid_cluster_labels.append(label)

    # --- 7. Prepare Masks for Evaluation ---
    det_masks = {cl: (cluster_labels == cl) for cl in valid_cluster_labels}

    # Build GT instance masks (thing classes only, over object points)
    thing_mask = np.isin(sem_obj, list(THING_CLASSES))
    gt_instances = np.unique(inst_obj[thing_mask])
    gt_instances = gt_instances[gt_instances > 0]  # Skip 0 (unlabeled/background)

    gt_masks = {}
    for gi in gt_instances:
        m = (inst_obj == gi) & thing_mask
        if m.sum() >= 10:  # Ignore microscopic/residual GT instances
            gt_masks[gi] = m

    # --- 8. Match & Evaluate ---
    return match_detections_to_gt(det_masks, gt_masks, CONFIG["iou_threshold"])


if __name__ == "__main__":
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    num_frames = len(bin_paths)
    print(f"Evaluating detection on {num_frames} frames (sequence {CONFIG['seq']})...")
    print(f"IoU threshold: {CONFIG['iou_threshold']}")
    print("-" * 80)

    for i in range(num_frames):
        tp, fp, fn, ious = evaluate_frame(i)

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
