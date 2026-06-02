"""Evaluate detection (segmentation) quality against SemanticKITTI ground truth labels.

Compares HDBSCAN clusters against GT instance segmentation using point-level IoU.
Prints per-frame and aggregate precision, recall, and F1.

Supports optional track-level filtering (offline/post-hoc): when enabled, a
centroid tracker links detections across frames, and only tracks meeting
minimum length and class-consistency requirements are accepted.
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from classifier import classify_cluster, load_classifier
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

# SemanticKITTI "thing" classes (objects with instance labels)
THING_CLASSES_ALL = {
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    252, 253, 254, 255, 256, 257, 258, 259,
}

# Only classes the classifier can recognize (car + moving-car)
THING_CLASSES_SUPPORTED = {
    10, 252,          # car, moving-car
}

TARGET_MODES = {
    "all-things": THING_CLASSES_ALL,
    "supported-vehicles": THING_CLASSES_SUPPORTED,
}


def resolve_track_class(votes, *, min_known_votes=2, min_known_ratio=0.5):
    """Majority vote over non-unknown labels with evidence thresholds.

    Returns the winning class name, or None if evidence is insufficient
    (too few known votes, low known ratio, or ambiguous tie).
    """
    known = [v for v in votes if v != "not-car"]
    if len(known) < min_known_votes:
        return None
    if len(known) / len(votes) < min_known_ratio:
        return None
    counts: dict[str, int] = {}
    for v in known:
        counts[v] = counts.get(v, 0) + 1
    top_count = max(counts.values())
    winners = [cls for cls, c in counts.items() if c == top_count]
    if len(winners) > 1:
        return None
    return winners[0]


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two boolean masks.

    Args:
        mask_a: Boolean array of points belonging to detection/instance A.
        mask_b: Boolean array of points belonging to detection/instance B.

    Returns:
        IoU value in [0, 1]. Returns 0 if the union is empty.
    """
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return intersection / union if union > 0 else 0.0


def match_detections_to_gt(det_masks: dict, gt_masks: dict, iou_thresh: float):
    """Greedy bipartite matching between detections and GT instances by IoU.

    All (detection, GT) pairs exceeding ``iou_thresh`` are sorted by
    descending IoU and greedily assigned — once a detection or GT
    instance is matched it cannot be reused.

    Args:
        det_masks: Dict mapping detection ID to a boolean point mask.
        gt_masks: Dict mapping GT instance ID to a boolean point mask.
        iou_thresh: Minimum IoU for a valid match.

    Returns:
        Tuple of (tp, fp, fn, match_ious) where *tp* is the number of
        matched pairs, *fp* is unmatched detections, *fn* is unmatched
        GT instances, and *match_ious* is the list of IoU values for
        each accepted match.
    """
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


class _CentroidProxy:
    """Minimal wrapper so CentroidTracker.update() can call get_center()."""

    def __init__(self, center: np.ndarray):
        self._center = center

    def get_center(self) -> np.ndarray:
        return self._center


def get_frame_detections(bin_path: str, label_path: str,
                         cls_model=None, cls_device=None, cls_bbox_stats=None,
                         unknown_threshold: float = 0.50,
                         thing_classes: set | None = None,
                         keep_unknown: bool = False):
    """Run detection pipeline for one frame and return raw detections + GT.

    Args:
        bin_path: Path to the Velodyne ``.bin`` point cloud file.
        label_path: Path to the SemanticKITTI ``.label`` file.
        cls_model: Optional loaded PointNetClassifier.
        cls_device: Torch device for classifier inference.
        cls_bbox_stats: Bbox normalization stats dict (mean/std).
        unknown_threshold: Confidence threshold for unknown rejection.
        thing_classes: Set of SemanticKITTI class IDs for GT objects.
        keep_unknown: If True, keep unknown-classified detections (for
            track-level filtering). If False, filter them per-frame.

    Returns:
        Tuple of (bboxes, det_cluster_ids, det_classes, cluster_labels,
        objects_pcd, det_masks, gt_masks).
    """
    if thing_classes is None:
        thing_classes = THING_CLASSES_ALL
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

    # --- 6. Classification (optional) ---
    bboxes = []
    det_cluster_ids = []
    det_classes = []

    if cls_model is not None:
        obj_points = np.asarray(objects_pcd.points)
        for bbox, cl in clusters:
            mask = cluster_labels == cl
            cluster_points = obj_points[mask]
            result = classify_cluster(
                cluster_points, cls_model, cls_device, cls_bbox_stats,
                unknown_threshold=unknown_threshold,
            )
            if keep_unknown or result.label != "not-car":
                bboxes.append(bbox)
                det_cluster_ids.append(cl)
                det_classes.append(result.label)
    else:
        for bbox, cl in clusters:
            bboxes.append(bbox)
            det_cluster_ids.append(cl)
            det_classes.append("car")

    # --- 6b. Prepare masks ---
    det_masks = {cl: (cluster_labels == cl) for cl in det_cluster_ids}

    thing_mask = np.isin(sem_obj, list(thing_classes))
    gt_instances = np.unique(inst_obj[thing_mask])
    gt_instances = gt_instances[gt_instances > 0]

    gt_masks = {}
    for gi in gt_instances:
        m = (inst_obj == gi) & thing_mask
        if m.sum() >= 10:
            gt_masks[gi] = m

    return bboxes, det_cluster_ids, det_classes, cluster_labels, objects_pcd, det_masks, gt_masks


def evaluate_frame(bin_path: str, label_path: str, iou_threshold: float,
                   cls_model=None, cls_device=None, cls_bbox_stats=None,
                   unknown_threshold: float = 0.50,
                   thing_classes: set | None = None):
    """Run the full detection pipeline on one frame and evaluate against GT.

    Thin wrapper around get_frame_detections + match_detections_to_gt.
    Used when track-level filtering is disabled.

    Args:
        bin_path: Path to the Velodyne ``.bin`` point cloud file.
        label_path: Path to the SemanticKITTI ``.label`` file.
        iou_threshold: Minimum IoU for a detection–GT match.
        cls_model: Optional loaded PointNetClassifier for filtering.
        cls_device: Torch device for classifier inference.
        cls_bbox_stats: Bbox normalization stats dict (mean/std).
        unknown_threshold: Confidence threshold below which the classifier
            rejects a cluster as unknown.
        thing_classes: Set of SemanticKITTI class IDs to treat as valid GT
            objects.  Defaults to ``THING_CLASSES_ALL``.

    Returns:
        Tuple of (tp, fp, fn, match_ious) from greedy IoU matching.
    """
    _, _, _, _, _, det_masks, gt_masks = get_frame_detections(
        bin_path, label_path,
        cls_model=cls_model, cls_device=cls_device,
        cls_bbox_stats=cls_bbox_stats,
        unknown_threshold=unknown_threshold,
        thing_classes=thing_classes,
        keep_unknown=False,
    )
    return match_detections_to_gt(det_masks, gt_masks, iou_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detection against GT")
    parser.add_argument("--seq", default="00", help="Sequence ID")
    parser.add_argument("--frames", type=int, default=100, help="Max frames")
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3, help="IoU threshold"
    )
    parser.add_argument(
        "--classifier-ckpt", type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "classifier_best.pth"),
        help="Path to classifier checkpoint",
    )
    parser.add_argument(
        "--classifier-unknown-threshold", type=float, default=0.50,
        help="Unknown class probability threshold",
    )
    parser.add_argument(
        "--no-learned-classifier", action="store_true",
        help="Disable learned classifier (geometric filters only)",
    )
    parser.add_argument(
        "--target", type=str, default="all-things",
        choices=list(TARGET_MODES.keys()),
        help="GT target classes: all-things (default) or supported-vehicles "
             "(car/bus/motorcycle only — use with classifier)",
    )
    parser.add_argument(
        "--no-track-filter", action="store_true",
        help="Disable track-level filtering (per-frame evaluation only)",
    )
    parser.add_argument(
        "--min-track-length", type=int, default=None,
        help="Override min_track_length from PIPELINE_CONFIG (for sweeps)",
    )
    args = parser.parse_args()

    thing_classes = TARGET_MODES[args.target]

    # Load classifier
    cls_model, cls_bbox_stats, cls_device = None, None, None
    if not args.no_learned_classifier:
        cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls_model, cls_bbox_stats = load_classifier(args.classifier_ckpt, cls_device)
        if cls_model is not None:
            print(f"Loaded classifier from {args.classifier_ckpt}")
        else:
            print(f"Classifier not found: {args.classifier_ckpt}. "
                  "Running without classifier.")
    else:
        print("Classifier disabled (geometric filters only).")

    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[
        : args.frames
    ]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[
        : args.frames
    ]

    if len(bin_paths) != len(label_paths):
        raise RuntimeError(
            f"Mismatched velodyne/label files: {len(bin_paths)} bins, "
            f"{len(label_paths)} labels")

    num_frames = len(bin_paths)
    cfg = PIPELINE_CONFIG
    if args.min_track_length is not None:
        cfg["min_track_length"] = args.min_track_length
    use_track_filter = (
        not args.no_track_filter
        and cfg["min_track_length"] > 0
        and cls_model is not None
    )

    print(f"Evaluating detection on {num_frames} frames (sequence {args.seq})...")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Target classes: {args.target}")
    if use_track_filter:
        print(f"Track-level filtering: ON (offline/post-hoc, "
              f"min_length={cfg['min_track_length']}, "
              f"min_known_votes={cfg['min_track_known_votes']}, "
              f"min_known_ratio={cfg['min_track_known_ratio']})")
    else:
        print("Track-level filtering: OFF (per-frame evaluation)")
    print("-" * 80)

    if use_track_filter:
        # ===== Two-pass evaluation with track-level filtering =====
        poses = load_poses(os.path.join(seq_dir, "poses.txt"))
        calib = load_calib(os.path.join(seq_dir, "calib.txt"))
        Tr = calib["Tr"]

        tracker = CentroidTracker(
            max_distance=cfg["tracker_max_distance"],
            max_disappeared=cfg["tracker_max_disappeared"],
        )

        # Per-frame storage
        frame_assignments: list[dict[int, int]] = []
        frame_det_cluster_ids: list[list] = []
        frame_det_masks: list[dict] = []
        frame_gt_masks: list[dict] = []

        # Track-level accumulation
        track_class_votes: dict[int, list[str]] = {}
        track_frame_count: dict[int, int] = {}

        # --- Pass 1: accumulate tracks ---
        print("Pass 1: detecting and tracking...")
        for i in range(num_frames):
            bboxes, det_ids, det_classes, _, _, det_masks, gt_masks = \
                get_frame_detections(
                    bin_paths[i], label_paths[i],
                    cls_model=cls_model, cls_device=cls_device,
                    cls_bbox_stats=cls_bbox_stats,
                    unknown_threshold=args.classifier_unknown_threshold,
                    thing_classes=thing_classes,
                    keep_unknown=True,
                )

            # Transform centroids to global frame for tracker
            T_total = poses[i] @ Tr
            R = T_total[:3, :3]
            t = T_total[:3, 3]
            global_proxies = []
            for bbox in bboxes:
                c = np.asarray(bbox.get_center())
                global_c = R @ c + t
                global_proxies.append(_CentroidProxy(global_c))

            _, assignments = tracker.update(global_proxies)

            # Accumulate track info
            for det_idx, track_id in assignments.items():
                if track_id not in track_class_votes:
                    track_class_votes[track_id] = []
                    track_frame_count[track_id] = 0
                track_class_votes[track_id].append(det_classes[det_idx])
                track_frame_count[track_id] += 1

            frame_assignments.append(assignments)
            frame_det_cluster_ids.append(det_ids)
            frame_det_masks.append(det_masks)
            frame_gt_masks.append(gt_masks)

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"  Frame {i:3d}: {len(bboxes)} detections, "
                      f"{len(tracker.objects)} active tracks")

        # --- Determine surviving tracks ---
        min_len = cfg["min_track_length"]
        min_kv = cfg["min_track_known_votes"]
        min_kr = cfg["min_track_known_ratio"]

        n_total = len(track_class_votes)
        n_short = 0
        n_rejected_class = 0
        valid_tracks: set[int] = set()
        accepted_lengths: list[int] = []

        for tid, votes in track_class_votes.items():
            if track_frame_count[tid] < min_len:
                n_short += 1
                continue
            resolved = resolve_track_class(
                votes, min_known_votes=min_kv, min_known_ratio=min_kr)
            if resolved is None:
                n_rejected_class += 1
                continue
            valid_tracks.add(tid)
            accepted_lengths.append(track_frame_count[tid])

        print("-" * 80)
        print(f"Track-level filtering (offline/post-hoc):")
        print(f"  Total tracks: {n_total}")
        print(f"  Accepted: {len(valid_tracks)}")
        print(f"  Rejected (too short): {n_short}")
        print(f"  Rejected (class vote failed): {n_rejected_class}")
        if accepted_lengths:
            print(f"  Mean track length (accepted): "
                  f"{np.mean(accepted_lengths):.1f} frames")
            print(f"  Median track length (accepted): "
                  f"{np.median(accepted_lengths):.0f} frames")
        print("-" * 80)

        # --- Pass 2: evaluate with track filter ---
        print("Pass 2: evaluating with track filter...")
        total_tp, total_fp, total_fn = 0, 0, 0
        all_ious: list[float] = []

        for i in range(num_frames):
            assignments = frame_assignments[i]
            det_ids = frame_det_cluster_ids[i]
            det_masks = frame_det_masks[i]
            gt_masks = frame_gt_masks[i]

            # Keep only detections belonging to valid tracks
            filtered_det_masks = {}
            for det_idx, track_id in assignments.items():
                if track_id in valid_tracks:
                    cl = det_ids[det_idx]
                    if cl in det_masks:
                        filtered_det_masks[cl] = det_masks[cl]

            tp, fp, fn, ious = match_detections_to_gt(
                filtered_det_masks, gt_masks, args.iou_threshold)

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

    else:
        # ===== Original per-frame evaluation (no tracking) =====
        total_tp, total_fp, total_fn = 0, 0, 0
        all_ious: list[float] = []

        for i in range(num_frames):
            tp, fp, fn, ious = evaluate_frame(
                bin_paths[i], label_paths[i], args.iou_threshold,
                cls_model=cls_model, cls_device=cls_device,
                cls_bbox_stats=cls_bbox_stats,
                unknown_threshold=args.classifier_unknown_threshold,
                thing_classes=thing_classes,
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
