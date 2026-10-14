"""Evaluate detection (segmentation) quality against SemanticKITTI ground truth labels.

Compares HDBSCAN clusters against GT instance segmentation using point-level IoU.
Prints per-frame and aggregate precision, recall, and F1.

Supports optional track-level filtering (offline/post-hoc): when enabled, a
centroid tracker links detections across frames, and only tracks meeting
minimum length and class-consistency requirements are accepted.
"""

import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from classifier import classify_clusters_batch, load_classifier
from pipeline import (
    PIPELINE_CONFIG,
    cluster_objects,
    filter_clusters,
    load_calib,
    load_poses,
    remove_ground,
)
from tracker import CentroidTracker, resolve_track_class

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

    if cfg.get("voxel_before_denoise", False):
        # Perf Tier 2: voxel first, then denoise the ~10x smaller cloud.
        # Behavior-changing (surviving point set differs). Labels propagate
        # from the nearest z-filtered original point to each surviving point.
        pcd_voxel = pcd.voxel_down_sample(voxel_size=cfg["voxel_size"])
        pcd_down, _ = pcd_voxel.remove_statistical_outlier(
            nb_neighbors=cfg["denoise_nb_neighbors"],
            std_ratio=cfg["denoise_std_ratio"],
        )
        xyz_down = np.asarray(pcd_down.points)
        tree = cKDTree(xyz_filtered)
        _, nn_idx = tree.query(xyz_down)
        sem_down = sem_filtered[nn_idx]
        inst_down = inst_filtered[nn_idx]
    else:
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
        cluster_points_list = [obj_points[cluster_labels == cl] for _, cl in clusters]
        results = classify_clusters_batch(
            cluster_points_list, cls_model, cls_device, cls_bbox_stats,
            unknown_threshold=unknown_threshold,
        )
        for (bbox, cl), result in zip(clusters, results):
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
        default=os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth"),
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
        "--target", type=str, default="supported-vehicles",
        choices=list(TARGET_MODES.keys()),
        help="GT target classes: supported-vehicles (default; car/bus/motorcycle "
             "only — use with classifier) or all-things",
    )
    parser.add_argument(
        "--no-track-filter", action="store_true",
        help="Disable track-level filtering (per-frame evaluation only)",
    )
    parser.add_argument(
        "--min-track-length", type=int, default=None,
        help="Override min_track_length from PIPELINE_CONFIG (for sweeps)",
    )
    parser.add_argument(
        "--min-known-votes", type=int, default=None,
        help="Override min_track_known_votes (for sweeps)",
    )
    parser.add_argument(
        "--min-known-ratio", type=float, default=None,
        help="Override min_track_known_ratio (for sweeps)",
    )
    parser.add_argument(
        "--min-points", type=int, default=None,
        help="Override min_points_in_cluster (for sweeps)",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size", type=int, default=None,
        help="Override hdbscan_min_cluster_size (for sweeps)",
    )
    parser.add_argument(
        "--tracker-max-distance", type=float, default=None,
        help="Override tracker_max_distance (for sweeps)",
    )
    parser.add_argument(
        "--tracker-max-disappeared", type=int, default=None,
        help="Override tracker_max_disappeared (for sweeps)",
    )
    parser.add_argument(
        "--ransac-distance-threshold", type=float, default=None,
        help="Override ransac_distance_threshold (for sweeps)",
    )
    parser.add_argument(
        "--ransac-iterations", type=int, default=None,
        help="Override ransac_iterations (perf Tier 2; PIPELINE_CONFIG default 300)",
    )
    parser.add_argument(
        "--voxel-before-denoise", action="store_true",
        help="Perf Tier 2: voxel-downsample before denoise (behavior-changing). "
             "No-op: PIPELINE_CONFIG default is already True; kept for symmetry "
             "with --no-voxel-before-denoise.",
    )
    parser.add_argument(
        "--no-voxel-before-denoise", action="store_true",
        help="Perf Tier 2: reach pre-#34 configs by denoising before voxel "
             "downsampling (PIPELINE_CONFIG default is voxel-before-denoise=True)",
    )
    parser.add_argument(
        "--cluster-voxel-size", type=float, default=None,
        help="Perf Tier 3: cluster on a coarser voxel grid then propagate "
             "labels back by nearest neighbour (behavior-changing; e.g. 0.10)",
    )
    parser.add_argument(
        "--clustering-method", type=str, default=None,
        choices=["hdbscan", "bev", "dbscan", "euclidean"],
        help="Override clustering method (hdbscan, bev, dbscan, or euclidean; "
             "dbscan/euclidean are T10 benchmark alternatives, not production)",
    )
    parser.add_argument(
        "--bev-resolution", type=float, default=None,
        help="Override bev_resolution (for sweeps)",
    )
    parser.add_argument(
        "--bev-morph-kernel", type=int, default=None,
        help="Override bev_morph_kernel (for sweeps)",
    )
    parser.add_argument(
        "--merge-fragments", action="store_true",
        help="Enable post-clustering fragment merge",
    )
    parser.add_argument(
        "--merge-max-dist", type=float, default=None,
        help="Override merge_max_centroid_dist (for sweeps)",
    )
    parser.add_argument(
        "--merge-small-threshold", type=int, default=None,
        help="Override merge_small_threshold (for sweeps)",
    )
    parser.add_argument(
        "--adaptive-hdbscan", action="store_true",
        help="Enable distance-adaptive HDBSCAN min_cluster_size",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Write final aggregate metrics as JSON to this path (for LOSO "
             "cross-validation aggregation). Does not change stdout output.",
    )
    parser.add_argument(
        "--frame-fraction", type=float, default=None,
        help="Evaluate only a centered CONTIGUOUS window covering this fraction "
             "of the sequence (e.g. 0.2 = middle 20%%). Frames stay consecutive "
             "so the two-pass track filter still works -- unlike a strided "
             "subsample, which would break centroid tracking and collapse "
             "recall. Used to speed up LOSO cross-validation eval.",
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
            # Hard error rather than silently scoring geometric-only: a typo'd
            # checkpoint path (e.g. a wrong LOSO fold) would otherwise produce a
            # bogus number attributed to the classifier pipeline. If geometric-
            # only is genuinely intended, pass --no-learned-classifier.
            parser.error(
                f"Classifier checkpoint not found: {args.classifier_ckpt}. "
                "Refusing to run (would silently fall back to geometric-only). "
                "Pass --no-learned-classifier to evaluate without a classifier.")
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

    # Optional centered contiguous window (LOSO fast eval). Frames must stay
    # consecutive so the centroid tracker in the two-pass filter keeps linking
    # across frames; a strided/sparse subsample would break tracking entirely.
    window_start = 0
    if args.frame_fraction is not None:
        if not (0.0 < args.frame_fraction <= 1.0):
            parser.error("--frame-fraction must be in (0, 1]")
        if args.frame_fraction < 1.0:
            W = max(1, int(round(args.frame_fraction * num_frames)))
            window_start = (num_frames - W) // 2
            bin_paths = bin_paths[window_start:window_start + W]
            label_paths = label_paths[window_start:window_start + W]
            num_frames = len(bin_paths)
            print(f"Frame-fraction {args.frame_fraction}: centered contiguous "
                  f"window [{window_start}, {window_start + W}) -> "
                  f"{num_frames} frames")

    cfg = PIPELINE_CONFIG
    if args.min_track_length is not None:
        cfg["min_track_length"] = args.min_track_length
    if args.min_known_votes is not None:
        cfg["min_track_known_votes"] = args.min_known_votes
    if args.min_known_ratio is not None:
        cfg["min_track_known_ratio"] = args.min_known_ratio
    if args.min_points is not None:
        cfg["min_points_in_cluster"] = args.min_points
    if args.hdbscan_min_cluster_size is not None:
        cfg["hdbscan_min_cluster_size"] = args.hdbscan_min_cluster_size
    if args.tracker_max_distance is not None:
        cfg["tracker_max_distance"] = args.tracker_max_distance
    if args.tracker_max_disappeared is not None:
        cfg["tracker_max_disappeared"] = args.tracker_max_disappeared
    if args.ransac_distance_threshold is not None:
        cfg["ransac_distance_threshold"] = args.ransac_distance_threshold
    if args.ransac_iterations is not None:
        cfg["ransac_iterations"] = args.ransac_iterations
    if args.voxel_before_denoise and args.no_voxel_before_denoise:
        parser.error("--voxel-before-denoise and --no-voxel-before-denoise are mutually exclusive")
    if args.voxel_before_denoise:
        cfg["voxel_before_denoise"] = True
    if args.no_voxel_before_denoise:
        cfg["voxel_before_denoise"] = False
    if args.cluster_voxel_size is not None:
        cfg["cluster_voxel_size"] = args.cluster_voxel_size
    if args.clustering_method is not None:
        cfg["clustering_method"] = args.clustering_method
    if args.bev_resolution is not None:
        cfg["bev_resolution"] = args.bev_resolution
    if args.bev_morph_kernel is not None:
        cfg["bev_morph_kernel"] = args.bev_morph_kernel
    if args.merge_fragments:
        cfg["merge_fragments"] = True
    if args.merge_max_dist is not None:
        cfg["merge_max_centroid_dist"] = args.merge_max_dist
    if args.merge_small_threshold is not None:
        cfg["merge_small_threshold"] = args.merge_small_threshold
    if args.adaptive_hdbscan:
        cfg["adaptive_hdbscan"] = True
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

    if args.json_out is not None:
        result = {
            "seq": args.seq,
            "frames": int(num_frames),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_iou": float(mean_iou),
            "ious": [float(x) for x in all_ious],
            "classifier_ckpt": (None if args.no_learned_classifier
                                else args.classifier_ckpt),
            "iou_threshold": float(args.iou_threshold),
            "target": args.target,
            "frame_fraction": args.frame_fraction,
            "window_start": int(window_start),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote metrics JSON: {args.json_out}")
