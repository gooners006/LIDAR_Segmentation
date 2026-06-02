"""Point cloud classifier: dual-branch PointNet (shape + bbox features).

Stage A: trained on ShapeNet partials + synthetic negatives.
Classes: car (0), not-car (1).
"""

import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_POINTS_FOR_CLASSIFIER = 10
MIN_POINTS_FOR_OBB = 20
MIN_EXTENT = 1e-3
MAX_ASPECT = 100.0
MAX_VOLUME = 1e4
BBOX_CLIP = 5.0
NUM_POINTS = 512
BBOX_FEAT_DIM = 8

CLASS_LABELS = ["car", "not-car"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_LABELS)}
NUM_CLASSES = len(CLASS_LABELS)


@dataclass
class ClassificationResult:
    """Output of a classifier — a predicted label and its confidence score."""
    label: str
    confidence: float


# ---------------------------------------------------------------------------
# Shared preprocessing (used by both training and inference)
# ---------------------------------------------------------------------------


def compute_cluster_extent(points: np.ndarray) -> np.ndarray:
    """Compute bounding-box extents for a point cluster.

    Uses an oriented bounding box (OBB) when the cluster has at least
    ``MIN_POINTS_FOR_OBB`` points.  Falls back to an axis-aligned
    bounding box (AABB) if the OBB computation fails or the cluster is
    too small.

    Args:
        points: (N, 3) array of cluster points.

    Returns:
        (3,) float32 array of unsorted extents, each >= ``MIN_EXTENT``.
    """
    if len(points) >= MIN_POINTS_FOR_OBB:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            obb = pcd.get_oriented_bounding_box()
            extent = np.asarray(obb.extent, dtype=np.float32)
            if extent.shape == (3,) and np.isfinite(extent).all():
                return np.maximum(extent, MIN_EXTENT)
        except Exception:
            pass
    # AABB fallback
    ptp = points.max(axis=0) - points.min(axis=0)
    return np.maximum(ptp.astype(np.float32), MIN_EXTENT)


def extract_bbox_features(points: np.ndarray) -> np.ndarray:
    """Build an 8-dimensional metric-scale feature vector for the bbox branch.

    Features (in order): sorted extents (min, med, max), volume,
    aspect ratio max/min, aspect ratio med/min, log(1 + point_count),
    and vertical height span.

    Args:
        points: (N, 3) array of raw cluster points in metres.

    Returns:
        (8,) float32 feature vector.
    """
    extent = compute_cluster_extent(points)
    min_d, med_d, max_d = np.sort(extent)

    volume = min(float(np.prod(extent)), MAX_VOLUME)
    aspect_max_min = min(float(max_d / min_d), MAX_ASPECT)
    aspect_med_min = min(float(med_d / min_d), MAX_ASPECT)
    log_count = float(np.log1p(len(points)))
    height = float(points[:, 2].max() - points[:, 2].min())

    return np.array([
        min_d, med_d, max_d,
        volume,
        aspect_max_min, aspect_med_min,
        log_count,
        height,
    ], dtype=np.float32)


def sample_or_pad(points: np.ndarray, num_points: int,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample or pad a point cloud to exactly *num_points*.

    If the cluster has more points than needed, a random subset is
    selected without replacement.  If fewer, all points are kept and
    the remainder is filled by random duplication.

    Args:
        points: (N, 3) array of cluster points.
        num_points: Target number of points.
        rng: NumPy random generator.  Defaults to a deterministic seed
            for reproducible inference; pass a seeded generator for
            training augmentation.

    Returns:
        (num_points, 3) float32 array.
    """
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(0)
    if n >= num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        pad_idx = rng.choice(n, num_points - n, replace=True)
        idx = np.concatenate([np.arange(n), pad_idx])
    return points[idx].astype(np.float32)


def normalize_unit_sphere(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize points to fit within a unit sphere.

    Centers on the centroid, then scales so the farthest point is at
    distance 1.  This makes the point branch scale-invariant; absolute
    size is captured separately by the bbox features.

    Args:
        points: (N, 3) array of cluster points.
        eps: Minimum scale to avoid division by zero.

    Returns:
        (N, 3) float32 array of normalized points, or zeros if the
        cluster is degenerate.
    """
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    if scale < eps or not np.isfinite(scale):
        return np.zeros_like(points, dtype=np.float32)
    return (points / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Heuristic fallback (original classifier, renamed)
# ---------------------------------------------------------------------------


def classify_bbox_heuristic(extent: np.ndarray,
                            center: np.ndarray) -> ClassificationResult:
    """Rule-based fallback classifier using OBB dimensions.

    Applies hardcoded size ranges for car detection.  Not used when a
    learned model is loaded; kept as a fallback for runs without a
    checkpoint.

    Args:
        extent: (3,) OBB extents in metres (unsorted).
        center: (3,) OBB centre position.

    Returns:
        ClassificationResult with predicted label and confidence.
    """
    min_d, med_d, max_d = np.sort(extent)

    if 3.0 <= max_d <= 5.5 and 1.5 <= med_d <= 2.5 and 1.0 <= min_d <= 2.0:
        return ClassificationResult("car", 0.85)
    return ClassificationResult("not-car", 0.6)


# ---------------------------------------------------------------------------
# PointNet Classifier Model
# ---------------------------------------------------------------------------


class PointNetClassifier(nn.Module):
    """Dual-branch PointNet classifier (point geometry + metric bbox features).

    The point branch processes unit-sphere-normalized coordinates through
    Conv1d layers and max-pools into a 256-d shape descriptor.  The bbox
    branch maps 8 metric features to 32-d.  The concatenated 288-d
    vector is classified into ``num_classes`` categories.

    Args:
        bbox_feat_dim: Dimension of the bounding-box feature vector.
        num_classes: Number of output classes.
    """

    def __init__(self, bbox_feat_dim: int = BBOX_FEAT_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        # Point branch: Conv1d 3 -> 64 -> 128 -> 256, max pool
        self.point_branch = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        # Bbox branch: Linear 8 -> 32
        self.bbox_branch = nn.Sequential(
            nn.Linear(bbox_feat_dim, 32),
            nn.ReLU(inplace=True),
        )
        # Classification head: 288 -> 128 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, points: torch.Tensor,
                bbox_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points:     (B, N, 3)
            bbox_feats: (B, bbox_feat_dim)
        Returns:
            logits:     (B, num_classes)
        """
        x = points.transpose(1, 2)             # (B, 3, N)
        x = self.point_branch(x)               # (B, 256, N)
        x = x.max(dim=2).values                # (B, 256)

        b = self.bbox_branch(bbox_feats)        # (B, 32)

        combined = torch.cat([x, b], dim=1)     # (B, 288)
        return self.head(combined)              # (B, num_classes)


# ---------------------------------------------------------------------------
# Learned classifier inference
# ---------------------------------------------------------------------------


def classify_cluster(
    points: np.ndarray,
    model: PointNetClassifier,
    device: torch.device,
    bbox_stats: dict,
    unknown_threshold: float = 0.50,
) -> ClassificationResult:
    """Classify a single cluster using the learned model.

    Preprocesses the raw cluster (filter non-finite points, extract
    bbox features with z-score normalization, sample/pad to
    ``NUM_POINTS``, normalize to unit sphere), runs inference, and
    applies the unknown-threshold rejection.

    Args:
        points: (N, 3) raw cluster points in metric scale.
        model: Loaded PointNetClassifier (already in eval mode).
        device: Torch device for inference.
        bbox_stats: Dict with ``"mean"`` and ``"std"`` arrays of shape
            (8,) from training-set feature statistics.
        unknown_threshold: If the maximum softmax probability is below
            this value, the cluster is labeled ``"not-car"``.

    Returns:
        ClassificationResult with the predicted label and confidence.
    """
    points = np.asarray(points, dtype=np.float32)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) < MIN_POINTS_FOR_CLASSIFIER:
        return ClassificationResult("not-car", 0.0)

    # Bbox features from raw metric-scale points
    bbox_feats = extract_bbox_features(points)
    bbox_feats = (bbox_feats - bbox_stats["mean"]) / (bbox_stats["std"] + 1e-6)
    bbox_feats = np.clip(bbox_feats, -BBOX_CLIP, BBOX_CLIP)

    # Point branch input
    pts = sample_or_pad(points, NUM_POINTS)
    pts = normalize_unit_sphere(pts)

    # Inference
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
    bbox_t = torch.from_numpy(bbox_feats).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pts_t, bbox_t)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    confidence = float(probs.max())
    pred_idx = int(probs.argmax())

    if confidence < unknown_threshold:
        return ClassificationResult("not-car", confidence)

    return ClassificationResult(CLASS_LABELS[pred_idx], confidence)


def load_classifier(ckpt_path: str, device: torch.device):
    """Load a classifier model and bbox normalization stats from a checkpoint.

    Validates that the checkpoint's metadata (class labels, bbox
    feature dimension, number of input points) matches the runtime
    constants defined in this module.

    Args:
        ckpt_path: Path to the ``.pth`` checkpoint file.
        device: Torch device to load the model onto.

    Returns:
        Tuple of ``(model, bbox_stats)`` where *model* is a
        :class:`PointNetClassifier` in eval mode and *bbox_stats* is a
        dict with ``"mean"`` and ``"std"`` arrays.  Returns
        ``(None, None)`` if the file does not exist.

    Raises:
        ValueError: If checkpoint metadata is inconsistent with runtime
            constants.
    """
    if not os.path.isfile(ckpt_path):
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Read from top-level keys (authoritative), fall back to config, then defaults
    ckpt_bbox_dim = ckpt.get("bbox_feat_dim", cfg.get("bbox_feat_dim", BBOX_FEAT_DIM))
    ckpt_num_cls = len(ckpt["class_labels"]) if "class_labels" in ckpt else cfg.get("num_classes", NUM_CLASSES)

    # Validate checkpoint metadata against runtime constants
    ckpt_labels = ckpt.get("class_labels")
    if ckpt_labels is not None and list(ckpt_labels) != list(CLASS_LABELS):
        raise ValueError(
            f"Checkpoint class labels {ckpt_labels} != runtime {CLASS_LABELS}")
    if ckpt_bbox_dim != BBOX_FEAT_DIM:
        raise ValueError(
            f"Checkpoint bbox_feat_dim={ckpt_bbox_dim} != runtime {BBOX_FEAT_DIM}")
    ckpt_num_pts = ckpt.get("num_points", cfg.get("num_points"))
    if ckpt_num_pts is not None and ckpt_num_pts != NUM_POINTS:
        raise ValueError(
            f"Checkpoint num_points={ckpt_num_pts} != runtime {NUM_POINTS}")

    model = PointNetClassifier(
        bbox_feat_dim=ckpt_bbox_dim,
        num_classes=ckpt_num_cls,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    bbox_stats = {
        "mean": np.array(ckpt["bbox_mean"], dtype=np.float32),
        "std": np.array(ckpt["bbox_std"], dtype=np.float32),
    }
    return model, bbox_stats


