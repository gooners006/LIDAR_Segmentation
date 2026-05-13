"""Point cloud classifier: dual-branch PointNet (shape + bbox features).

Stage A: trained on ShapeNet partials + synthetic unknowns.
Classes: car (0), bus (1), motorcycle (2), unknown (3).
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

CLASS_LABELS = ["car", "bus", "motorcycle", "unknown"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_LABELS)}
NUM_CLASSES = len(CLASS_LABELS)


@dataclass
class ClassificationResult:
    label: str
    confidence: float


# ---------------------------------------------------------------------------
# Shared preprocessing (used by both training and inference)
# ---------------------------------------------------------------------------


def compute_cluster_extent(points: np.ndarray) -> np.ndarray:
    """Compute OBB extents if enough points, else AABB. Returns (3,) unsorted."""
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
    """8-dim metric-scale feature vector from raw cluster points."""
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
    """Subsample or pad to exactly num_points."""
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
    """Center on centroid, scale to unit sphere."""
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
    """Heuristic classification from OBB dimensions and center position."""
    min_d, med_d, max_d = np.sort(extent)

    if min_d < 0.15 and max_d < 3.0 and center[2] > 0.5:
        return ClassificationResult("unknown", 0.7)
    if max_d < 2.0 and med_d < 1.0 and min_d < 0.8:
        return ClassificationResult("unknown", 0.8)
    if 3.0 <= max_d <= 5.5 and 1.5 <= med_d <= 2.5 and 1.0 <= min_d <= 2.0:
        return ClassificationResult("car", 0.85)
    if max_d > 5.5 or (max_d > 4.0 and med_d > 2.0):
        return ClassificationResult("bus", 0.75)
    if max_d > 2.0 and min_d > 0.3 and med_d > 0.5:
        return ClassificationResult("unknown", 0.6)
    return ClassificationResult("unknown", 0.0)


# ---------------------------------------------------------------------------
# PointNet Classifier Model
# ---------------------------------------------------------------------------


class PointNetClassifier(nn.Module):
    """Dual-branch: point geometry (PointNet) + metric bbox features."""

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
    unknown_threshold: float = 0.65,
) -> ClassificationResult:
    """Classify a single cluster using the learned model.

    Args:
        points: (N, 3) raw cluster points in metric scale.
        model: loaded PointNetClassifier (already in eval mode).
        device: torch device.
        bbox_stats: {"mean": ndarray(8,), "std": ndarray(8,)}.
        unknown_threshold: reject if max softmax < this.
    """
    points = np.asarray(points, dtype=np.float32)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) < MIN_POINTS_FOR_CLASSIFIER:
        return ClassificationResult("unknown", 0.0)

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
        return ClassificationResult("unknown", confidence)

    return ClassificationResult(CLASS_LABELS[pred_idx], confidence)


def load_classifier(ckpt_path: str, device: torch.device):
    """Load classifier model and bbox stats from checkpoint.

    Returns (model, bbox_stats) or (None, None) if file missing.
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


