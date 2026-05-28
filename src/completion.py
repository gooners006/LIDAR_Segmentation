"""Step 7: Point cloud completion (DL-based, thesis contribution).

Provides a stub interface for a pretrained completion network. In passthrough
mode (no model loaded), returns the input cloud unchanged so the pipeline
runs end-to-end without model weights.

Includes sim-to-real augmentation and real-world fine-tuning infrastructure
to bridge the ShapeNet-to-LiDAR domain gap.
"""

import glob
import os
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Sim-to-Real augmentation
# ---------------------------------------------------------------------------

def simulate_lidar_noise(
    points: np.ndarray,
    range_sigma: float = 0.005,
    angular_resolution: float = 0.0035,
    dropout_rate: float = 0.05,
    max_range: float = 80.0,
) -> np.ndarray:
    """Apply LiDAR-realistic noise to synthetic (e.g. ShapeNet) point clouds.

    Simulates range-dependent Gaussian noise, angular quantization (beam
    pattern), and distance-dependent point dropout.

    Args:
        points: (N, 3) clean point cloud.
        range_sigma: Noise std per metre of range.
        angular_resolution: Beam angular resolution in radians (~0.2 deg).
        dropout_rate: Base probability of dropping a point at max_range.
        max_range: Normalisation constant for range-dependent effects.
    """
    if len(points) == 0:
        return points

    pts = points.copy()
    ranges = np.linalg.norm(pts, axis=1, keepdims=True).clip(min=1e-6)

    # Range-proportional Gaussian noise
    pts += np.random.randn(*pts.shape) * (range_sigma * ranges)

    # Angular quantization (simulate discrete beam angles)
    r_xy = np.linalg.norm(pts[:, :2], axis=1, keepdims=True).clip(min=1e-6)
    elevation = np.arctan2(pts[:, 2:3], r_xy)
    azimuth = np.arctan2(pts[:, 1:2], pts[:, 0:1])
    elevation = np.round(elevation / angular_resolution) * angular_resolution
    azimuth = np.round(azimuth / angular_resolution) * angular_resolution
    pts[:, 0:1] = r_xy * np.cos(azimuth)
    pts[:, 1:2] = r_xy * np.sin(azimuth)
    pts[:, 2:3] = r_xy * np.tan(elevation)

    # Distance-dependent dropout
    drop_prob = dropout_rate * (ranges.squeeze() / max_range)
    keep_mask = np.random.rand(len(pts)) > drop_prob
    return pts[keep_mask]


# ---------------------------------------------------------------------------
# Real-world dataset loader
# ---------------------------------------------------------------------------

class KITTIObjectDataset:
    """Load sparse/dense point cloud pairs for completion training.

    Expected directory layout::

        root/<class>/sparse_XXXX.npy   # partial single-frame observation (N, 3)
        root/<class>/dense_XXXX.npy    # accumulated multi-frame ground truth (M, 3)

    Use ``extract_pairs_from_sequence`` to build this from pipeline tracking output.
    """

    def __init__(self, root: str, classes: Optional[list] = None, max_points: int = 2048):
        self.root = root
        self.max_points = max_points
        self.pairs: list[tuple[str, str, str]] = []

        class_dirs = os.listdir(root) if classes is None else classes
        for cls in class_dirs:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            sparse_files = sorted(glob.glob(os.path.join(cls_dir, "sparse_*.npy")))
            for sp in sparse_files:
                dp = sp.replace("sparse_", "dense_")
                if os.path.exists(dp):
                    self.pairs.append((sp, dp, cls))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        sparse_path, dense_path, cls = self.pairs[idx]
        sparse = np.load(sparse_path).astype(np.float32)
        dense = np.load(dense_path).astype(np.float32)
        sparse = self._subsample(sparse, self.max_points)
        dense = self._subsample(dense, self.max_points)
        return sparse, dense, cls

    @staticmethod
    def _subsample(pts: np.ndarray, n: int) -> np.ndarray:
        if len(pts) <= n:
            pad = np.zeros((n - len(pts), 3), dtype=pts.dtype)
            return np.vstack([pts, pad])
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]

    @staticmethod
    def extract_pairs_from_sequence(
        tracks_json: str, objects_dir: str, output_dir: str, min_frames: int = 5
    ):
        """Build sparse/dense pairs from pipeline tracking output.

        The accumulated PLY is the dense target; a random subsample of it
        (sized to approximate a single-frame observation) is the sparse input.
        """
        import json
        import open3d as o3d

        with open(tracks_json) as f:
            meta = json.load(f)

        for track in meta["tracks"]:
            n_frames = track["last_frame"] - track["first_frame"] + 1
            if n_frames < min_frames:
                continue

            tid = track["track_id"]
            cls = track["class"]
            ply_path = os.path.join(objects_dir, f"{tid}.ply")
            if not os.path.exists(ply_path):
                continue

            dense_pcd = o3d.io.read_point_cloud(ply_path)
            dense_pts = np.asarray(dense_pcd.points, dtype=np.float32)

            cls_dir = os.path.join(output_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)

            np.save(os.path.join(cls_dir, f"dense_{tid:04d}.npy"), dense_pts)

            n_sparse = max(len(dense_pts) // n_frames, 64)
            sparse_pts = dense_pts[
                np.random.choice(len(dense_pts), min(n_sparse, len(dense_pts)), replace=False)
            ]
            np.save(os.path.join(cls_dir, f"sparse_{tid:04d}.npy"), sparse_pts)


# ---------------------------------------------------------------------------
# Completion model
# ---------------------------------------------------------------------------

PCN_N_INPUT = 2048
PCN_NUM_COARSE = 1024
PCN_GRID_SIZE = 4


class PointCloudCompleter:
    def __init__(self, model_path: Optional[str] = None, seed: int = 0):
        self.model_path = model_path
        self._model = None
        self._device = None
        self._rng = np.random.default_rng(seed)
        if model_path is not None:
            self._load_model(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self, path: str):
        import torch
        from pcn import PCN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)

        cfg = ckpt.get("config", {})
        if cfg.get("coarse_n_points", PCN_NUM_COARSE) != PCN_NUM_COARSE:
            raise ValueError(
                f"Checkpoint coarse_n_points={cfg['coarse_n_points']} "
                f"!= expected {PCN_NUM_COARSE}"
            )

        model = PCN(num_coarse=PCN_NUM_COARSE, grid_size=PCN_GRID_SIZE).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self._model = model
        self._device = device

    @staticmethod
    def _fix_size(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        pts = pts.astype(np.float32)
        if len(pts) == 0:
            return np.zeros((n, 3), dtype=np.float32)
        if len(pts) == n:
            return pts
        if len(pts) > n:
            idx = rng.choice(len(pts), n, replace=False)
            return pts[idx]
        pad_idx = rng.choice(len(pts), n - len(pts), replace=True)
        return np.vstack([pts, pts[pad_idx]])

    @staticmethod
    def _pca_axes(pts: np.ndarray) -> np.ndarray:
        """Principal axes as columns, sorted by descending eigenvalue (right-handed)."""
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        axes = eigenvectors[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        return axes

    def complete(
        self, partial_xyz: np.ndarray, class_label: str
    ) -> tuple[np.ndarray, Optional[str]]:
        """Complete a partial point cloud using PCN.

        Returns (output_points, skip_reason). skip_reason is None on success.
        class_label is currently unused; reserved for future class-specific completers.
        """
        if self._model is None:
            return partial_xyz.astype(np.float32), "model_not_loaded"

        pts = np.asarray(partial_xyz, dtype=np.float32)
        if len(pts) == 0:
            return pts, "empty_input"

        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        radius = float(np.linalg.norm(pts_centered, axis=1).max())
        if radius < 1e-6:
            return pts, "degenerate_radius"

        # PCA alignment: rotate longest axis to X for canonical orientation
        axes = self._pca_axes(pts_centered)
        pts_aligned = pts_centered @ axes

        pts_norm = pts_aligned / radius
        pts_fixed = self._fix_size(pts_norm, PCN_N_INPUT, self._rng)

        import torch
        with torch.no_grad():
            inp = torch.from_numpy(pts_fixed).unsqueeze(0).to(self._device)
            _, fine = self._model(inp)
            fine_np = fine.squeeze(0).cpu().numpy()

        # Undo: scale → undo PCA rotation → undo centering
        completed = (fine_np * radius @ axes.T + centroid).astype(np.float32)
        return completed, None


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Bidirectional Chamfer distance between two point clouds."""
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)
    d_pred_to_gt, _ = tree_gt.query(pred)
    d_gt_to_pred, _ = tree_pred.query(gt)
    return float(d_pred_to_gt.mean() + d_gt_to_pred.mean())


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.01) -> float:
    """F-Score at a given distance threshold."""
    tree_pred = cKDTree(pred)
    tree_gt = cKDTree(gt)
    d_pred_to_gt, _ = tree_gt.query(pred)
    d_gt_to_pred, _ = tree_pred.query(gt)
    precision = float((d_gt_to_pred < threshold).mean())
    recall = float((d_pred_to_gt < threshold).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
