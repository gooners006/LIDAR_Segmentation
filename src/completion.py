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

    Expected directory layout (from ``--mine-pairs``)::

        root/<class>/dense_s00_0003.npy            # accumulated track (M, 3)
        root/<class>/sparse_s00_0003_f0005.npy     # per-frame observation (N, 3)

    Each sparse file is paired with the dense file sharing its track tag.
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
            dense_by_tag = {}
            for dp in glob.glob(os.path.join(cls_dir, "dense_*.npy")):
                tag = os.path.basename(dp).replace("dense_", "").replace(".npy", "")
                dense_by_tag[tag] = dp
            for sp in sorted(glob.glob(os.path.join(cls_dir, "sparse_*.npy"))):
                fname = os.path.basename(sp).replace("sparse_", "").replace(".npy", "")
                parts = fname.split("_f")
                tag = parts[0] if len(parts) >= 2 else fname
                if tag in dense_by_tag:
                    self.pairs.append((sp, dense_by_tag[tag], cls))

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

PCN_N_INPUT = 256
PCN_NUM_COARSE = 1024
PCN_GRID_SIZE = 2

# Corrected-inference constants, calibrated on synthetic val cars (Finding #26,
# scratchpad/verify_pcn_step1.py). The model trains in a canonical car frame
# (X=width, Y=up, Z=length) centered on the GT centroid and normalized by the GT
# radius. complete() must reproduce that normalization from a partial observation:
#   - COMPLETION_SCALE_CORRECTION: median partial_radius / gt_radius. The partial
#     radius (from a one-sided view) systematically over/under-estimates the true
#     GT radius; divide by this to recover the training scale.
#   - COMPLETION_CAR_WIDTH_PRIOR: assumed full car width (m). The occluded far side
#     is unobserved, so push the center toward it to the half-width prior.
#   - COMPLETION_UP_SHIFT: ground removal cuts the lower body, biasing the bbox
#     center upward; shift it back down. This is the dominant center correction.
COMPLETION_SCALE_CORRECTION = 1.137
COMPLETION_CAR_WIDTH_PRIOR = 1.9
COMPLETION_UP_SHIFT = 0.25

# Heading estimation for the canonical-frame reorientation. The major horizontal
# PCA axis is ambiguous on near-square / two-face / merged BEV footprints (dense
# but poorly-completed tracks, e.g. seq-08 tid 301/762/884): the eigenvectors no
# longer align with the car body, so length/width get swapped or rotated ~45 deg.
# Search-based L-shape fitting (Zhang et al. 2017, "Efficient L-Shape Fitting for
# Vehicle Detection Using Laser Scanners") scores edge adherence instead of
# variance spread, so it locks onto the true rectangle even when it isn't
# elongated. "lshape" is the default; "pca" is kept for A/B comparison.
COMPLETION_HEADING_METHOD = "lshape"
# Number of yaw samples in [0, pi/2) for the L-shape search (1 deg resolution).
COMPLETION_LSHAPE_ANGLES = 90
# Input-quality gate from the L-shape footprint fit. Validated on seq-08 (A/B
# run, scratchpad/ab_heading.py): of 47 completed car tracks, every implausible
# completion came from a fragment or merge input, and gating them lifts
# completion precision from 18/47 to 18/26. Heading (lshape vs pca) is neutral on
# real data; the fit's real value is detecting these bad inputs.
#   - COMPLETION_FRAGMENT_MIN_LENGTH: fitted footprint length (m) below which the
#     cluster is a fragment, not a whole car (0/15 plausible below 2.7 m).
#   - COMPLETION_MERGE_MAX_WIDTH: fitted footprint width (m) above which the
#     cluster is two merged cars / a 90-deg error (0/7 plausible above 2.3 m).
# Set either to None to disable that gate.
COMPLETION_FRAGMENT_MIN_LENGTH = 2.7
COMPLETION_MERGE_MAX_WIDTH = 2.3


class PointCloudCompleter:
    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 0,
        heading_method: str = COMPLETION_HEADING_METHOD,
        merge_max_width: Optional[float] = COMPLETION_MERGE_MAX_WIDTH,
        fragment_min_length: Optional[float] = COMPLETION_FRAGMENT_MIN_LENGTH,
    ):
        self.model_path = model_path
        self.heading_method = heading_method
        self.merge_max_width = merge_max_width
        self.fragment_min_length = fragment_min_length
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
    def _pca_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Length/width axes from the horizontal PCA eigenvectors (legacy).

        Returns (length_dir, width_dir, length_extent, width_extent). The length
        axis is the major eigenvector; extents are the point spread along each.
        """
        cov = xy.T @ xy
        w, v = np.linalg.eigh(cov)
        len_dir = v[:, int(np.argmax(w))]
        wid_dir = np.array([-len_dir[1], len_dir[0]])
        e_len = xy @ len_dir
        e_wid = xy @ wid_dir
        return (len_dir, wid_dir,
                float(e_len.max() - e_len.min()),
                float(e_wid.max() - e_wid.min()))

    @staticmethod
    def _lshape_axes(
        xy: np.ndarray, n_angles: int = COMPLETION_LSHAPE_ANGLES, d0: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Length/width axes by search-based L-shape fitting (Zhang et al. 2017).

        Rotates a bounding rectangle through yaw in [0, pi/2) and scores each
        angle by the *closeness* criterion: every point is bounded by two pairs
        of parallel edges, so its distance to the nearest edge (in whichever of
        the two directions is closer) should be small if the rectangle is aligned
        with the car body. Summing 1/distance rewards points hugging the edges,
        which stays well-defined even when the footprint isn't elongated (where
        PCA degenerates). Returns (length_dir, width_dir, length_extent,
        width_extent); the longer rectangle side is the length axis.
        """
        thetas = np.linspace(0.0, np.pi / 2.0, n_angles, endpoint=False)
        best_score, best_t = -np.inf, 0.0
        for t in thetas:
            ct, st = np.cos(t), np.sin(t)
            c1 = xy[:, 0] * ct + xy[:, 1] * st
            c2 = -xy[:, 0] * st + xy[:, 1] * ct
            d1 = np.minimum(c1.max() - c1, c1 - c1.min())
            d2 = np.minimum(c2.max() - c2, c2 - c2.min())
            d = np.maximum(np.minimum(d1, d2), d0)
            score = float(np.sum(1.0 / d))
            if score > best_score:
                best_score, best_t = score, t

        ct, st = np.cos(best_t), np.sin(best_t)
        u1 = np.array([ct, st])
        u2 = np.array([-st, ct])
        ext1 = float(np.ptp(xy @ u1))
        ext2 = float(np.ptp(xy @ u2))
        if ext1 >= ext2:
            return u1, u2, ext1, ext2
        return u2, u1, ext2, ext1

    def complete(
        self, partial_xyz: np.ndarray, class_label: str
    ) -> tuple[np.ndarray, Optional[str]]:
        """Complete a single-frame partial car cluster using PCN.

        ``partial_xyz`` must be a single-frame observation in the **sensor frame**
        (Z = gravity up, ego at the origin), not accumulated multi-frame points.
        The completed cloud is returned in the same sensor frame.

        Reproduces the training normalization from the partial (Finding #26): the
        old PCA + partial-centroid + partial-radius path was never seen in
        training and produced blobs. Instead we reorient to the canonical car
        frame (X=width, Y=up, Z=length), estimate the full-car center, and
        normalize by a scale-corrected radius.

        Returns (output_points, skip_reason). skip_reason is None on success.
        class_label is currently unused; the priors below are car-specific (the
        pipeline only completes cars).
        """
        if self._model is None:
            return partial_xyz.astype(np.float32), "model_not_loaded"

        pts = np.asarray(partial_xyz, dtype=np.float64)
        if len(pts) < 16:
            return pts.astype(np.float32), "too_few_points"

        # Reorient sensor frame -> canonical car frame (X=width, Y=up, Z=length).
        # Gravity (sensor +Z) maps to the canonical up axis; the horizontal
        # heading axis (from L-shape fitting, or PCA for A/B) maps to length.
        xy = pts[:, :2] - pts[:, :2].mean(0)

        # Input-quality gate from the L-shape footprint (computed independently of
        # the heading choice so the gate is consistent). Fragments and merges
        # never complete into plausible cars, so skip them rather than emit junk.
        ls_len_dir, _, fit_length, fit_width = self._lshape_axes(xy)
        if self.fragment_min_length is not None and fit_length < self.fragment_min_length:
            return pts.astype(np.float32), "fragment_input"
        if self.merge_max_width is not None and fit_width > self.merge_max_width:
            return pts.astype(np.float32), "merge_suspected"

        if self.heading_method == "pca":
            len_dir, _, _, _ = self._pca_axes(xy)
        else:
            len_dir = ls_len_dir
        norm = np.linalg.norm(len_dir)
        if norm < 1e-9:
            return pts.astype(np.float32), "degenerate_orientation"
        e_len = np.array([len_dir[0] / norm, len_dir[1] / norm, 0.0])
        e_wid = np.array([-e_len[1], e_len[0], 0.0])         # perpendicular in ground
        e_up = np.array([0.0, 0.0, 1.0])
        basis = np.column_stack([e_wid, e_up, e_len])        # sensor -> canonical
        pts_c = pts @ basis

        # Full-car center estimate from a one-sided partial:
        #   - up axis (Y): shift the bbox center down to undo the ground-cut bias
        #   - width axis (X): push toward the occluded far-from-ego side to the
        #     car-width prior (sign of bbox-center X = direction away from ego)
        center = 0.5 * (pts_c.min(0) + pts_c.max(0))
        center[1] -= COMPLETION_UP_SHIFT
        sign = np.sign(center[0]) if abs(center[0]) > 1e-9 else 1.0
        observed_w = pts_c[:, 0].max() - pts_c[:, 0].min()
        center[0] += sign * max(0.5 * COMPLETION_CAR_WIDTH_PRIOR - 0.5 * observed_w, 0.0)

        radius = float(np.linalg.norm(pts_c - center, axis=1).max()) / COMPLETION_SCALE_CORRECTION
        if radius < 1e-6:
            return pts.astype(np.float32), "degenerate_radius"

        pts_norm = ((pts_c - center) / radius).astype(np.float32)
        pts_fixed = self._fix_size(pts_norm, PCN_N_INPUT, self._rng)

        import torch
        with torch.no_grad():
            inp = torch.from_numpy(pts_fixed).unsqueeze(0).to(self._device)
            _, fine = self._model(inp)
            fine_np = fine.squeeze(0).cpu().numpy()

        # Un-normalize in the canonical frame, then rotate back to the sensor frame.
        pred_c = fine_np * radius + center
        completed = (pred_c @ basis.T).astype(np.float32)
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
