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

# Longitudinal (length-axis, Z) center prior — Direction 2 (far-end under-completion).
# The width prior (X) and up-shift (Y) recenter the occluded lateral side and the
# ground-cut, but there is no length correction: a partial truncated at the
# occluded far *end* gets normalized around its observed (near) portion, so PCN's
# full-car output stops short of the far end (donor metric #32: far_end cov 0.133,
# the worst region). Mirror the width prior along Z — extend-only, toward the
# ego-far end — to push the center out so the completion reaches the unseen end.
# Shipped ON by default (constructor length_prior default = this constant; pass
# length_prior=None to A/B it off). 4.14 m = real amodal-GT median car length
# (#29); it is the "full car length" analogue of the 1.9 m width prior above.
# Evidence (Direction 2, seq 08, all paired):
#   - synthetic true-GT (scratchpad/length_prior_synth_check.py): far-end coverage
#     0.42 -> 0.59, CD/F both improve, far-end under-reach halved;
#   - donor metric #32 (per-car median, tau=0.15): far_end cov 0.123 -> 0.324,
#     overall cov 0.307 -> 0.428, out-of-box 0.0004 -> 0.0014 (<< 0.0083 guard);
#   - box metric #29: reverses length under-completion (signed dL -0.44 -> -0.32,
#     completed now beats raw), |dL| 0.44 -> 0.35, width flat, |dH| +1.8 cm.
# 4.5 m was rejected: higher coverage but out-of-box 0.0122 breaks the guard
# (over-extends real compacts). Extend-only, so full-length observations are
# untouched. See docs/findings.md (#35) and docs/completion/donor_metric.md.
COMPLETION_CAR_LENGTH_PRIOR = 4.14

# Per-car length estimate (Direction 2 Step 1b). The fixed prior above is a
# population median, so it is wrong in BOTH tails: measured against the 40
# amodal-GT cars of seq 08 it over-extends compacts (< 3.6 m: bias +0.63 m) and
# under-extends long cars (>= 4.6 m: bias -0.59 m); only the middle band is
# unbiased. track_length_estimate() replaces it with a per-car number taken from
# the car's own observed footprint lengths across its track.
#
# Why a quantile of fit_length and nothing cleverer — all measured offline
# against amodal GT (scratchpad/length_estimator_probe{,2}.py, n=1508 pairs):
#   - length-from-width is impossible: corr(GT L, GT W) = +0.018, so even a
#     perfect width measurement predicts length worse (MAE 0.53 m) than the
#     constant (0.35 m). Height / range / point count are equally uninformative.
#   - a far-end face-support test does NOT detect longitudinal truncation (the
#     missing length is *larger* when the far end looks well covered).
#   - the observed footprint length does carry the signal: corr +0.52 per frame,
#     +0.87 once aggregated per car.
#   - the aggregate must be a quantile, not the max: track-max is the worst
#     estimator tested (compact bias +0.95 m) because one contaminated L-shape
#     fit sets it.
# Aggregate only over gate-passed frames — fragments and merges are exactly the
# contamination that ruins the max.
#
# IMPORTANT (Finding #36): this fixes the compact over-extension, but targeting a
# car's TRUE length still leaves the completed box short on normal/long cars
# (signed dL -0.33 / -0.58 even when L_est is unbiased). A control run that
# deliberately over-shot the target (q90 + 0.45) improved BOTH the box metric and
# donor coverage on those bands — so a second, independent under-extension lives
# inside the completion itself (PCN under-fills, and the center push also drives
# `radius`, hence output scale). A per-car length estimate cannot correct that;
# see the Step-1c note in docs/completion/plan.md.
# Promoted 2026-08-02 (Finding #36): q90 + 0.12 m beat q95 (offset 0) on both
# metrics and every band. The offset corrects q90's known low bias (-0.136 m);
# slope of the car-level fit came out 1.00, so it is a pure bias correction, and
# leave-one-car-out only degrades MAE 0.171 -> 0.181.
COMPLETION_LENGTH_TRACK_QUANTILE = 90
COMPLETION_LENGTH_TRACK_OFFSET = 0.12
# Below this many gate-passed frames a quantile degenerates toward the max, so
# fall back to the fixed prior instead.
COMPLETION_LENGTH_MIN_FRAMES = 5


def track_length_estimate(
    fit_lengths,
    quantile: float = COMPLETION_LENGTH_TRACK_QUANTILE,
    offset: float = COMPLETION_LENGTH_TRACK_OFFSET,
    min_frames: int = COMPLETION_LENGTH_MIN_FRAMES,
    fallback: Optional[float] = COMPLETION_CAR_LENGTH_PRIOR,
) -> Optional[float]:
    """Per-car full-length estimate (m) from one track's footprint fits.

    ``fit_lengths`` are the L-shape ``fit_length`` values of the track's
    gate-passed frames (see estimate_canonical_frame). Returns ``fallback`` when
    the track is too short for the quantile to be meaningful, and None only if
    ``fallback`` is None (which disables the length push entirely).
    """
    vals = np.asarray([v for v in fit_lengths if v is not None], dtype=np.float64)
    if len(vals) < min_frames:
        return fallback
    return float(np.percentile(vals, quantile) + offset)


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
        length_prior: Optional[float] = COMPLETION_CAR_LENGTH_PRIOR,
    ):
        self.model_path = model_path
        self.heading_method = heading_method
        self.merge_max_width = merge_max_width
        self.fragment_min_length = fragment_min_length
        self.length_prior = length_prior
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Dispatch on checkpoint type. PoinTr checkpoints carry a "pointr_config";
        # PCN checkpoints do not. Both expose the same forward(partial)->(coarse,
        # fine) contract and consume a PCN_N_INPUT-point partial, so complete()
        # is model-agnostic below.
        if "pointr_config" in ckpt:
            from pointr import PoinTr

            pc = ckpt["pointr_config"]
            model = PoinTr(
                num_proxies=pc["num_proxies"],
                num_query=pc["num_query"],
                trans_dim=pc["trans_dim"],
                num_heads=pc["num_heads"],
                enc_depth=pc["enc_depth"],
                dec_depth=pc["dec_depth"],
                grid_size=pc["grid_size"],
                knn_dgcnn=pc["knn_dgcnn"],
                knn_geo=pc["knn_geo"],
            ).to(device)
        else:
            from pcn import PCN

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

    def estimate_canonical_frame(
        self, partial_xyz: np.ndarray, length_estimate: Optional[float] = None
    ) -> tuple[Optional[dict], Optional[str]]:
        """Input gate + canonical-car-frame estimate used by complete().

        ``partial_xyz`` is a single-frame observation in the sensor frame
        (Z = gravity up). ``length_estimate`` overrides the fixed
        ``self.length_prior`` with a per-car full-length estimate (m) for this
        observation — see the ``COMPLETION_CAR_LENGTH_PRIOR`` note on why a
        constant is wrong in both tails. The caller owns the estimate because it
        is a *track*-level quantity (main.py has the track's frames; this module
        stays stateless). Returns (frame, skip_reason); exactly one is None.
        frame keys: ``basis`` (3x3, sensor -> canonical via ``pts @ basis``),
        ``center`` (3, in canonical coords), ``radius`` (scale-corrected),
        ``fit_length``/``fit_width`` (L-shape footprint, m). The canonical car
        frame is X=width, Y=up, Z=length — the estimated symmetry plane is
        X=center[0], which external callers (e.g. the mirrored-partial
        baseline) can reuse so their geometry matches complete() exactly.
        """
        pts = np.asarray(partial_xyz, dtype=np.float64)
        if len(pts) < 16:
            return None, "too_few_points"

        # Reorient sensor frame -> canonical car frame (X=width, Y=up, Z=length).
        # Gravity (sensor +Z) maps to the canonical up axis; the horizontal
        # heading axis (from L-shape fitting, or PCA for A/B) maps to length.
        xy = pts[:, :2] - pts[:, :2].mean(0)

        # Input-quality gate from the L-shape footprint (computed independently of
        # the heading choice so the gate is consistent). Fragments and merges
        # never complete into plausible cars, so skip them rather than emit junk.
        ls_len_dir, _, fit_length, fit_width = self._lshape_axes(xy)
        if self.fragment_min_length is not None and fit_length < self.fragment_min_length:
            return None, "fragment_input"
        if self.merge_max_width is not None and fit_width > self.merge_max_width:
            return None, "merge_suspected"

        if self.heading_method == "pca":
            len_dir, _, _, _ = self._pca_axes(xy)
        else:
            len_dir = ls_len_dir
        norm = np.linalg.norm(len_dir)
        if norm < 1e-9:
            return None, "degenerate_orientation"
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

        # length axis (Z): extend-only push toward the occluded far end. Same
        # mechanism as the width prior — sign(center[2]) is the car's position
        # along length relative to ego (origin), so the far-from-ego end is the
        # occluded one. A per-car length_estimate takes precedence over the fixed
        # prior; disabled when both are None (Direction 2, #32).
        length_target = (length_estimate if length_estimate is not None
                         else self.length_prior)
        if length_target is not None:
            sign_z = np.sign(center[2]) if abs(center[2]) > 1e-9 else 1.0
            observed_len = pts_c[:, 2].max() - pts_c[:, 2].min()
            center[2] += sign_z * max(0.5 * length_target - 0.5 * observed_len, 0.0)

        radius = float(np.linalg.norm(pts_c - center, axis=1).max()) / COMPLETION_SCALE_CORRECTION
        if radius < 1e-6:
            return None, "degenerate_radius"
        return {
            "basis": basis,
            "center": center,
            "radius": radius,
            "fit_length": fit_length,
            "fit_width": fit_width,
        }, None

    def complete(
        self, partial_xyz: np.ndarray, class_label: str,
        length_estimate: Optional[float] = None,
        sample_seed: Optional[int] = None,
    ) -> tuple[np.ndarray, Optional[str]]:
        """Complete a single-frame partial car cluster using PCN.

        ``partial_xyz`` must be a single-frame observation in the **sensor frame**
        (Z = gravity up, ego at the origin), not accumulated multi-frame points.
        The completed cloud is returned in the same sensor frame.

        Reproduces the training normalization from the partial (Finding #26): the
        old PCA + partial-centroid + partial-radius path was never seen in
        training and produced blobs. Instead we reorient to the canonical car
        frame (X=width, Y=up, Z=length), estimate the full-car center, and
        normalize by a scale-corrected radius (see estimate_canonical_frame).

        ``length_estimate`` (m) optionally replaces the fixed length prior with a
        per-car estimate for this observation; see estimate_canonical_frame.

        ``sample_seed`` controls the PCN_N_INPUT-point subsample (_fix_size).
        When given, a fresh RNG is seeded per call, so the completed cloud for a
        given cluster is reproducible and independent of how many completions ran
        before it (Finding #38: self._rng carries state across complete() calls,
        making production output call-order dependent). When None, self._rng is
        used, preserving the eval scripts that reset _rng externally per pair.

        Returns (output_points, skip_reason). skip_reason is None on success.
        class_label is currently unused; the priors are car-specific (the
        pipeline only completes cars).
        """
        if self._model is None:
            return partial_xyz.astype(np.float32), "model_not_loaded"

        pts = np.asarray(partial_xyz, dtype=np.float64)
        frame, skip = self.estimate_canonical_frame(pts, length_estimate)
        if skip is not None:
            return pts.astype(np.float32), skip
        basis, center, radius = frame["basis"], frame["center"], frame["radius"]
        pts_c = pts @ basis

        pts_norm = ((pts_c - center) / radius).astype(np.float32)
        rng = (np.random.default_rng(sample_seed)
               if sample_seed is not None else self._rng)
        pts_fixed = self._fix_size(pts_norm, PCN_N_INPUT, rng)

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
