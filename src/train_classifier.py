"""Train PointNet binary classifier (car / not-car).

Stage A: synthetic ShapeNet partials.
Stage B: fine-tune on real mined LiDAR clusters (--stage-b).

Usage:
    python src/train_classifier.py
    python src/train_classifier.py --epochs 30 --batch-size 16
    python src/train_classifier.py --resume checkpoints/classifier_best.pth
    python src/train_classifier.py --eval-only --resume checkpoints/classifier_best.pth

    # Stage B fine-tuning (loads Stage A checkpoint automatically):
    python src/train_classifier.py --stage-b --epochs 15
    python src/train_classifier.py --stage-b --resume checkpoints/classifier_best.pth --epochs 15
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from classifier import (
    BBOX_FEAT_DIM,
    CLASS_LABELS,
    CLASS_TO_IDX,
    NUM_CLASSES,
    NUM_POINTS,
    PointNetClassifier,
    extract_bbox_features,
    normalize_unit_sphere,
    sample_or_pad,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_CONFIG = {
    "shapenet_root": os.path.join(PROJECT_ROOT, "dataset", "shapenet_data"),
    "categories": {
        "02958343": "car",
    },
    "category_scale_m": {
        "02958343": 4.5,
    },
    "category_label": {
        "02958343": CLASS_TO_IDX["car"],
    },
    "num_points": NUM_POINTS,
    "bbox_feat_dim": BBOX_FEAT_DIM,
    "depth_h": 256,
    "depth_w": 256,
    "viewpoint_elev_range": (-20.0, 30.0),
    "viewpoint_azim_range": (0.0, 360.0),
    "viewpoint_radius_factor_range": (1.5, 2.5),
    "val_fraction": 0.2,
    "split_seed": 42,
    "unknown_fraction": 0.50,
    "unknown_scale_range": (0.5, 4.0),
    "unknown_train_seed": 10042,
    "unknown_val_seed": 20042,
    "jitter_std": 0.005,
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-3,
    "lr_decay_step": 30,
    "lr_decay_gamma": 0.5,
    "num_workers": 4,
    "num_classes": NUM_CLASSES,
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "checkpoint_every": 10,
    "log_every": 50,
    "unknown_threshold": 0.50,
}

STAGE_B_CONFIG = {
    "stage_b_root": os.path.join(PROJECT_ROOT, "dataset", "stage_b"),
    "num_points": NUM_POINTS,
    "bbox_feat_dim": BBOX_FEAT_DIM,
    "jitter_std": 0.005,
    "epochs": 15,
    "batch_size": 64,
    "lr": 1e-4,
    "lr_decay_step": 10,
    "lr_decay_gamma": 0.5,
    "num_workers": 4,
    "num_classes": NUM_CLASSES,
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "checkpoint_every": 5,
    "log_every": 50,
    "unknown_threshold": 0.50,
    "stage_a_ckpt": os.path.join(PROJECT_ROOT, "checkpoints", "classifier_best.pth"),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class StageBDataset(data.Dataset):
    """Real mined LiDAR clusters for Stage B fine-tuning.

    Loads centroid-centered .npy files from dataset/stage_b/{split}/{class}/.
    """

    # Fraction of the training pool carved off as a monitoring val set in
    # leave-one-sequence-out mode. Selection is last-epoch, so this slice never
    # affects a fold's evaluated result; it only produces readable loss curves.
    LOSO_MONITOR_FRAC = 0.05
    LOSO_MONITOR_SEED = 1234

    def __init__(self, root: str, split: str, config: dict,
                 held_out_seq: str | None = None):
        """Load real mined cluster paths from disk.

        Args:
            root: Root directory containing train/val splits.
            split: Either ``"train"`` or ``"val"``.
            config: Stage B config dict with ``num_points``, ``jitter_std``.
            held_out_seq: If set (e.g. ``"08"``), enables leave-one-sequence-out
                mode. Clusters are pooled from BOTH ``root/train`` and
                ``root/val`` (the on-disk folders together hold all sequences)
                and every file whose SemanticKITTI sequence prefix equals
                ``held_out_seq`` is excluded entirely — that sequence is the
                detection test set, evaluated separately via ``evaluate.py`` and
                never seen in training or checkpoint selection. The remaining
                pool is split into a ~95%% ``"train"`` / ~5%% ``"val"`` monitoring
                slice (deterministic, disjoint, per class).
        """
        self.config = config
        self.split = split
        self.held_out_seq = held_out_seq
        self.bbox_mean: np.ndarray | None = None
        self.bbox_std: np.ndarray | None = None

        self.items: list[tuple[str, int]] = []  # (npy_path, label)
        if held_out_seq is None:
            source_dirs = [os.path.join(root, split)]
        else:
            # Pool both on-disk folders; the held-out sequence lives in whichever
            # of them it was originally mined into, so we must scan both.
            source_dirs = [os.path.join(root, "train"), os.path.join(root, "val")]

        for cls_name in CLASS_LABELS:
            label = CLASS_TO_IDX[cls_name]
            cls_files: list[str] = []
            for sdir in source_dirs:
                cls_dir = os.path.join(sdir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if not fname.endswith(".npy"):
                        continue
                    if held_out_seq is not None and \
                            fname.split("_")[0] == held_out_seq:
                        continue  # this file belongs to the held-out test seq
                    cls_files.append(os.path.join(cls_dir, fname))
            cls_files.sort()  # stable, deterministic ordering across folds

            if held_out_seq is not None:
                # Deterministic 95/5 carve for the monitoring val set. Same seed
                # and pool length for both the "train" and "val" constructions,
                # so the two are exact complements (disjoint, reproducible).
                rng = np.random.default_rng(self.LOSO_MONITOR_SEED)
                perm = rng.permutation(len(cls_files))
                n_val = max(1, int(round(self.LOSO_MONITOR_FRAC * len(cls_files))))
                val_idx = set(perm[:n_val].tolist())
                if split == "val":
                    cls_files = [f for i, f in enumerate(cls_files) if i in val_idx]
                else:
                    cls_files = [f for i, f in enumerate(cls_files) if i not in val_idx]

            for f in cls_files:
                self.items.append((f, label))

        self.class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, label in self.items:
            self.class_counts[label] += 1
        tag = split if held_out_seq is None else f"{split}/heldout={held_out_seq}"
        print(f"  [{tag}] {len(self.items)} samples: "
              f"{dict(zip(CLASS_LABELS, self.class_counts.tolist()))}")

    def __len__(self) -> int:
        return len(self.items)

    def get_raw_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Load raw (unaugmented) sample: centroid-centered points, bbox features, label."""
        path, label = self.items[idx]
        points = np.load(path).astype(np.float32)
        bbox_feats = extract_bbox_features(points)
        return points, bbox_feats, label

    def __getitem__(self, idx: int) -> dict:
        points, bbox_feats, label = self.get_raw_sample(idx)

        if self.bbox_mean is not None:
            bbox_feats = (bbox_feats - self.bbox_mean) / (self.bbox_std + 1e-6)
            bbox_feats = np.clip(bbox_feats, -5.0, 5.0)

        if self.split == "train":
            points = self._augment_rotation(points)

        if self.split == "train":
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(42 + idx)
        pts = sample_or_pad(points, self.config["num_points"], rng=rng)
        pts = normalize_unit_sphere(pts)

        if self.split == "train":
            pts += np.random.normal(0, self.config["jitter_std"],
                                    size=pts.shape).astype(np.float32)

        return {
            "points": torch.from_numpy(pts).float(),
            "bbox_feats": torch.from_numpy(bbox_feats).float(),
            "label": label,
        }

    @staticmethod
    def _augment_rotation(points: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return (points @ R.T).astype(np.float32)


class ShapeNetClassificationDataset(data.Dataset):
    """ShapeNet partial renders for classifier training.

    Vehicle classes use category-specific scales. Unknowns are rendered from
    non-vehicle ShapeNet categories with random scales.
    """

    def __init__(self, config: dict, split: str = "train"):
        """Discover ShapeNet meshes and prepare vehicle/unknown splits.

        Args:
            config: Stage A training config with ShapeNet paths, viewpoint
                parameters, and unknown fraction settings.
            split: Either ``"train"`` or ``"val"``.
        """
        self.config = config
        self.split = split
        self.bbox_mean: np.ndarray | None = None
        self.bbox_std: np.ndarray | None = None

        vehicle_synsets = set(config["categories"].keys())

        # Vehicle items: (obj_path, synset_id)
        self.vehicle_items: list[tuple[str, str]] = []
        for synset_id in config["categories"]:
            self.vehicle_items.extend(
                self._discover_category(synset_id, split, config))

        # Unknown items: all non-vehicle categories
        all_unknown_items: list[tuple[str, str]] = []
        for synset_dir in sorted(os.listdir(config["shapenet_root"])):
            if synset_dir in vehicle_synsets:
                continue
            full = os.path.join(config["shapenet_root"], synset_dir)
            if not os.path.isdir(full):
                continue
            all_unknown_items.extend(
                self._discover_category(synset_dir, split, config))

        # Subsample unknowns to maintain unknown_fraction
        n_unknown = int(len(self.vehicle_items) * config["unknown_fraction"]
                        / (1.0 - config["unknown_fraction"]))
        seed = (config["unknown_train_seed"] if split == "train"
                else config["unknown_val_seed"])
        rng = np.random.default_rng(seed)
        if len(all_unknown_items) > n_unknown:
            indices = rng.choice(len(all_unknown_items), n_unknown, replace=False)
            self.unknown_items = [all_unknown_items[i] for i in indices]
        else:
            self.unknown_items = all_unknown_items

        self.total_len = len(self.vehicle_items) + len(self.unknown_items)
        print(f"  [{split}] Unknown pool: {len(all_unknown_items)} models "
              f"from {len(set(s for _, s in all_unknown_items))} categories, "
              f"sampled {len(self.unknown_items)}")

    @staticmethod
    def _discover_category(synset_id: str, split: str,
                           config: dict) -> list[tuple[str, str]]:
        """Find all .obj model paths for a ShapeNet category in the given split."""
        cat_dir = os.path.join(config["shapenet_root"], synset_id)
        if not os.path.isdir(cat_dir):
            return []
        model_dirs = sorted(os.listdir(cat_dir))
        rng = np.random.default_rng(config["split_seed"])
        rng.shuffle(model_dirs)
        n_val = int(len(model_dirs) * config["val_fraction"])
        n_train = len(model_dirs) - n_val
        if split == "train":
            model_dirs = model_dirs[:n_train]
        else:
            model_dirs = model_dirs[n_train:]
        items = []
        for md in model_dirs:
            obj_path = os.path.join(cat_dir, md, "models", "model_normalized.obj")
            if os.path.isfile(obj_path):
                items.append((obj_path, synset_id))
        return items

    def __len__(self) -> int:
        return self.total_len

    def get_raw_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Deterministic: (metric_points, bbox_feats, label). No augmentation.

        Uses a per-index seeded RNG so repeated calls with the same idx
        return the same partial render and bbox features.
        """
        if idx < len(self.vehicle_items):
            rng = np.random.default_rng(self.config["split_seed"] + idx)
            return self._get_vehicle_raw(idx, rng=rng)
        else:
            unk_idx = idx - len(self.vehicle_items)
            rng = np.random.default_rng(self.config["split_seed"] + idx)
            return self._get_unknown_raw(unk_idx, rng=rng)

    def _get_vehicle_raw(self, idx: int,
                         rng: np.random.Generator | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, int]:
        """Load a ShapeNet vehicle partial in metric scale."""
        if rng is None:
            rng = np.random.default_rng()
        max_retries = 10
        for attempt in range(max_retries):
            try_idx = idx if attempt == 0 else int(
                rng.integers(0, len(self.vehicle_items)))
            obj_path, synset_id = self.vehicle_items[try_idx]
            scale_m = self.config["category_scale_m"][synset_id]
            label = self.config["category_label"][synset_id]

            mesh = self._load_and_scale(obj_path, scale_m)
            if mesh is None or mesh.get_surface_area() < 1e-6:
                continue

            pts = self._render_partial(mesh, scale_m, rng=rng)
            if pts is None:
                continue

            centroid = pts.mean(axis=0)
            pts = pts - centroid
            bbox_feats = extract_bbox_features(pts)
            return pts, bbox_feats, label

        raise RuntimeError(
            f"Failed to load vehicle sample after {max_retries} retries (idx={idx})")

    def _get_unknown_raw(self, unk_idx: int,
                         rng: np.random.Generator | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, int]:
        """Load a non-vehicle ShapeNet partial with random scale."""
        if rng is None:
            rng = np.random.default_rng()
        scale_lo, scale_hi = self.config["unknown_scale_range"]
        max_retries = 10
        for attempt in range(max_retries):
            try_idx = unk_idx if attempt == 0 else int(
                rng.integers(0, len(self.unknown_items)))
            obj_path, synset_id = self.unknown_items[try_idx]
            scale_m = float(rng.uniform(scale_lo, scale_hi))

            mesh = self._load_and_scale(obj_path, scale_m)
            if mesh is None or mesh.get_surface_area() < 1e-6:
                continue

            pts = self._render_partial(mesh, scale_m, rng=rng)
            if pts is None:
                continue

            centroid = pts.mean(axis=0)
            pts = pts - centroid
            bbox_feats = extract_bbox_features(pts)
            return pts, bbox_feats, CLASS_TO_IDX["not-car"]

        raise RuntimeError(
            f"Failed to load unknown sample after {max_retries} retries "
            f"(unk_idx={unk_idx})")

    def __getitem__(self, idx: int) -> dict:
        pts_metric, bbox_feats, label = self.get_raw_sample(idx)

        # Normalize bbox features if stats are set
        if self.bbox_mean is not None:
            bbox_feats = (bbox_feats - self.bbox_mean) / (self.bbox_std + 1e-6)
            bbox_feats = np.clip(bbox_feats, -5.0, 5.0)

        # Augmentation (train only): Z-axis rotation on metric points
        if self.split == "train":
            pts_metric = self._augment_rotation(pts_metric)

        # Sample/pad and normalize
        if self.split == "train":
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.config["split_seed"] + idx)
        pts = sample_or_pad(pts_metric, self.config["num_points"], rng=rng)
        pts = normalize_unit_sphere(pts)

        # Jitter (train only, after normalization)
        if self.split == "train":
            pts += np.random.normal(0, self.config["jitter_std"],
                                    size=pts.shape).astype(np.float32)

        return {
            "points": torch.from_numpy(pts).float(),
            "bbox_feats": torch.from_numpy(bbox_feats).float(),
            "label": label,
        }

    # --- Mesh loading and rendering (adapted from train_pcn.py) ---

    @staticmethod
    def _sample_mesh_surface(mesh, n_points: int,
                             rng: np.random.Generator) -> np.ndarray:
        """Area-weighted triangle sampling with controlled RNG."""
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles)
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        probs = areas / areas.sum()
        tri_idx = rng.choice(len(tris), size=n_points, p=probs)
        r1 = rng.random(n_points)
        r2 = rng.random(n_points)
        sqrt_r1 = np.sqrt(r1)
        pts = ((1 - sqrt_r1)[:, None] * v0[tri_idx]
               + (sqrt_r1 * (1 - r2))[:, None] * v1[tri_idx]
               + (sqrt_r1 * r2)[:, None] * v2[tri_idx])
        return pts.astype(np.float32)

    @staticmethod
    def _load_and_scale(obj_path: str, scale_m: float):
        """Load mesh from .obj and rescale so its max extent equals *scale_m* metres."""
        try:
            mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
        except Exception:
            return None
        if not mesh.has_vertices() or not mesh.has_triangles():
            return None
        verts = np.asarray(mesh.vertices)
        extents = verts.max(axis=0) - verts.min(axis=0)
        max_extent = extents.max()
        if max_extent < 1e-6:
            return None
        scale = scale_m / max_extent
        centroid = verts.mean(axis=0)
        mesh.vertices = o3d.utility.Vector3dVector((verts - centroid) * scale)
        return mesh

    def _render_partial(self, mesh, scale_m: float,
                        rng: np.random.Generator | None = None,
                        max_retries: int = 5) -> np.ndarray | None:
        """Render a partial point cloud via raycasting from a random viewpoint.

        Returns (N, 3) hit points in world coordinates, or None if all
        viewpoints produce fewer than 50 hits.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh()
        mesh_t.vertex.positions = o3d.core.Tensor(vertices)
        mesh_t.triangle.indices = o3d.core.Tensor(faces)
        scene.add_triangles(mesh_t)

        H, W = self.config["depth_h"], self.config["depth_w"]
        fov_deg = 60.0
        fx = fy = W / (2.0 * np.tan(np.radians(fov_deg / 2)))
        cx, cy = W / 2.0, H / 2.0
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float32)

        for _ in range(max_retries):
            extrinsic = self._random_extrinsic(scale_m, rng=rng)
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(intrinsic),
                o3d.core.Tensor(extrinsic),
                width_px=W, height_px=H,
            )
            ans = scene.cast_rays(rays)
            t_hit = ans["t_hit"].numpy()
            rays_np = rays.numpy()
            origins = rays_np[..., :3]
            directions = rays_np[..., 3:]
            valid = np.isfinite(t_hit)
            if valid.sum() < 50:
                continue
            pts_3d = origins[valid] + t_hit[valid, np.newaxis] * directions[valid]
            return pts_3d.astype(np.float32)
        return None

    def _random_extrinsic(self, scale_m: float,
                          rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random camera extrinsic looking at the origin."""
        if rng is None:
            rng = np.random.default_rng()
        elev_min, elev_max = self.config["viewpoint_elev_range"]
        azim_min, azim_max = self.config["viewpoint_azim_range"]
        f_min, f_max = self.config["viewpoint_radius_factor_range"]
        elev = np.radians(float(rng.uniform(elev_min, elev_max)))
        azim = np.radians(float(rng.uniform(azim_min, azim_max)))
        r = float(rng.uniform(f_min * scale_m, f_max * scale_m))
        cam_pos = np.array([
            r * np.cos(elev) * np.cos(azim),
            r * np.cos(elev) * np.sin(azim),
            r * np.sin(elev),
        ], dtype=np.float64)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
        right /= right_norm
        up = np.cross(right, forward)
        R = np.stack([right, -up, forward], axis=0)
        t = -R @ cam_pos
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R.astype(np.float32)
        extrinsic[:3, 3] = t.astype(np.float32)
        return extrinsic

    @staticmethod
    def _augment_rotation(points: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return (points @ R.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Bbox stats computation
# ---------------------------------------------------------------------------


def compute_bbox_stats(dataset, max_samples: int = 0):
    """Compute mean/std of bbox features from unaugmented training samples.

    Args:
        dataset: any dataset with get_raw_sample(idx) -> (pts, bbox_feats, label).
        max_samples: if > 0, subsample to this many for speed.
    """
    n = len(dataset)
    if max_samples > 0 and n > max_samples:
        indices = np.random.default_rng(42).choice(n, max_samples, replace=False)
        print(f"Computing bbox stats on {max_samples}/{n} samples...")
    else:
        indices = np.arange(n)
        print(f"Computing bbox stats on {n} samples...")
    feats = []
    failed = 0
    for i in tqdm(indices, desc="bbox stats"):
        try:
            _, bf, _ = dataset.get_raw_sample(int(i))
            feats.append(bf)
        except Exception:
            failed += 1
    if not feats:
        raise RuntimeError("No valid bbox features; cannot compute stats.")
    if failed:
        print(f"  Warning: skipped {failed}/{len(indices)} samples")
    feats = np.stack(feats)
    mean = feats.mean(axis=0).astype(np.float32)
    std = np.maximum(feats.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        points = batch["points"].to(device)
        bbox_feats = batch["bbox_feats"].to(device)
        labels = batch["label"].to(device)

        logits = model(points, bbox_feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    """Evaluate model on validation set with both raw and thresholded metrics.

    Returns a dict with val_loss, val_acc, val_macro_f1,
    val_macro_f1_thresh, val_unknown_rate_thresh, per_class report,
    and confusion_matrix.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        points = batch["points"].to(device)
        bbox_feats = batch["bbox_feats"].to(device)
        labels = batch["label"].to(device)

        logits = model(points, bbox_feats)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    n = len(all_labels)

    # Unthresholded metrics
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_LABELS, output_dict=True, zero_division=0)

    # Thresholded metrics
    threshold = config["unknown_threshold"]
    thresh_preds = all_preds.copy()
    max_probs = all_probs.max(axis=1)
    thresh_preds[max_probs < threshold] = CLASS_TO_IDX["not-car"]
    macro_f1_thresh = f1_score(
        all_labels, thresh_preds, average="macro", zero_division=0)
    unknown_rate = (thresh_preds == CLASS_TO_IDX["not-car"]).mean()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    return {
        "val_loss": total_loss / n,
        "val_acc": (all_preds == all_labels).mean(),
        "val_macro_f1": macro_f1,
        "val_macro_f1_thresh": macro_f1_thresh,
        "val_unknown_rate_thresh": unknown_rate,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Train or evaluate the PointNet classifier (Stage A or Stage B).

    Handles dataset creation, bbox stats computation, class weighting,
    checkpoint resume logic, training loop with CSV logging, and
    periodic/best checkpoint saving.
    """
    parser = argparse.ArgumentParser(description="Train PointNet classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--stage-b", action="store_true",
                        help="Fine-tune on real mined clusters (Stage B)")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Stage B only: train from random init "
                             "(skip Stage A checkpoint) — for ablation")
    parser.add_argument("--tag", type=str, default=None,
                        help="Override checkpoint/log filename prefix "
                             "(avoids overwriting existing runs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--held-out-seq", type=str, default=None,
                        help="Stage B only: leave-one-sequence-out mode. Excludes "
                             "this sequence (e.g. '08') from training entirely so "
                             "it can serve as a held-out detection test set. A ~5%% "
                             "monitoring val slice is carved from the remaining "
                             "sequences; checkpoint selection is last-epoch.")
    args = parser.parse_args()

    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")

    is_stage_b = args.stage_b

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    if is_stage_b:
        config = STAGE_B_CONFIG.copy()
        # Auto-resume from Stage A checkpoint if no explicit --resume
        if args.no_pretrain:
            print("Stage B: --no-pretrain set, training from random init "
                  "(no Stage A checkpoint)")
        elif args.resume is None and os.path.isfile(config["stage_a_ckpt"]):
            args.resume = config["stage_a_ckpt"]
            print(f"Stage B: auto-loading Stage A checkpoint: {args.resume}")
    else:
        config = TRAIN_CONFIG.copy()

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage: {'B (real LiDAR fine-tuning)' if is_stage_b else 'A (synthetic ShapeNet)'}")

    # Datasets
    if args.held_out_seq is not None and not is_stage_b:
        parser.error("--held-out-seq requires --stage-b")

    if is_stage_b:
        if args.held_out_seq is not None:
            print(f"Stage B: leave-one-sequence-out, holding out seq "
                  f"{args.held_out_seq} (excluded from training + selection)")
        train_ds = StageBDataset(config["stage_b_root"], "train", config,
                                 held_out_seq=args.held_out_seq)
        val_ds = StageBDataset(config["stage_b_root"], "val", config,
                               held_out_seq=args.held_out_seq)
    else:
        train_ds = ShapeNetClassificationDataset(config, split="train")
        val_ds = ShapeNetClassificationDataset(config, split="val")
        print(f"Train: {len(train_ds)} samples "
              f"({len(train_ds.vehicle_items)} vehicles + "
              f"{len(train_ds.unknown_items)} unknowns)")
        print(f"Val:   {len(val_ds)} samples "
              f"({len(val_ds.vehicle_items)} vehicles + "
              f"{len(val_ds.unknown_items)} unknowns)")

    # Bbox stats: always recompute for Stage B (different domain),
    # for Stage A load from checkpoint if resuming
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    if is_stage_b:
        bbox_mean, bbox_std = compute_bbox_stats(train_ds, max_samples=50_000)
    elif ckpt is not None and "bbox_mean" in ckpt:
        bbox_mean = np.array(ckpt["bbox_mean"], dtype=np.float32)
        bbox_std = np.array(ckpt["bbox_std"], dtype=np.float32)
        print("Loaded bbox stats from checkpoint")
    else:
        bbox_mean, bbox_std = compute_bbox_stats(train_ds)
    train_ds.bbox_mean = bbox_mean
    train_ds.bbox_std = bbox_std
    val_ds.bbox_mean = bbox_mean
    val_ds.bbox_std = bbox_std
    print(f"Bbox mean: {bbox_mean}")
    print(f"Bbox std:  {bbox_std}")

    # Class weights
    if is_stage_b:
        class_counts = train_ds.class_counts.astype(np.float64)
    else:
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for i in range(len(train_ds)):
            try:
                _, _, label = train_ds.get_raw_sample(i)
                class_counts[label] += 1
            except Exception:
                continue
    # Handle missing classes: use 1.0 as placeholder weight (won't affect training)
    has_all_classes = np.all(class_counts > 0)
    if not has_all_classes:
        missing = [CLASS_LABELS[i] for i in range(NUM_CLASSES) if class_counts[i] == 0]
        print(f"Warning: missing classes in training data: {missing}")
        class_counts_safe = np.where(class_counts > 0, class_counts, 1.0)
    else:
        class_counts_safe = class_counts
    weights = 1.0 / np.sqrt(class_counts_safe)
    weights = weights / weights.mean()
    if not has_all_classes:
        for i in range(NUM_CLASSES):
            if class_counts[i] == 0:
                weights[i] = 0.0
    class_weights = torch.from_numpy(weights).float().to(device)
    print(f"Class counts: {dict(zip(CLASS_LABELS, class_counts.astype(int)))}")
    print(f"Class weights: {dict(zip(CLASS_LABELS, weights.round(3)))}")

    # Dataloaders
    train_loader = data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True)

    # Model
    model = PointNetClassifier(
        bbox_feat_dim=config["bbox_feat_dim"],
        num_classes=config["num_classes"],
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_decay_step"],
        gamma=config["lr_decay_gamma"])

    start_epoch = 0
    best_macro_f1 = 0.0

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        if is_stage_b:
            # Stage B: load weights only, fresh optimizer/scheduler/epoch
            print(f"Loaded model weights from {args.resume} (Stage B: fresh optimizer)")
        elif not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_macro_f1 = ckpt.get("metrics", {}).get("val_macro_f1_thresh", 0.0)
            print(f"Resumed from {args.resume} (epoch {start_epoch})")
        else:
            print(f"Loaded from {args.resume}")

    # Eval only
    if args.eval_only:
        model.eval()
        metrics = validate(model, val_loader, criterion, device, config)
        print(f"\nVal loss: {metrics['val_loss']:.4f} | "
              f"Acc: {metrics['val_acc']:.4f} | "
              f"Macro F1: {metrics['val_macro_f1']:.4f} | "
              f"Macro F1 (thresh): {metrics['val_macro_f1_thresh']:.4f} | "
              f"Unknown rate (thresh): {metrics['val_unknown_rate_thresh']:.4f}")
        print("\nPer-class report:")
        for cls in CLASS_LABELS:
            r = metrics["per_class"].get(cls, {})
            print(f"  {cls:12s}  P={r.get('precision', 0):.3f}  "
                  f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}")
        print(f"\nConfusion matrix:\n{metrics['confusion_matrix']}")
        return

    # CSV log
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    prefix = args.tag if args.tag else ("stage_b" if is_stage_b else "classifier")
    csv_path = os.path.join(config["checkpoint_dir"],
                            f"{prefix}_training_log.csv")
    csv_fields = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "val_macro_f1", "val_macro_f1_thresh", "val_unknown_rate_thresh", "lr",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # Training loop
    print(f"\nTraining for {config['epochs'] - start_epoch} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config)
        metrics = validate(model, val_loader, criterion, device, config)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train L={train_loss:.4f} A={train_acc:.4f} | "
              f"Val L={metrics['val_loss']:.4f} A={metrics['val_acc']:.4f} "
              f"F1={metrics['val_macro_f1']:.4f} "
              f"F1t={metrics['val_macro_f1_thresh']:.4f} | "
              f"LR={lr:.1e} | {elapsed:.1f}s")

        # CSV
        row = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "val_acc": f"{metrics['val_acc']:.4f}",
            "val_macro_f1": f"{metrics['val_macro_f1']:.4f}",
            "val_macro_f1_thresh": f"{metrics['val_macro_f1_thresh']:.4f}",
            "val_unknown_rate_thresh": f"{metrics['val_unknown_rate_thresh']:.4f}",
            "lr": f"{lr:.1e}",
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_labels": CLASS_LABELS,
            "class_to_idx": CLASS_TO_IDX,
            "bbox_mean": bbox_mean.tolist(),
            "bbox_std": bbox_std.tolist(),
            "bbox_feature_names": [
                "min_d", "med_d", "max_d", "volume",
                "max_over_min", "med_over_min", "log_count", "height"],
            "num_points": NUM_POINTS,
            "bbox_feat_dim": BBOX_FEAT_DIM,
            "bbox_clip": 5.0,
            "min_points_for_obb": 20,
            "min_points_for_classifier": 10,
            "point_normalization": "unit_sphere",
            "unknown_threshold_default": config["unknown_threshold"],
            "config": config,
            "metrics": metrics,
        }

        # Best — select on thresholded F1 since pipeline uses thresholding
        if metrics["val_macro_f1_thresh"] > best_macro_f1:
            best_macro_f1 = metrics["val_macro_f1_thresh"]
            torch.save(ckpt_data, os.path.join(
                config["checkpoint_dir"], f"{prefix}_best.pth"))
            print(f"  -> New best macro F1 (thresh): {best_macro_f1:.4f}")

        # Last
        torch.save(ckpt_data, os.path.join(
            config["checkpoint_dir"], f"{prefix}_last.pth"))

        # Periodic
        if (epoch + 1) % config["checkpoint_every"] == 0:
            torch.save(ckpt_data, os.path.join(
                config["checkpoint_dir"], f"{prefix}_epoch{epoch+1}.pth"))

    # Final report
    print(f"\nTraining complete. Best macro F1 (thresh): {best_macro_f1:.4f}")

    model.eval()
    final = validate(model, val_loader, criterion, device, config)
    print(f"\nFinal per-class report:")
    for cls in CLASS_LABELS:
        r = final["per_class"].get(cls, {})
        print(f"  {cls:12s}  P={r.get('precision', 0):.3f}  "
              f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}")
    print(f"\nConfusion matrix:\n{final['confusion_matrix']}")

    if is_stage_b:
        print(f"\nStage B checkpoints saved with prefix '{prefix}_'.")
        print("To evaluate on pipeline: python src/evaluate.py "
              f"--classifier-ckpt checkpoints/{prefix}_best.pth")
    else:
        print("\nNote: unknown-class metrics are based on non-vehicle ShapeNet "
              "categories. Real-world rejection quality requires evaluation on "
              "real LiDAR negatives (Stage B).")


if __name__ == "__main__":
    main()
