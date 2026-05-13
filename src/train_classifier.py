"""Train PointNet classifier on ShapeNet partials + synthetic unknowns.

Usage:
    python src/train_classifier.py
    python src/train_classifier.py --epochs 30 --batch-size 16
    python src/train_classifier.py --resume checkpoints/classifier_last.pth
    python src/train_classifier.py --eval-only --resume checkpoints/classifier_best.pth
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.utils.data as data
import trimesh
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

logging.getLogger("trimesh").setLevel(logging.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_CONFIG = {
    "shapenet_root": os.path.join(PROJECT_ROOT, "dataset", "shapenet_data"),
    "categories": {
        "02958343": "car",
        "02924116": "bus",
        "03790512": "motorcycle",
    },
    "category_scale_m": {
        "02958343": 4.5,
        "02924116": 10.0,
        "03790512": 2.2,
    },
    "category_label": {
        "02958343": CLASS_TO_IDX["car"],
        "02924116": CLASS_TO_IDX["bus"],
        "03790512": CLASS_TO_IDX["motorcycle"],
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
    "unknown_fraction": 0.30,
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
    "unknown_threshold": 0.65,
}

# Synthetic unknown shapes: (name, (x_size, y_size, z_size))
UNKNOWN_SHAPES = [
    ("pole",       (0.2, 0.2, 3.0)),
    ("sign",       (0.1, 1.0, 1.5)),
    ("barrier",    (2.0, 0.3, 0.8)),
    ("box_carish", (3.5, 1.8, 1.5)),
    ("box_busish", (8.0, 2.5, 3.0)),
    ("flat_patch", (2.0, 2.0, 0.05)),
    ("pedestrian", (0.6, 0.6, 1.7)),
    ("small_obj",  (0.5, 0.5, 0.5)),
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ShapeNetClassificationDataset(data.Dataset):
    """ShapeNet partial renders + synthetic unknowns for classifier training."""

    def __init__(self, config: dict, split: str = "train"):
        self.config = config
        self.split = split
        self.bbox_mean: np.ndarray | None = None
        self.bbox_std: np.ndarray | None = None

        # Vehicle items: (obj_path, synset_id)
        self.vehicle_items: list[tuple[str, str]] = []
        for synset_id in config["categories"]:
            cat_dir = os.path.join(config["shapenet_root"], synset_id)
            if not os.path.isdir(cat_dir):
                continue
            model_dirs = sorted(os.listdir(cat_dir))
            rng = np.random.default_rng(config["split_seed"])
            rng.shuffle(model_dirs)
            n_val = int(len(model_dirs) * config["val_fraction"])
            n_train = len(model_dirs) - n_val
            if split == "train":
                model_dirs = model_dirs[:n_train]
            else:
                model_dirs = model_dirs[n_train:]
            for md in model_dirs:
                obj_path = os.path.join(cat_dir, md, "models", "model_normalized.obj")
                if os.path.isfile(obj_path):
                    self.vehicle_items.append((obj_path, synset_id))

        # Generate synthetic unknowns
        n_unknown = int(len(self.vehicle_items) * config["unknown_fraction"]
                        / (1.0 - config["unknown_fraction"]))
        seed = (config["unknown_train_seed"] if split == "train"
                else config["unknown_val_seed"])
        self.unknown_points = self._generate_unknowns(n_unknown, seed)

        self.total_len = len(self.vehicle_items) + len(self.unknown_points)

    def _generate_unknowns(self, n: int, seed: int) -> list[np.ndarray]:
        """Generate n synthetic unknown point clouds in metric scale.

        Mix: ~50% geometric shapes, ~25% cropped vehicle partials,
             ~25% noisy vehicle subsets.
        """
        rng = np.random.default_rng(seed)
        unknowns = []

        n_geometric = n // 2
        n_cropped = n // 4
        n_noisy = n - n_geometric - n_cropped

        for i in range(n_geometric):
            unknowns.append(self._generate_geometric_unknown(rng, i))

        for _ in range(n_cropped):
            pts = self._generate_cropped_partial(rng)
            if pts is None:
                pts = self._generate_geometric_unknown(rng)
            unknowns.append(pts)

        for _ in range(n_noisy):
            pts = self._generate_noisy_subset(rng)
            if pts is None:
                pts = self._generate_geometric_unknown(rng)
            unknowns.append(pts)

        return unknowns

    def _generate_geometric_unknown(self, rng: np.random.Generator,
                                     shape_idx: int | None = None) -> np.ndarray:
        if shape_idx is None:
            shape_idx = int(rng.integers(0, len(UNKNOWN_SHAPES)))
        else:
            shape_idx = shape_idx % len(UNKNOWN_SHAPES)
        _, size_xyz = UNKNOWN_SHAPES[shape_idx]
        n_pts = int(rng.integers(50, 513))
        pts = rng.uniform(-0.5, 0.5, size=(n_pts, 3)).astype(np.float32)
        pts *= np.array(size_xyz, dtype=np.float32)
        pts += rng.normal(0, 0.02, size=pts.shape).astype(np.float32)
        return pts

    def _generate_cropped_partial(self, rng: np.random.Generator) -> np.ndarray | None:
        """Load a vehicle mesh, surface-sample, then axis-aligned half-crop."""
        if not self.vehicle_items:
            return None
        idx = int(rng.integers(0, len(self.vehicle_items)))
        obj_path, synset_id = self.vehicle_items[idx]
        scale_m = self.config["category_scale_m"][synset_id]
        mesh = self._load_and_scale(obj_path, scale_m)
        if mesh is None or mesh.area < 1e-6:
            return None
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, 1024)
            pts = pts.astype(np.float32)
        except Exception:
            return None
        axis = int(rng.integers(0, 3))
        median = float(np.median(pts[:, axis]))
        mask = pts[:, axis] > median if rng.random() < 0.5 else pts[:, axis] < median
        pts = pts[mask]
        if len(pts) < 20:
            return None
        pts -= pts.mean(axis=0)
        return pts

    def _generate_noisy_subset(self, rng: np.random.Generator) -> np.ndarray | None:
        """Load a vehicle mesh, surface-sample, subsample 30-60%, add heavy noise."""
        if not self.vehicle_items:
            return None
        idx = int(rng.integers(0, len(self.vehicle_items)))
        obj_path, synset_id = self.vehicle_items[idx]
        scale_m = self.config["category_scale_m"][synset_id]
        mesh = self._load_and_scale(obj_path, scale_m)
        if mesh is None or mesh.area < 1e-6:
            return None
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, 512)
            pts = pts.astype(np.float32)
        except Exception:
            return None
        keep_frac = float(rng.uniform(0.3, 0.6))
        n_keep = max(int(len(pts) * keep_frac), 20)
        keep_idx = rng.choice(len(pts), n_keep, replace=False)
        pts = pts[keep_idx]
        pts += rng.normal(0, 0.1, size=pts.shape).astype(np.float32)
        pts -= pts.mean(axis=0)
        return pts

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
            pts = self.unknown_points[unk_idx]
            bbox_feats = extract_bbox_features(pts)
            return pts, bbox_feats, CLASS_TO_IDX["unknown"]

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
            if mesh is None or mesh.area < 1e-6:
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
    def _load_and_scale(obj_path: str, scale_m: float):
        try:
            mesh = trimesh.load(obj_path, force="mesh", process=False,
                                skip_materials=True)
        except Exception:
            return None
        extents = mesh.bounding_box.extents
        max_extent = extents.max()
        if max_extent < 1e-6:
            return None
        mesh.apply_scale(scale_m / max_extent)
        mesh.vertices -= mesh.centroid
        return mesh

    def _render_partial(self, mesh, scale_m: float,
                        rng: np.random.Generator | None = None,
                        max_retries: int = 5) -> np.ndarray | None:
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

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


def compute_bbox_stats(dataset: ShapeNetClassificationDataset):
    """Compute mean/std of bbox features from unaugmented training samples."""
    print("Computing bbox feature statistics...")
    feats = []
    for i in tqdm(range(len(dataset)), desc="bbox stats"):
        try:
            _, bf, _ = dataset.get_raw_sample(i)
            feats.append(bf)
        except Exception:
            continue
    feats = np.stack(feats)
    mean = feats.mean(axis=0).astype(np.float32)
    std = np.maximum(feats.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device, config):
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
    thresh_preds[max_probs < threshold] = CLASS_TO_IDX["unknown"]
    macro_f1_thresh = f1_score(
        all_labels, thresh_preds, average="macro", zero_division=0)
    unknown_rate = (thresh_preds == CLASS_TO_IDX["unknown"]).mean()

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
    parser = argparse.ArgumentParser(description="Train PointNet classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    config = TRAIN_CONFIG.copy()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    train_ds = ShapeNetClassificationDataset(config, split="train")
    val_ds = ShapeNetClassificationDataset(config, split="val")
    print(f"Train: {len(train_ds)} samples "
          f"({len(train_ds.vehicle_items)} vehicles + "
          f"{len(train_ds.unknown_points)} unknowns)")
    print(f"Val:   {len(val_ds)} samples "
          f"({len(val_ds.vehicle_items)} vehicles + "
          f"{len(val_ds.unknown_points)} unknowns)")

    # Bbox stats: load from checkpoint if resuming, else compute fresh
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    if ckpt is not None and "bbox_mean" in ckpt:
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
    class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in range(len(train_ds)):
        try:
            _, _, label = train_ds.get_raw_sample(i)
            class_counts[label] += 1
        except Exception:
            continue
    if np.any(class_counts == 0):
        raise ValueError(f"Missing classes in training data: {class_counts}")
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.mean()
    class_weights = torch.from_numpy(weights).float().to(device)
    print(f"Class counts: {dict(zip(CLASS_LABELS, class_counts.astype(int)))}")
    print(f"Class weights: {dict(zip(CLASS_LABELS, weights.round(3)))}")

    # Majority-class baseline
    val_labels = []
    for i in range(len(val_ds)):
        try:
            _, _, label = val_ds.get_raw_sample(i)
            val_labels.append(label)
        except Exception:
            continue
    val_labels = np.array(val_labels)
    majority_class = np.bincount(val_labels).argmax()
    majority_preds = np.full_like(val_labels, majority_class)
    baseline_f1 = f1_score(val_labels, majority_preds, average="macro",
                           zero_division=0)
    print(f"Majority-class baseline macro F1: {baseline_f1:.4f} "
          f"(class={CLASS_LABELS[majority_class]})")

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

    # Resume model/optimizer state from checkpoint loaded earlier
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        if not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_macro_f1 = ckpt.get("metrics", {}).get("val_macro_f1", 0.0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

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
    csv_path = os.path.join(config["checkpoint_dir"],
                            "classifier_training_log.csv")
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

        # Best
        if metrics["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["val_macro_f1"]
            torch.save(ckpt_data, os.path.join(
                config["checkpoint_dir"], "classifier_best.pth"))
            print(f"  -> New best macro F1: {best_macro_f1:.4f}")

        # Last
        torch.save(ckpt_data, os.path.join(
            config["checkpoint_dir"], "classifier_last.pth"))

        # Periodic
        if (epoch + 1) % config["checkpoint_every"] == 0:
            torch.save(ckpt_data, os.path.join(
                config["checkpoint_dir"], f"classifier_epoch{epoch+1}.pth"))

    # Final report
    print(f"\nTraining complete. Best macro F1: {best_macro_f1:.4f}")
    print(f"Majority-class baseline: {baseline_f1:.4f}")

    # Print final confusion matrix
    model.eval()
    final = validate(model, val_loader, criterion, device, config)
    print(f"\nFinal per-class report:")
    for cls in CLASS_LABELS:
        r = final["per_class"].get(cls, {})
        print(f"  {cls:12s}  P={r.get('precision', 0):.3f}  "
              f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}")
    print(f"\nConfusion matrix:\n{final['confusion_matrix']}")
    print("\nNote: unknown-class metrics are based on synthetic unknowns only. "
          "Real-world rejection quality requires evaluation on real LiDAR "
          "negatives (Stage B).")


if __name__ == "__main__":
    main()
