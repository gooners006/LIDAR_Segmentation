"""Train PCN on ShapeNet for point cloud completion.

Usage:
    python src/train_pcn.py                          # default: 100 epochs, batch 8
    python src/train_pcn.py --epochs 50 --batch-size 4
    python src/train_pcn.py --resume checkpoints/pcn_last.pth
    python src/train_pcn.py --eval-only --resume checkpoints/pcn_best.pth
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from completion import f_score
from pcn import PCN, pcn_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "gt_n_points": 16384,
    "partial_n_points": 2048,
    "coarse_n_points": 1024,
    "depth_h": 256,
    "depth_w": 256,
    "viewpoint_elev_range": (-20.0, 30.0),
    "viewpoint_azim_range": (0.0, 360.0),
    "viewpoint_radius_factor_range": (1.5, 2.5),
    "val_fraction": 0.2,
    "split_seed": 42,
    "alpha": 0.5,
    "epochs": 100,
    "batch_size": 8,
    "lr": 1e-4,
    "lr_decay_step": 40,
    "lr_decay_gamma": 0.5,
    "num_workers": 4,
    "seed": 42,
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "checkpoint_every": 10,
    "log_every": 50,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ShapeNetCompletionDataset(data.Dataset):
    """ShapeNet meshes → (partial, gt, gt_coarse) via depth-image back-projection."""

    def __init__(self, config: dict, split: str = "train"):
        """Discover ShapeNet models and prepare train/val split.

        Args:
            config: Training config with paths, point counts, and split params.
            split: Either ``"train"`` or ``"val"``.
        """
        self.config = config
        self.split = split
        self.items: list[tuple[str, str]] = []  # (obj_path, synset_id)

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
                    self.items.append((obj_path, synset_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        """Load one sample: partial render, full GT, and coarse GT subsample."""
        max_retries = 10
        for attempt in range(max_retries):
            try_idx = idx if attempt == 0 else np.random.randint(0, len(self))
            obj_path, synset_id = self.items[try_idx]
            scale_m = self.config["category_scale_m"][synset_id]

            mesh = self._load_and_scale(obj_path, scale_m)
            if mesh is None or mesh.get_surface_area() < 1e-6:
                continue

            gt_pts = self._sample_gt(mesh)

            partial_pts = self._render_partial(mesh, scale_m)
            if partial_pts is None:
                continue

            # Center both on GT centroid
            centroid = gt_pts.mean(axis=0)
            gt_pts -= centroid
            partial_pts -= centroid

            # Normalize to unit sphere so CD loss is scale-invariant across categories
            radius = np.linalg.norm(gt_pts, axis=1).max()
            if radius > 1e-6:
                gt_pts /= radius
                partial_pts /= radius

            # Random Z-axis rotation augmentation
            if self.split == "train":
                gt_pts, partial_pts = self._augment_rotation(gt_pts, partial_pts)

            # Subsample GT for coarse loss target
            coarse_idx = np.random.choice(
                len(gt_pts), self.config["coarse_n_points"], replace=False
            )
            gt_coarse = gt_pts[coarse_idx]

            return {
                "partial": torch.from_numpy(partial_pts).float(),
                "gt": torch.from_numpy(gt_pts).float(),
                "gt_coarse": torch.from_numpy(gt_coarse).float(),
            }

        raise RuntimeError(
            f"Failed to load a valid sample after {max_retries} retries (initial idx={idx})"
        )

    @staticmethod
    def _load_and_scale(obj_path: str, scale_m: float):
        """Load mesh via Open3D and rescale so max extent equals *scale_m* metres."""
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

    def _sample_gt(self, mesh) -> np.ndarray:
        """Uniformly sample ``gt_n_points`` from the mesh surface via area-weighted triangle sampling."""
        n_points = self.config["gt_n_points"]
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles)
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        probs = areas / areas.sum()
        tri_idx = np.random.choice(len(tris), size=n_points, p=probs)
        r1 = np.random.random(n_points)
        r2 = np.random.random(n_points)
        sqrt_r1 = np.sqrt(r1)
        pts = ((1 - sqrt_r1)[:, None] * v0[tri_idx]
               + (sqrt_r1 * (1 - r2))[:, None] * v1[tri_idx]
               + (sqrt_r1 * r2)[:, None] * v2[tri_idx])
        return pts.astype(np.float32)

    def _render_partial(
        self, mesh, scale_m: float, max_retries: int = 5
    ) -> np.ndarray | None:
        """Render depth from a random viewpoint and back-project to 3D."""
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        # Build raycasting scene
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh()
        mesh_t.vertex.positions = o3d.core.Tensor(vertices)
        mesh_t.triangle.indices = o3d.core.Tensor(faces)
        scene.add_triangles(mesh_t)

        H, W = self.config["depth_h"], self.config["depth_w"]
        fov_deg = 60.0
        fx = fy = W / (2.0 * np.tan(np.radians(fov_deg / 2)))
        cx, cy = W / 2.0, H / 2.0

        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        for _ in range(max_retries):
            extrinsic = self._random_extrinsic(scale_m)
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(intrinsic),
                o3d.core.Tensor(extrinsic),
                width_px=W,
                height_px=H,
            )
            ans = scene.cast_rays(rays)
            t_hit = ans["t_hit"].numpy()  # (H, W)

            rays_np = rays.numpy()  # (H, W, 6)
            origins = rays_np[..., :3]
            directions = rays_np[..., 3:]

            valid = np.isfinite(t_hit)
            if valid.sum() < 512:
                continue

            pts_3d = origins[valid] + t_hit[valid, np.newaxis] * directions[valid]
            pts_3d = pts_3d.astype(np.float32)

            return self._fix_size(pts_3d, self.config["partial_n_points"])

        return None

    def _random_extrinsic(self, scale_m: float) -> np.ndarray:
        """Sample a random viewpoint and return the world-to-camera 4×4 matrix."""
        elev_min, elev_max = self.config["viewpoint_elev_range"]
        azim_min, azim_max = self.config["viewpoint_azim_range"]
        f_min, f_max = self.config["viewpoint_radius_factor_range"]

        elev = np.radians(np.random.uniform(elev_min, elev_max))
        azim = np.radians(np.random.uniform(azim_min, azim_max))
        r = np.random.uniform(f_min * scale_m, f_max * scale_m)

        # Camera position in world coordinates
        cam_pos = np.array(
            [
                r * np.cos(elev) * np.cos(azim),
                r * np.cos(elev) * np.sin(azim),
                r * np.sin(elev),
            ],
            dtype=np.float64,
        )

        # Look-at: camera looks at origin
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

        # World-to-camera rotation (camera axes: right=X, down=Y, forward=Z in OpenCV)
        R = np.stack([right, -up, forward], axis=0)  # (3, 3)
        t = -R @ cam_pos  # translation

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R.astype(np.float32)
        extrinsic[:3, 3] = t.astype(np.float32)
        return extrinsic

    @staticmethod
    def _fix_size(pts: np.ndarray, n: int) -> np.ndarray:
        """Subsample or pad (by duplication) to exactly *n* points."""
        if len(pts) == n:
            return pts
        if len(pts) > n:
            idx = np.random.choice(len(pts), n, replace=False)
            return pts[idx]
        # Pad by repeating
        pad_idx = np.random.choice(len(pts), n - len(pts), replace=True)
        return np.vstack([pts, pts[pad_idx]])

    @staticmethod
    def _augment_rotation(gt: np.ndarray, partial: np.ndarray):
        """Apply a random Z-axis rotation to both GT and partial consistently."""
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return gt @ R.T, partial @ R.T


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, config, epoch):
    """Run one training epoch. Returns dict with avg train_loss, train_cd_coarse, train_cd_fine."""
    model.train()
    total_loss = 0.0
    total_cd_c = 0.0
    total_cd_f = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
    for batch in pbar:
        partial = batch["partial"].to(device)
        gt = batch["gt"].to(device)
        gt_coarse = batch["gt_coarse"].to(device)

        coarse, fine = model(partial)
        loss, cd_c, cd_f = pcn_loss(coarse, fine, gt, gt_coarse, config["alpha"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cd_c += cd_c.item()
        total_cd_f += cd_f.item()
        n_batches += 1

        if n_batches % config["log_every"] == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                cd_c=f"{cd_c.item():.5f}",
                cd_f=f"{cd_f.item():.5f}",
            )

    return {
        "train_loss": total_loss / n_batches,
        "train_cd_coarse": total_cd_c / n_batches,
        "train_cd_fine": total_cd_f / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, config, compute_fscore: bool = False):
    """Evaluate on validation set. Returns dict with val_loss, val_cd_coarse, val_cd_fine, and optionally val_fscore."""
    model.eval()
    total_loss = 0.0
    total_cd_c = 0.0
    total_cd_f = 0.0
    n_batches = 0
    fscores = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        partial = batch["partial"].to(device)
        gt = batch["gt"].to(device)
        gt_coarse = batch["gt_coarse"].to(device)

        coarse, fine = model(partial)
        loss, cd_c, cd_f = pcn_loss(coarse, fine, gt, gt_coarse, config["alpha"])

        total_loss += loss.item()
        total_cd_c += cd_c.item()
        total_cd_f += cd_f.item()
        n_batches += 1

        if compute_fscore and n_batches <= 20:
            pred_np = fine[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fscores.append(f_score(pred_np, gt_np, threshold=0.1))

    metrics = {
        "val_loss": total_loss / n_batches,
        "val_cd_coarse": total_cd_c / n_batches,
        "val_cd_fine": total_cd_f / n_batches,
    }
    if fscores:
        metrics["val_fscore"] = np.mean(fscores)
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model, optimizer, scheduler, and metrics to a checkpoint file."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": TRAIN_CONFIG,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Train or evaluate PCN on ShapeNet partial→complete pairs.

    Handles dataset creation, model setup, checkpoint resume, training
    loop with CSV logging, and best/last/periodic checkpoint saving.
    """
    parser = argparse.ArgumentParser(description="Train PCN on ShapeNet")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--workers", type=int, default=TRAIN_CONFIG["num_workers"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    config = TRAIN_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["num_workers"] = args.workers

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Datasets
    train_ds = ShapeNetCompletionDataset(config, split="train")
    val_ds = ShapeNetCompletionDataset(config, split="val")
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Model
    model = PCN(num_coarse=config["coarse_n_points"], grid_size=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PCN parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_decay_step"], gamma=config["lr_decay_gamma"]
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["metrics"].get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    if args.eval_only:
        metrics = evaluate(model, val_loader, device, config, compute_fscore=True)
        print(f"Eval results: {metrics}")
        return

    # Training
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    log_path = os.path.join(config["checkpoint_dir"], "training_log.csv")
    log_exists = os.path.isfile(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_cd_coarse",
                "train_cd_fine",
                "val_loss",
                "val_cd_coarse",
                "val_cd_fine",
                "val_fscore",
                "lr",
                "time_s",
            ]
        )

    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, config, epoch
        )
        scheduler.step()

        compute_fs = (epoch + 1) % 10 == 0 or epoch == config["epochs"] - 1
        val_metrics = evaluate(
            model, val_loader, device, config, compute_fscore=compute_fs
        )

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"train={train_metrics['train_loss']:.5f} | "
            f"val={val_metrics['val_loss']:.5f} | "
            f"cd_f={val_metrics['val_cd_fine']:.5f} | "
            f"lr={lr:.2e} | {elapsed:.0f}s"
        )

        log_writer.writerow(
            [
                epoch + 1,
                train_metrics["train_loss"],
                train_metrics["train_cd_coarse"],
                train_metrics["train_cd_fine"],
                val_metrics["val_loss"],
                val_metrics["val_cd_coarse"],
                val_metrics["val_cd_fine"],
                val_metrics.get("val_fscore", ""),
                lr,
                f"{elapsed:.1f}",
            ]
        )
        log_file.flush()

        # Save checkpoints
        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            all_metrics,
            os.path.join(config["checkpoint_dir"], "pcn_last.pth"),
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                all_metrics,
                os.path.join(config["checkpoint_dir"], "pcn_best.pth"),
            )
            print(f"  → new best (val_loss={best_val_loss:.6f})")

        if (epoch + 1) % config["checkpoint_every"] == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                all_metrics,
                os.path.join(config["checkpoint_dir"], f"pcn_epoch{epoch+1:03d}.pth"),
            )

    log_file.close()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {config['checkpoint_dir']}")


if __name__ == "__main__":
    main()
