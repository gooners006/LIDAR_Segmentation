"""Train PoinTr on ShapeNet for point cloud completion.

Reuses the exact ShapeNet dataset / KITTI-like partial generator from
``train_pcn.py`` so PoinTr is trained and validated on the *same* data as the
PCN baseline — the only change is the model and its loss. This keeps the
PoinTr-vs-PCN comparison clean per the experiment protocol.

Usage:
    python src/train_pointr.py --kitti-like --epochs 100
    python src/train_pointr.py --kitti-like --eval-only --resume checkpoints/pointr_kitti_best.pth
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from completion import f_score
from pointr import PoinTr, pointr_loss
from train_pcn import TRAIN_CONFIG, ShapeNetCompletionDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PoinTr-specific config; data params are inherited from train_pcn.TRAIN_CONFIG.
POINTR_CONFIG = {
    "num_proxies": 64,
    "num_query": 256,
    "trans_dim": 256,
    "num_heads": 4,
    "enc_depth": 4,
    "dec_depth": 4,
    "grid_size": 4,      # fine missing = num_query * grid_size**2 = 256*16 = 4096
    "knn_dgcnn": 16,
    "knn_geo": 8,
    # Optimisation — AdamW per the PoinTr paper (Supp. A).
    "alpha": 1.0,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "lr_decay_step": 40,
    "lr_decay_gamma": 0.5,
    "epochs": 100,
    "batch_size": 16,   # peak ~3.8 GB VRAM; chunked-CD loop makes 16 the throughput sweet spot
    "num_workers": 4,
    "seed": 42,
    "log_every": 50,
    "checkpoint_every": 10,
}


def build_model(device: torch.device) -> PoinTr:
    """Instantiate the PoinTr model from POINTR_CONFIG."""
    return PoinTr(
        num_proxies=POINTR_CONFIG["num_proxies"],
        num_query=POINTR_CONFIG["num_query"],
        trans_dim=POINTR_CONFIG["trans_dim"],
        num_heads=POINTR_CONFIG["num_heads"],
        enc_depth=POINTR_CONFIG["enc_depth"],
        dec_depth=POINTR_CONFIG["dec_depth"],
        grid_size=POINTR_CONFIG["grid_size"],
        knn_dgcnn=POINTR_CONFIG["knn_dgcnn"],
        knn_geo=POINTR_CONFIG["knn_geo"],
    ).to(device)


def train_one_epoch(model, loader, optimizer, device, config, epoch):
    """Run one training epoch. Returns dict of average losses."""
    model.train()
    total_loss = total_cd_c = total_cd_f = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
    for batch in pbar:
        partial = batch["partial"].to(device)
        gt = batch["gt"].to(device)

        coarse, fine = model(partial)
        loss, cd_c, cd_f = pointr_loss(coarse, fine, gt, config["alpha"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cd_c += cd_c.item()
        total_cd_f += cd_f.item()
        n_batches += 1

        if n_batches % config["log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.5f}", cd_f=f"{cd_f.item():.5f}")

    return {
        "train_loss": total_loss / n_batches,
        "train_cd_coarse": total_cd_c / n_batches,
        "train_cd_fine": total_cd_f / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, config, compute_fscore: bool = False):
    """Evaluate on the validation set. Returns dict of metrics."""
    model.eval()
    total_loss = total_cd_c = total_cd_f = 0.0
    n_batches = 0
    fscores = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        partial = batch["partial"].to(device)
        gt = batch["gt"].to(device)

        coarse, fine = model(partial)
        loss, cd_c, cd_f = pointr_loss(coarse, fine, gt, config["alpha"])

        total_loss += loss.item()
        total_cd_c += cd_c.item()
        total_cd_f += cd_f.item()
        n_batches += 1

        if compute_fscore and n_batches <= 20:
            fine_np = fine.cpu().numpy()
            gt_np = gt.cpu().numpy()
            fscores.extend(
                f_score(f, g, threshold=0.1) for f, g in zip(fine_np, gt_np)
            )

    metrics = {
        "val_loss": total_loss / n_batches,
        "val_cd_coarse": total_cd_c / n_batches,
        "val_cd_fine": total_cd_f / n_batches,
    }
    if fscores:
        metrics["val_fscore"] = float(np.mean(fscores))
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, path):
    """Save a training checkpoint (includes both data and model config)."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,           # data config (kitti-like params etc.)
            "pointr_config": POINTR_CONFIG,
        },
        path,
    )


def main():
    """Train or evaluate PoinTr on ShapeNet partial→complete pairs."""
    parser = argparse.ArgumentParser(description="Train PoinTr on ShapeNet")
    parser.add_argument("--epochs", type=int, default=POINTR_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=POINTR_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=POINTR_CONFIG["lr"])
    parser.add_argument("--workers", type=int, default=POINTR_CONFIG["num_workers"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--kitti-like", action="store_true",
                        help="Use KITTI-like single-view partials (matches the real "
                             "single-frame clusters; see docs/pcn/kitti_like_partial.md)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Override checkpoint/log filename prefix")
    args = parser.parse_args()

    # Data config inherited from PCN so the comparison is apples-to-apples.
    config = TRAIN_CONFIG.copy()
    config["kitti_like_partial"] = args.kitti_like
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["num_workers"] = args.workers
    config["alpha"] = POINTR_CONFIG["alpha"]
    config["log_every"] = POINTR_CONFIG["log_every"]
    config["checkpoint_every"] = POINTR_CONFIG["checkpoint_every"]
    if args.kitti_like:
        print("KITTI-like single-view partial generation enabled")

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    torch.manual_seed(POINTR_CONFIG["seed"])
    np.random.seed(POINTR_CONFIG["seed"])

    train_ds = ShapeNetCompletionDataset(config, split="train")
    val_ds = ShapeNetCompletionDataset(config, split="val")
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PoinTr parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=POINTR_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=POINTR_CONFIG["lr_decay_step"],
        gamma=POINTR_CONFIG["lr_decay_gamma"],
    )

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

    os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)
    ckpt_prefix = args.tag or ("pointr_kitti" if args.kitti_like else "pointr")
    ckpt_dir = TRAIN_CONFIG["checkpoint_dir"]
    log_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_training_log.csv")
    log_exists = os.path.isfile(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "epoch", "train_loss", "train_cd_coarse", "train_cd_fine",
            "val_loss", "val_cd_coarse", "val_cd_fine", "val_fscore", "lr", "time_s",
        ])

    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, epoch)
        scheduler.step()

        compute_fs = (epoch + 1) % 10 == 0 or epoch == config["epochs"] - 1
        val_metrics = evaluate(model, val_loader, device, config, compute_fscore=compute_fs)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"train={train_metrics['train_loss']:.5f} | "
            f"val={val_metrics['val_loss']:.5f} | "
            f"cd_f={val_metrics['val_cd_fine']:.5f}"
            + (f" | F@0.1={val_metrics['val_fscore']:.3f}" if "val_fscore" in val_metrics else "")
            + f" | lr={lr:.2e} | {elapsed:.0f}s"
        )

        log_writer.writerow([
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
        ])
        log_file.flush()

        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config,
                        os.path.join(ckpt_dir, f"{ckpt_prefix}_last.pth"))

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config,
                            os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth"))
            print(f"  -> new best (val_loss={best_val_loss:.6f})")

        if (epoch + 1) % config["checkpoint_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config,
                            os.path.join(ckpt_dir, f"{ckpt_prefix}_epoch{epoch+1:03d}.pth"))

    log_file.close()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
