"""Matched synthetic eval: PCN vs PoinTr under the verify_pcn_step1 protocol.

Motivation: Finding #28's synthetic table compares PCN's val_loss (0.1246,
coarse + 0.5*fine) against PoinTr's val_cd_fine (0.0634) — different columns —
and PCN's F@0.1m (0.76) came from this meter-scale protocol while PoinTr's
(0.987) came from the training log's normalized-frame F-score. This script runs
BOTH checkpoints through the exact same verify_pcn_step1.py protocol on
literally identical inputs, so the numbers are finally like-for-like.

Paths evaluated (subset of verify_pcn_step1, the two that matter):
  1.train           — GT centroid + GT radius (in-distribution lower bound;
                      the row that produced PCN's documented CD 0.16 / F 0.76)
  3.partialC+scaleR — partial centroid + calibrated-scale radius (the GT-free
                      corrected inference path that transfers to real data)

All CD/F-scores in metres (un-normalized), same chamfer_distance/f_score as
completion.py. np.random is seeded (verify_pcn_step1 did not seed it, so exact
sample-level reproduction of the June numbers is not possible; n=30 means small
drift in the PCN row is expected and itself informative).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pcn import ShapeNetCompletionDataset, TRAIN_CONFIG  # noqa: E402
from train_pointr import build_model as build_pointr  # noqa: E402
from pcn import PCN  # noqa: E402
from completion import chamfer_distance, f_score  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCN_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")
POINTR_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "pointr_kitti_best.pth")
CAR_SYNSET = "02958343"
N_CALIB = 80
N_EVAL = 30
N_INPUT = 256  # both models trained on 256-point partials


def raw_sample(ds, idx):
    """Return (gt_pts, partial_pts) in the mesh frame, no normalization."""
    obj_path, synset_id = ds.items[idx]
    scale_m = ds.config["category_scale_m"][synset_id]
    mesh = ds._load_and_scale(obj_path, scale_m)
    if mesh is None or mesh.get_surface_area() < 1e-6:
        return None
    gt = ds._sample_gt(mesh)
    partial = ds._render_kitti_like(mesh, scale_m)
    if partial is None or len(partial) < 16:
        return None
    return gt, partial


def fix_size(pts, n, rng):
    pts = pts.astype(np.float32)
    if len(pts) == n:
        return pts
    if len(pts) > n:
        return pts[rng.choice(len(pts), n, replace=False)]
    return np.vstack([pts, pts[rng.choice(len(pts), n - len(pts), replace=True)]])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)  # dataset sampling (verify_pcn_step1 left this unseeded)
    rng = np.random.default_rng(0)

    config = TRAIN_CONFIG.copy()
    config["kitti_like_partial"] = True
    ds = ShapeNetCompletionDataset(config, split="val")
    car_items = [i for i, (_, syn) in enumerate(ds.items) if syn == CAR_SYNSET]
    print(f"Val car models: {len(car_items)}")

    # ---- Load both models ----
    pcn_ckpt = torch.load(PCN_CKPT, map_location=device, weights_only=False)
    pcn = PCN(num_coarse=1024, grid_size=2).to(device)
    pcn.load_state_dict(pcn_ckpt["model_state_dict"])
    pcn.eval()
    print(f"Loaded {os.path.basename(PCN_CKPT)} "
          f"(epoch {pcn_ckpt.get('epoch')}, val_loss={pcn_ckpt['metrics'].get('val_loss'):.5f})")

    pointr_ckpt = torch.load(POINTR_CKPT, map_location=device, weights_only=False)
    pointr = build_pointr(device)
    pointr.load_state_dict(pointr_ckpt["model_state_dict"])
    pointr.eval()
    print(f"Loaded {os.path.basename(POINTR_CKPT)} "
          f"(epoch {pointr_ckpt.get('epoch')}, val_loss={pointr_ckpt['metrics'].get('val_loss'):.5f})")

    models = {"PCN": pcn, "PoinTr": pointr}

    # ---- Collect samples + calibration (same as verify_pcn_step1 section A) ----
    ratios, samples = [], []
    for idx in car_items:
        s = raw_sample(ds, idx)
        if s is None:
            continue
        gt, partial = s
        gt_c = gt.mean(0)
        gt_r = float(np.linalg.norm(gt - gt_c, axis=1).max())
        p_c = partial.mean(0)
        p_self_r = float(np.linalg.norm(partial - p_c, axis=1).max())
        ratios.append(p_self_r / gt_r)
        samples.append((gt, partial, gt_c, gt_r))
        if len(samples) >= N_CALIB:
            break
    scale_correction = float(np.median(ratios))
    print(f"\nCalibration (n={len(samples)}): partial_r/gt_r median = {scale_correction:.3f}")

    # ---- Matched eval: identical normalized inputs to both models ----
    paths = ["1.train", "3.partialC+scaleR"]
    results = {(m, p): {"cd": [], "fs": [], "cdn": []} for m in models for p in paths}

    def run(model, p_fixed_norm, radius, center):
        with torch.no_grad():
            inp = torch.from_numpy(p_fixed_norm).float().unsqueeze(0).to(device)
            _, fine = model(inp)
        return fine.squeeze(0).cpu().numpy() * radius + center

    for gt, partial, gt_c, gt_r in samples[:N_EVAL]:
        p_c = partial.mean(0)
        p_self_r = float(np.linalg.norm(partial - p_c, axis=1).max())
        scale_r = p_self_r / scale_correction

        for path, center, radius in [
            ("1.train", gt_c, gt_r),
            ("3.partialC+scaleR", p_c, scale_r),
        ]:
            # fix_size ONCE per (sample, path): both models see identical input
            p_fixed = fix_size((partial - center) / radius, N_INPUT, rng)
            for name, model in models.items():
                pred = run(model, p_fixed, radius, center)
                cd = chamfer_distance(pred, gt)
                r = results[(name, path)]
                r["cd"].append(cd)
                r["fs"].append(f_score(pred, gt, threshold=0.10))
                r["cdn"].append(cd / radius)  # normalized-frame CD (log cross-check)

    print(f"\n==== MATCHED EVAL (n={min(N_EVAL, len(samples))}, metres) ====")
    print(f"{'path':20s} {'model':8s} {'CD(m)':>16s} {'F@0.1m':>8s} {'CD(norm)':>10s}")
    for path in paths:
        for name in models:
            r = results[(name, path)]
            cd, fs, cdn = np.array(r["cd"]), np.array(r["fs"]), np.array(r["cdn"])
            print(f"{path:20s} {name:8s} {cd.mean():8.4f}+-{cd.std():.4f} "
                  f"{fs.mean():8.3f} {cdn.mean():10.4f}")

    # Per-sample paired deltas on the headline path
    d = (np.array(results[("PCN", "1.train")]["cd"])
         - np.array(results[("PoinTr", "1.train")]["cd"]))
    wins = int((d > 0).sum())
    print(f"\n1.train paired per-sample: PoinTr better CD on {wins}/{len(d)} samples; "
          f"mean CD delta (PCN-PoinTr) = {d.mean():+.4f} m")


if __name__ == "__main__":
    main()
