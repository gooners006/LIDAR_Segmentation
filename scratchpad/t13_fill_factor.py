"""T13/Step 1c D2 — measure PCN's per-axis under-fill on synthetic true GT.

Fill factor = GT extent / PCN-output extent, per canonical axis (X=width,
Y=up, Z=length), measured on KITTI-like synthetic val cars where true GT exists.
Measured under the D1 condition that ships: the normalization radius is computed
against the center BEFORE the Z length-push (decouple_radius), so pred scale
(hence extent) is independent of the length target — the fill factor is a pure
property of PCN's frame-filling, calibrated once and applied (not refit) on real.

Pre-registered widen rule (t13_step1c_plan.md D2): default to correcting Z only;
adopt a W or H correction ONLY if that axis's median fill exceeds 1.10.

Run: .venv\\Scripts\\python.exe scratchpad/t13_fill_factor.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pcn import ShapeNetCompletionDataset, TRAIN_CONFIG  # noqa: E402
from pcn import PCN  # noqa: E402
from completion import (  # noqa: E402
    COMPLETION_SCALE_CORRECTION, COMPLETION_CAR_WIDTH_PRIOR, COMPLETION_UP_SHIFT,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")
CAR_SYNSET = "02958343"
PCN_N_INPUT = 256
SEED = 0
N_MAX = 300  # use as many val car models as available (large-n calibration)


def raw_sample(ds, idx):
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


def fixed_indices(n_have, n_want, rng):
    if n_have == n_want:
        return np.arange(n_have)
    if n_have > n_want:
        return rng.choice(n_have, n_want, replace=False)
    return np.concatenate([np.arange(n_have),
                           rng.choice(n_have, n_want - n_have, replace=True)])


def center_pre_z(partial, gt_c):
    """Center with production X-width + Y-up adjustments, NO Z push (= the D1
    radius center). True occluded sign used for width (synthetic ceiling)."""
    bb = 0.5 * (partial.min(0) + partial.max(0))
    ext = partial.max(0) - partial.min(0)
    c = bb.copy()
    w_sign = np.sign((gt_c - bb)[0]) or 1.0
    c[0] = bb[0] + w_sign * max(0.5 * COMPLETION_CAR_WIDTH_PRIOR - 0.5 * ext[0], 0.0)
    c[1] = bb[1] - COMPLETION_UP_SHIFT
    return c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    config = TRAIN_CONFIG.copy()
    config["kitti_like_partial"] = True
    ds = ShapeNetCompletionDataset(config, split="val")
    car_items = [i for i, (_, syn) in enumerate(ds.items) if syn == CAR_SYNSET]
    print(f"Val car models: {len(car_items)}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = PCN(num_coarse=1024, grid_size=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fills = {"X": [], "Y": [], "Z": []}  # GT_ext / pred_ext per axis
    n = 0
    for idx in car_items:
        s = raw_sample(ds, idx)
        if s is None:
            continue
        gt, partial = s
        gt_c = gt.mean(0)
        sel = fixed_indices(len(partial), PCN_N_INPUT, rng)
        p = partial[sel].astype(np.float64)

        c = center_pre_z(p, gt_c)
        radius = float(np.linalg.norm(p - c, axis=1).max()) / COMPLETION_SCALE_CORRECTION
        if radius < 1e-6:
            continue
        p_norm = ((p - c) / radius).astype(np.float32)
        with torch.no_grad():
            _, fine = model(torch.from_numpy(p_norm).unsqueeze(0).to(device))
            fine = fine.squeeze(0).cpu().numpy()
        pred_ext = np.ptp(fine, axis=0) * radius  # metric extent of PCN output
        gt_ext = np.ptp(gt, axis=0)
        for ax, k in zip(range(3), ["X", "Y", "Z"]):
            if pred_ext[ax] > 1e-6:
                fills[k].append(gt_ext[ax] / pred_ext[ax])
        n += 1
        if n >= N_MAX:
            break

    print(f"\n==== Per-axis fill factor (GT extent / PCN output extent), n={n} ====")
    print(f"{'axis':>6} {'median':>8} {'mean':>8} {'p25':>7} {'p75':>7}")
    med = {}
    for k in ["X", "Y", "Z"]:
        a = np.array(fills[k])
        med[k] = float(np.median(a))
        print(f"{k:>6} {np.median(a):>8.3f} {np.mean(a):>8.3f} "
              f"{np.percentile(a,25):>7.3f} {np.percentile(a,75):>7.3f}")
    print("\n(X=width, Y=up, Z=LENGTH). fill>1 = PCN under-fills that axis.")
    print(f"Widen rule (>1.10): "
          f"X {'WIDEN' if med['X']>1.10 else 'no'}, "
          f"Y {'WIDEN' if med['Y']>1.10 else 'no'}, "
          f"Z (length; always corrected) fill_z = {med['Z']:.3f}")


if __name__ == "__main__":
    main()
