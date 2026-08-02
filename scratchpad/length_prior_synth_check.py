"""Direction 2 — synthetic mechanism check for the longitudinal (length) prior.

`estimate_canonical_frame()` has a width prior (X) and an up-shift (Y) but NO
length prior (Z): the Z center is the observed bbox midpoint, so a partial that
is truncated at the occluded far end gets normalized around the *near* portion
and PCN's full-car output is placed short of the true far end (donor metric
Finding #32: far_end cov 0.133, the worst region).

This script tests the MECHANISM on synthetic val cars, where we have true GT:
does pushing the Z center toward the occluded far end (extend-only, to a length
prior) and re-normalizing make PCN cover the true far end better, without
wrecking overall completion? We use the TRUE far-end sign here (ceiling) — the
sign proxy (ego direction) is separately self-validated on real data, because
the donor metric's far_end region is itself ego-defined.

Width (true sign, W_PRIOR) and up-shift are held at production values in EVERY
condition, so the only variable is the length push. Isolates the length prior.

Run:
    .venv\\Scripts\\python.exe scratchpad\\length_prior_synth_check.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pcn import ShapeNetCompletionDataset, TRAIN_CONFIG  # noqa: E402
from pcn import PCN  # noqa: E402
from completion import (  # noqa: E402
    chamfer_distance, f_score,
    COMPLETION_SCALE_CORRECTION, COMPLETION_CAR_WIDTH_PRIOR, COMPLETION_UP_SHIFT,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")
CAR_SYNSET = "02958343"
N_EVAL = 40
PCN_N_INPUT = 256
COV_THRESH = 0.10
SEED = 0

# Length priors to test (m). None = no length push (current production behavior).
# "true" = per-car GT length extent (the ceiling); the fixed values probe transfer
# robustness (4.14 = real amodal-GT median #29; 4.5 = synthetic nominal scale).
L_PRIORS = [None, "true", 4.5, 4.14, 3.8]


def raw_sample(ds, idx):
    """(gt, partial) in the canonical mesh frame (X=width, Y=up, Z=length)."""
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
    """Index set to resize a cloud to n_want (shared across conditions)."""
    if n_have == n_want:
        return np.arange(n_have)
    if n_have > n_want:
        return rng.choice(n_have, n_want, replace=False)
    return np.concatenate([np.arange(n_have),
                           rng.choice(n_have, n_want - n_have, replace=True)])


def est_center(partial, gt_c, Lp):
    """Full-car center in the canonical frame, mirroring estimate_canonical_frame.

    Width (true occluded sign) and up-shift are always applied. The length push
    (Z) is the variable under test; Lp=None disables it (production today).
    ``Lp`` is the numeric length prior in metres (caller resolves "true"->GT L).
    """
    bb = 0.5 * (partial.min(0) + partial.max(0))
    ext = partial.max(0) - partial.min(0)
    c = bb.copy()
    # width (X): push toward the true occluded side to the half-width prior
    w_sign = np.sign((gt_c - bb)[0]) or 1.0
    c[0] = bb[0] + w_sign * max(0.5 * COMPLETION_CAR_WIDTH_PRIOR - 0.5 * ext[0], 0.0)
    # up (Y): undo the ground-cut upward bias
    c[1] = bb[1] - COMPLETION_UP_SHIFT
    # length (Z): extend-only push toward the true occluded far end
    if Lp is not None:
        l_sign = np.sign((gt_c - bb)[2]) or 1.0
        c[2] = bb[2] + l_sign * max(0.5 * float(Lp) - 0.5 * ext[2], 0.0)
    return c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)  # _render_kitti_like uses global np.random

    config = TRAIN_CONFIG.copy()
    config["kitti_like_partial"] = True
    ds = ShapeNetCompletionDataset(config, split="val")
    car_items = [i for i, (_, syn) in enumerate(ds.items) if syn == CAR_SYNSET]
    print(f"Val car models: {len(car_items)}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = PCN(num_coarse=1024, grid_size=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {os.path.basename(CKPT)} "
          f"(val_loss={ckpt['metrics'].get('val_loss'):.5f})")

    def run(partial_sel, center):
        radius = float(np.linalg.norm(partial_sel - center, axis=1).max()) \
            / COMPLETION_SCALE_CORRECTION
        p_norm = ((partial_sel - center) / radius).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(p_norm).unsqueeze(0).to(device)
            _, fine = model(inp)
            pred = fine.squeeze(0).cpu().numpy()
        return pred * radius + center

    # per-condition accumulators: cd, f, far_end_cov, signed far-reach error
    keys = ["none" if lp is None else str(lp) for lp in L_PRIORS]
    acc = {k: {"cd": [], "f": [], "far_cov": [], "reach_err": []} for k in keys}

    n_done = 0
    for idx in car_items:
        s = raw_sample(ds, idx)
        if s is None:
            continue
        gt, partial = s
        gt_c = gt.mean(0)
        L_gt = float(np.ptp(gt[:, 2]))

        # true occluded far-end direction and the GT far quarter
        bb2 = 0.5 * (partial[:, 2].min() + partial[:, 2].max())
        l_sign = np.sign(gt_c[2] - bb2) or 1.0
        d_len = (gt[:, 2] - gt_c[2]) * l_sign          # + = toward far end
        far_mask = d_len > 0.25 * L_gt
        if far_mask.sum() < 20:
            continue                                   # symmetric view, no far end
        gt_far = float(d_len.max())                    # GT reach toward far end

        sel = fixed_indices(len(partial), PCN_N_INPUT, rng)
        partial_sel = partial[sel].astype(np.float64)

        for lp, k in zip(L_PRIORS, keys):
            Lp = None if lp is None else (L_gt if lp == "true" else float(lp))
            pred = run(partial_sel, est_center(partial_sel, gt_c, Lp))
            acc[k]["cd"].append(chamfer_distance(pred, gt))
            acc[k]["f"].append(f_score(pred, gt, threshold=COV_THRESH))
            from scipy.spatial import cKDTree
            d_far, _ = cKDTree(pred).query(gt[far_mask])
            acc[k]["far_cov"].append(float((d_far < COV_THRESH).mean()))
            pred_reach = float(((pred[:, 2] - gt_c[2]) * l_sign).max())
            acc[k]["reach_err"].append(pred_reach - gt_far)

        n_done += 1
        if n_done >= N_EVAL:
            break

    print(f"\n==== Length-prior mechanism check "
          f"(synthetic val cars, n={n_done}, true far-end sign) ====")
    print(f"{'L_prior':>8} {'CD(m)':>8} {'F@0.1':>7} {'far_cov':>8} "
          f"{'reach_err(m)':>13}")
    for k in keys:
        a = acc[k]
        print(f"{k:>8} {np.median(a['cd']):>8.4f} {np.median(a['f']):>7.3f} "
              f"{np.median(a['far_cov']):>8.3f} {np.median(a['reach_err']):>13.3f}")
    print("\nfar_cov = GT far-quarter coverage @0.1 m (per-car median; higher better)")
    print("reach_err = pred far-reach - GT far-reach (m; <0 = under-extends)")
    print("'none' = current production (no length push); 'true' = per-car GT length (ceiling)")


if __name__ == "__main__":
    main()
