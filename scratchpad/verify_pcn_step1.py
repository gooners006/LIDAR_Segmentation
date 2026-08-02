"""PCN verification — Step 1: synthetic in-distribution sanity + normalization calibration.

Goal: isolate "is the model good in-distribution?" from "does the inference
normalization in completion.py break a known-good input?"

For the KITTI-like synthetic val set we have BOTH the partial and the GT, so we can:
  A. Measure the orientation (up-axis) and scale/centroid relationship that
     training assumes (the numbers completion.py must match).
  B. Run pcn_kitti_best three ways on the SAME partials and compare CD/F-score:
       1. TRAINING normalization (center on GT centroid, /GT radius, no PCA)  -> lower bound
       2. completion.py complete()  (center on partial centroid, PCA-align, /partial radius)
       3. CORRECTED (center on partial centroid, no PCA, /scale-corrected radius)

All CD/F-scores are computed in metres (we un-normalize before comparing).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pcn import ShapeNetCompletionDataset, TRAIN_CONFIG  # noqa: E402
from pcn import PCN  # noqa: E402
from completion import PointCloudCompleter, chamfer_distance, f_score  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")
CAR_SYNSET = "02958343"
N_CALIB = 80
N_EVAL = 30
PCN_N_INPUT = 256


def raw_sample(ds, idx):
    """Return (gt_pts, partial_pts, synset) in the mesh frame, no normalization."""
    obj_path, synset_id = ds.items[idx]
    scale_m = ds.config["category_scale_m"][synset_id]
    mesh = ds._load_and_scale(obj_path, scale_m)
    if mesh is None or mesh.get_surface_area() < 1e-6:
        return None
    gt = ds._sample_gt(mesh)
    partial = ds._render_kitti_like(mesh, scale_m)
    if partial is None or len(partial) < 16:
        return None
    return gt, partial, synset_id


def fix_size(pts, n, rng):
    pts = pts.astype(np.float32)
    if len(pts) == n:
        return pts
    if len(pts) > n:
        return pts[rng.choice(len(pts), n, replace=False)]
    return np.vstack([pts, pts[rng.choice(len(pts), n - len(pts), replace=True)]])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    config = TRAIN_CONFIG.copy()
    config["kitti_like_partial"] = True
    ds = ShapeNetCompletionDataset(config, split="val")
    car_items = [i for i, (_, syn) in enumerate(ds.items) if syn == CAR_SYNSET]
    print(f"Val car models: {len(car_items)}")

    # ---- Load model (training-normalized path) ----
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = PCN(num_coarse=1024, grid_size=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {os.path.basename(CKPT)} | best val_loss={ckpt['metrics'].get('val_loss'):.5f}")

    # completion.py path (PCA + partial-radius)
    completer = PointCloudCompleter(model_path=CKPT, seed=0)

    # ---------------- A. Calibration ----------------
    ext_axes, ratios, offsets, gt_radii = [], [], [], []
    samples = []
    for idx in car_items:
        s = raw_sample(ds, idx)
        if s is None:
            continue
        gt, partial, _ = s
        gt_c = gt.mean(0)
        gt_cen = gt - gt_c
        gt_r = float(np.linalg.norm(gt_cen, axis=1).max())
        p_c = partial.mean(0)
        p_self_r = float(np.linalg.norm(partial - p_c, axis=1).max())

        ext = gt.max(0) - gt.min(0)
        ext_axes.append(ext)
        ratios.append(p_self_r / gt_r)
        offsets.append(float(np.linalg.norm(p_c - gt_c)) / gt_r)  # normalized by gt_r
        gt_radii.append(gt_r)
        samples.append((gt, partial, gt_c, gt_cen, gt_r))
        if len(samples) >= N_CALIB:
            break

    ext_axes = np.array(ext_axes)
    ratios = np.array(ratios)
    offsets = np.array(offsets)
    mean_extent = ext_axes.mean(0)
    print("\n==== A. CALIBRATION (synthetic val cars, n={}) ====".format(len(samples)))
    print(f"GT per-axis extent (m)  X={mean_extent[0]:.2f} Y={mean_extent[1]:.2f} Z={mean_extent[2]:.2f}")
    print(f"  -> smallest-extent axis (likely 'up'): {['X','Y','Z'][int(mean_extent.argmin())]}")
    print(f"  -> largest-extent axis (likely length): {['X','Y','Z'][int(mean_extent.argmax())]}")
    print(f"partial_radius / gt_radius : mean={ratios.mean():.3f} median={np.median(ratios):.3f} "
          f"std={ratios.std():.3f}  (->scale correction factor)")
    print(f"|partial_centroid - gt_centroid| / gt_radius : mean={offsets.mean():.3f} "
          f"median={np.median(offsets):.3f}")
    scale_correction = float(np.median(ratios))  # divide partial_radius by this to approx gt_radius

    # ---------------- B. In-distribution model quality, 3 paths ----------------
    print("\n==== B. IN-DISTRIBUTION COMPLETION QUALITY (n={}) ====".format(min(N_EVAL, len(samples))))

    def run_model_norm(partial, center, radius):
        """Normalize partial by (center, radius), run model, un-normalize. No PCA."""
        p_norm = (partial - center) / radius
        p_fixed = fix_size(p_norm, PCN_N_INPUT, rng)
        with torch.no_grad():
            inp = torch.from_numpy(p_fixed).float().unsqueeze(0).to(device)
            _, fine = model(inp)
            pred = fine.squeeze(0).cpu().numpy()
        return pred * radius + center

    results = {k: ([], []) for k in
               ["1.train", "2.complete()", "3.partialC+scaleR",
                "3a.partialC+trueR", "3b.gtC+scaleR", "3c.gtC+trueR",
                "3d.bboxC+scaleR", "3e.bboxC+widthSearch",
                "4a.widthpush(0.96)", "4b.widthpush+upshift", "4c.widthpush(ratio)",
                "4d.upshiftonly", "4e.trueWidthSign+up", "4f.up0.25",
                "4g.trueSign+up0.35", "4h.trueSign+up0.45"]}

    def est_center(partial, half_width=None, up_shift=0.0, width_ratio=None):
        """Estimate full-car center from a train-frame partial (X=width,Y=up,Z=length)."""
        bb = 0.5 * (partial.min(0) + partial.max(0))
        ext = partial.max(0) - partial.min(0)
        c = bb.copy()
        if width_ratio is not None:
            half_width = 0.5 * width_ratio * ext[2]  # width ~ ratio * observed length
        if half_width is not None:
            d = bb - partial.mean(0)                  # visible mass -> occluded side
            sign = np.sign(d[0]) if abs(d[0]) > 1e-9 else 1.0
            push = max(half_width - 0.5 * ext[0], 0.0)
            c[0] = bb[0] + sign * push
        c[1] = bb[1] - up_shift                        # undo ground-cut upward bias
        return c

    def add(key, pred, gt):
        results[key][0].append(chamfer_distance(pred, gt))
        results[key][1].append(f_score(pred, gt, threshold=0.10))

    for gt, partial, gt_c, gt_cen, gt_r in samples[:N_EVAL]:
        p_c = partial.mean(0)
        p_self_r = float(np.linalg.norm(partial - p_c, axis=1).max())
        scale_r = p_self_r / scale_correction  # estimate of gt_r from partial

        # 1. TRAINING normalization (GT centroid, GT radius)
        add("1.train", run_model_norm(partial, gt_c, gt_r), gt)
        # 2. completion.py complete()
        completed, skip = completer.complete(partial, "car")
        if skip is None:
            add("2.complete()", completed, gt)
        # 3 / ablations: factorial over centroid {partial, gt} x radius {scale-est, true}
        add("3.partialC+scaleR", run_model_norm(partial, p_c, scale_r), gt)
        add("3a.partialC+trueR", run_model_norm(partial, p_c, gt_r), gt)
        add("3b.gtC+scaleR", run_model_norm(partial, gt_c, scale_r), gt)
        add("3c.gtC+trueR", run_model_norm(partial, gt_c, gt_r), gt)
        bbox_c = 0.5 * (partial.min(0) + partial.max(0))
        add("3d.bboxC+scaleR", run_model_norm(partial, bbox_c, scale_r), gt)

        # 3e: search center along occluded direction (bbox_c - partial_centroid),
        #     pick offset by best partial->pred fit (GT-free, transfers to real data)
        d = bbox_c - p_c
        nd = np.linalg.norm(d)
        d = d / nd if nd > 1e-6 else np.array([1.0, 0.0, 0.0])
        best_pred, best_fit = None, np.inf
        for off in np.linspace(-0.2, 0.8, 11) * scale_r:
            c = bbox_c + off * d
            pred = run_model_norm(partial, c, scale_r)
            fit = cKDTree(pred).query(partial)[0].mean()  # one-sided partial->pred
            if fit < best_fit:
                best_fit, best_pred = fit, pred
        add("3e.bboxC+widthSearch", best_pred, gt)

        # 4: geometric center estimators (partial-only, transfer to real data)
        add("4a.widthpush(0.96)",
            run_model_norm(partial, est_center(partial, half_width=0.96), scale_r), gt)
        add("4b.widthpush+upshift",
            run_model_norm(partial, est_center(partial, half_width=0.96, up_shift=0.15), scale_r), gt)
        add("4c.widthpush(ratio)",
            run_model_norm(partial, est_center(partial, width_ratio=0.43, up_shift=0.15), scale_r), gt)
        # 4d: up-shift only (no width push) -> isolates the ground-cut correction
        c4d = 0.5 * (partial.min(0) + partial.max(0)); c4d[1] -= 0.15
        add("4d.upshiftonly", run_model_norm(partial, c4d, scale_r), gt)
        # 4e: width push with TRUE occluded sign (ceiling; real data gets this from ego dir) + up-shift
        bb = 0.5 * (partial.min(0) + partial.max(0)); ext = partial.max(0) - partial.min(0)
        c4e = bb.copy()
        true_sign = np.sign((gt_c - bb)[0]) or 1.0
        c4e[0] = bb[0] + true_sign * max(0.96 - 0.5 * ext[0], 0.0)
        c4e[1] -= 0.15
        add("4e.trueWidthSign+up", run_model_norm(partial, c4e, scale_r), gt)
        # 4f: stronger up-shift
        add("4f.up0.25",
            run_model_norm(partial, est_center(partial, half_width=0.96, up_shift=0.25), scale_r), gt)
        for key, up in [("4g.trueSign+up0.35", 0.35), ("4h.trueSign+up0.45", 0.45)]:
            c = bb.copy()
            c[0] = bb[0] + true_sign * max(0.96 - 0.5 * ext[0], 0.0)
            c[1] -= up
            add(key, run_model_norm(partial, c, scale_r), gt)

    print("(Chamfer distance in metres, lower is better; F-score higher is better)")
    for key, (cd, fs) in results.items():
        cd, fs = np.array(cd), np.array(fs)
        print(f"  {key:22s} CD(m)={cd.mean():.4f}+-{cd.std():.4f}  F@0.1m={fs.mean():.3f}")


if __name__ == "__main__":
    main()
