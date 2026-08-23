"""Thesis Fig 6.2 (box-overlay version) -- the inference-normalization fix in BEV.

Same idea as completion_box_eval_viz.py (GT / raw / completed boxes in bird's-eye
view) but the comparison is OLD vs CORRECTED completion, to illustrate Finding #26:
  - GT amodal box            (black)
  - OLD completion box       (red)   -- retired PCA + partial-radius normalization
  - CORRECTED completion box (green) -- production canonical-frame path
over the raw cluster points (gray), for a few representative static cars. BEV box
IoU vs the amodal GT is printed per panel, so "pose/scale recovery" is quantified,
not asserted. The old path is reconstructed (no longer a code path in
completion.py); nothing frozen is touched.

Output: output/experiments/fig62_deblob/box_overlay_deblob_<seq>.png
Run:    .venv\\Scripts\\python.exe scratchpad/fig62_box_overlay.py --seq 08
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from completion_box_eval import match_pairs, world_box  # noqa: E402
from completion_box_eval_step2 import bev_iou, box_corners_xz, pair_metrics  # noqa: E402
from classifier import load_classifier  # noqa: E402
from completion import PointCloudCompleter  # noqa: E402
from evaluate import THING_CLASSES_SUPPORTED, get_frame_detections  # noqa: E402
from pipeline import load_calib, load_poses  # noqa: E402
from pcn import PCN  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCN_N_INPUT = 256


def _fix_size(pts, n, rng):
    pts = pts.astype(np.float32)
    if len(pts) >= n:
        return pts[rng.choice(len(pts), n, replace=False)]
    return np.vstack([pts, pts[rng.choice(len(pts), n - len(pts), replace=True)]])


def complete_old_blob(model, device, partial_velo, rng):
    """Retired normalization: 3D PCA align + partial centroid + partial radius.

    No full-car-center estimate, no scale correction -- the path that produced
    blobs on real data (Finding #26). Reconstructed for the figure only; this is
    no longer a code path in completion.py.
    """
    pts = np.asarray(partial_velo, dtype=np.float64)
    if len(pts) < 16:
        return None
    c = pts.mean(0)                       # partial centroid (not the car center)
    X = pts - c
    cov = X.T @ X
    _, V = np.linalg.eigh(cov)            # 3D PCA; arbitrary principal-axis frame
    X_pca = X @ V
    r = float(np.linalg.norm(X_pca, axis=1).max())   # partial's own radius, no /1.137
    if r < 1e-6:
        return None
    p_norm = (X_pca / r).astype(np.float32)
    p_fixed = _fix_size(p_norm, PCN_N_INPUT, rng)
    with torch.no_grad():
        inp = torch.from_numpy(p_fixed).float().unsqueeze(0).to(device)
        _, fine = model(inp)
        pred_pca = fine.squeeze(0).cpu().numpy()
    pred_pca = pred_pca * r
    pred_velo = pred_pca @ V.T + c
    return pred_velo.astype(np.float32)


def candidate_bands(records, gt_by_inst):
    """Per density band, a gain-sorted candidate list (the records predate the
    promoted config, so some no longer re-match; the render loop falls through
    candidates until one does)."""
    scored = []
    for r in records:
        if r["skip_reason"] is not None or r["inst_id"] not in gt_by_inst:
            continue
        gt = gt_by_inst[r["inst_id"]]
        gain = pair_metrics(r["comp_box"], gt)["bev_iou"] - pair_metrics(r["raw_box"], gt)["bev_iou"]
        scored.append((r, gain))
    bands = [
        ("sparse (<100 pts)", lambda n: n < 100),
        ("mid (100-300 pts)", lambda n: 100 <= n < 300),
        ("dense (>=300 pts)", lambda n: n >= 300),
    ]
    out = []
    for title, pred in bands:
        cands = sorted((s for s in scored if pred(s[0]["n_raw_pts"])),
                       key=lambda s: -s[1])
        out.append((title, [r for r, _ in cands]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    args = ap.parse_args()
    out_png = os.path.join(PROJECT_ROOT, "output", "experiments", "fig62_deblob",
                           f"box_overlay_deblob_{args.seq}.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    with open(os.path.join(PROJECT_ROOT, "output", "experiments",
                           "completion_box_eval", f"step1_records_{args.seq}.json")) as f:
        records = json.load(f)["records"]
    with open(os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")) as f:
        amodal = json.load(f)
    gt_by_inst = {int(k): r for k, r in amodal["instances"].items()
                  if r.get("well_observed")}

    bands = candidate_bands(records, gt_by_inst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Detection uses the checkpoint the step1_records/amodal-GT were built with
    # (stage_b_best), so the pre-selected cars re-match. The old-vs-corrected
    # completion comparison is classifier-independent -- the classifier only
    # selects which GT-matched cluster gets completed, not its geometry.
    cls_model, bbox_stats = load_classifier(
        os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_best.pth"), device)
    kitti_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")
    completer = PointCloudCompleter(model_path=kitti_ckpt)          # corrected path
    ck = torch.load(kitti_ckpt, map_location=device, weights_only=False)
    raw_model = PCN(num_coarse=1024, grid_size=2).to(device)        # for the old path
    raw_model.load_state_dict(ck["model_state_dict"]); raw_model.eval()

    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "sequences", args.seq)
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne", "*.bin")))
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels", "*.label")))
    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    Tr = load_calib(os.path.join(seq_dir, "calib.txt"))["Tr"]

    used = set()

    def render_panel(ax, rec, title):
        fi, inst_id = rec["frame"], rec["inst_id"]
        if inst_id in used:
            return False
        _, _, _, cluster_labels, objects_pcd, det_masks, gt_masks = get_frame_detections(
            bin_paths[fi], label_paths[fi], cls_model=cls_model, cls_device=device,
            cls_bbox_stats=bbox_stats, thing_classes=THING_CLASSES_SUPPORTED,
            keep_unknown=False)
        det_id = next((d for d, g, _ in match_pairs(det_masks, gt_masks, 0.3)
                       if int(g) == inst_id), None)
        if det_id is None:
            return False

        raw_sensor = np.asarray(objects_pcd.points)[cluster_labels == det_id]
        old_sensor = complete_old_blob(raw_model, device, raw_sensor, np.random.default_rng(0))
        corr_sensor, skip = completer.complete(raw_sensor, "car", sample_seed=0)
        if skip is not None:
            return False
        T = poses[fi] @ Tr
        R, t = T[:3, :3], T[:3, 3]
        raw_w = raw_sensor @ R.T + t
        old_w = old_sensor @ R.T + t
        corr_w = corr_sensor @ R.T + t

        gt = gt_by_inst[inst_id]
        old_box, corr_box = world_box(old_w), world_box(corr_w)

        ax.scatter(raw_w[:, 0], raw_w[:, 2], s=4, c="dimgray", zorder=2, label="raw pts")
        for box, color, lbl in [(gt, "black", "GT (amodal)"),
                                (old_box, "tab:red", "old completion"),
                                (corr_box, "tab:green", "corrected completion")]:
            c = box_corners_xz([box["center_world"][0], box["center_world"][2]],
                               box["yaw_deg"], *box["dims_lwh"][:2])
            c = np.vstack([c, c[:1]])
            ax.plot(c[:, 0], c[:, 1], color=color, lw=2, label=lbl, zorder=3)

        iou_old, iou_corr = bev_iou(old_box, gt), bev_iou(corr_box, gt)
        ax.set_title(f"{title} | inst {inst_id}, {len(raw_sensor)} pts\n"
                     f"BEV IoU vs GT:  old {iou_old:.2f}  ->  corrected {iou_corr:.2f}",
                     fontsize=10)
        ax.set_aspect("equal"); ax.margins(0.35)
        ax.set_xlabel("world X (m)"); ax.set_ylabel("world Z (m)")
        ax.legend(fontsize=8, loc="upper right")
        used.add(inst_id)
        print(f"  {title}: inst {inst_id} frame {fi} ({len(raw_sensor)} pts) "
              f"old IoU {iou_old:.3f} -> corr {iou_corr:.3f}")
        return True

    fig, axes = plt.subplots(1, len(bands), figsize=(5.2 * len(bands), 5.2))
    for ax, (title, cands) in zip(np.ravel(axes), bands):
        for rec in cands:                      # fall through until one re-matches
            if render_panel(ax, rec, title):
                break
        else:
            ax.set_title(f"{title}\nno candidate re-matched under current config")

    fig.suptitle(
        f"Seq {args.seq}: the inference-normalization fix (Finding #26) in BEV -- "
        "old (PCA+partial-radius) vs corrected completion box vs amodal GT",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=130)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
