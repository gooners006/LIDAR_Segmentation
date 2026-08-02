"""Direction 1 — BEV overlay figure for the donor-frame metric.

Six X-Z bird's-eye panels spanning the completed-coverage range (2 best /
2 median / 2 worst qualified pairs, distinct cars where possible). Each panel:
donor novel points (black), input partial (red), completed cloud (green),
mirrored partial's added half (orange), amodal GT box (dashed). World frame is
cam0 (ground plane X-Z; #27 methodology note).

Run:
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_viz.py --seq 08
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amodal_gt import voxel_downsample  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gt_box_corners_xz(rec: dict) -> np.ndarray:
    cx, _, cz = rec["center_world"]
    L, W, _ = rec["dims_lwh"]
    yaw = np.deg2rad(rec["yaw_deg"])
    e1 = np.array([np.cos(yaw), np.sin(yaw)])
    e2 = np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.array([cx, cz])
    corners = [c + sl * L / 2 * e1 + sw * W / 2 * e2
               for sl, sw in [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]]
    return np.array(corners)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp_dir = args.dir or os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric")
    out_png = args.out or os.path.join(
        PROJECT_ROOT, "output", "figures", f"donor_metric_{args.seq}.png")

    with open(os.path.join(exp_dir, f"donor_metric_records_{args.seq}.json")) as f:
        payload = json.load(f)
    tp = f"{payload['config']['tau_primary']:.2f}"
    tau = payload["config"]["tau_primary"]
    with open(os.path.join(
            PROJECT_ROOT, "output", args.seq, "amodal_gt.json")) as f:
        amodal = json.load(f)

    qual = sorted((r for r in payload["records"]
                   if r["taus"][tp]["n_novel"] >= 100),
                  key=lambda r: r["taus"][tp]["completed"]["cov"])

    # 2 best / 2 median / 2 worst, preferring distinct cars.
    mid = len(qual) // 2
    candidates = [qual[-1], qual[-2], qual[mid], qual[mid + 1], qual[1], qual[0]]
    picks, seen = [], set()
    for r in candidates:
        if r["inst_id"] in seen and len(qual) > 12:
            alt = next((q for q in qual if q["inst_id"] not in seen), r)
            r = alt
        picks.append(r)
        seen.add(r["inst_id"])

    cache = np.load(os.path.join(exp_dir, f"accum_cache_{args.seq}.npz"))
    per_frame: dict[int, dict[int, np.ndarray]] = {}
    for key in cache.files:
        inst_s, frame_s = key.split("_")
        per_frame.setdefault(int(inst_s), {})[int(frame_s)] = cache[key]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, r in zip(axes.ravel(), picks):
        inst, fi = r["inst_id"], r["frame"]
        data = np.load(os.path.join(exp_dir, f"pairs_{args.seq}",
                                    f"p_f{fi:06d}_i{inst}.npz"))
        raw = data["raw"].astype(np.float64)
        completed = data["completed"].astype(np.float64)
        basis, center, T = data["basis"], data["center"], data["T"]
        R, t = T[:3, :3], T[:3, 3]

        raw_w = raw @ R.T + t
        comp_w = completed @ R.T + t
        pts_c = raw @ basis
        mir_c = pts_c.copy()
        mir_c[:, 0] = 2.0 * center[0] - pts_c[:, 0]
        mir_w = (mir_c @ basis.T) @ R.T + t

        donor = voxel_downsample(np.vstack(
            [a for fj, a in per_frame[inst].items() if fj != fi]
        ).astype(np.float64), 0.03)
        d_in, _ = cKDTree(raw_w).query(donor)
        novel = donor[d_in >= tau]

        ax.scatter(novel[:, 0], novel[:, 2], s=2, c="k", label="novel donor")
        ax.scatter(comp_w[:, 0], comp_w[:, 2], s=2, c="tab:green", alpha=0.5,
                   label="completed")
        ax.scatter(mir_w[:, 0], mir_w[:, 2], s=2, c="tab:orange", alpha=0.4,
                   label="mirrored half")
        ax.scatter(raw_w[:, 0], raw_w[:, 2], s=3, c="tab:red", label="input")
        box = gt_box_corners_xz(amodal["instances"][str(inst)])
        ax.plot(box[:, 0], box[:, 1], "k--", lw=1, label="amodal GT box")
        # Ego is usually outside the zoom window; show the viewing direction.
        c_inst = amodal["instances"][str(inst)]["center_world"]
        v = np.array([t[0] - c_inst[0], t[2] - c_inst[2]])
        v = v / max(np.linalg.norm(v), 1e-9)
        ax.annotate("", xy=(c_inst[0] + 3.5 * v[0], c_inst[2] + 3.5 * v[1]),
                    xytext=(c_inst[0] + 2.3 * v[0], c_inst[2] + 2.3 * v[1]),
                    arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
        ax.text(c_inst[0] + 3.7 * v[0], c_inst[2] + 3.7 * v[1], "ego",
                color="tab:blue", fontsize=9, ha="center")

        cov = r["taus"][tp]["completed"]["cov"]
        ax.set_title(f"inst {inst} f{fi}  n_raw={r['n_raw_pts']}  "
                     f"cov(completed)={cov:.2f}")
        ax.set_aspect("equal")
        pad = 2.5
        cx, _, cz = amodal["instances"][str(inst)]["center_world"]
        ax.set_xlim(cx - pad - 3, cx + pad + 3)
        ax.set_ylim(cz - pad - 3, cz + pad + 3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10,
               markerscale=3, frameon=False)
    fig.suptitle(f"Donor-frame occluded-side metric — seq {args.seq} "
                 f"(BEV X-Z; best->worst completed coverage)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
