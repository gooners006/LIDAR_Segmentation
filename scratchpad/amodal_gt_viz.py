"""Visual check for Step 0 amodal GT boxes: BEV scatter + fitted box + ego path.

Re-accumulates the selected instances and renders one BEV (X-Z) panel each:
accumulated points, fitted oriented box, ego positions at observation frames.

Run:
    .venv\\Scripts\\python.exe scratchpad\\amodal_gt_viz.py --seq 08 \\
        --insts 137 372 --n-typical 4 --n-outliers 8
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from amodal_gt import PROJECT_ROOT, accumulate_instances, voxel_downsample, CONFIG  # noqa: E402


def box_corners_xz(center_xz, yaw_deg, length, width):
    e1 = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))])
    e2 = np.array([-e1[1], e1[0]])
    c = np.asarray(center_xz)
    h_l, h_w = length / 2, width / 2
    return np.array([c + s1 * h_l * e1 + s2 * h_w * e2
                     for s1, s2 in [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="08")
    ap.add_argument("--insts", type=int, nargs="*", default=None,
                    help="Explicit instance IDs; default: auto-pick")
    ap.add_argument("--n-typical", type=int, default=4)
    ap.add_argument("--n-outliers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt_path = os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")
    with open(gt_path) as f:
        gt = json.load(f)
    recs = {int(k): v for k, v in gt["instances"].items() if "dims_lwh" in v}

    if args.insts:
        chosen = args.insts
    else:
        well = [r for r in recs.values() if r["well_observed"]]
        off = np.array([np.abs(np.array(r["dims_lwh"]) - [4.2, 1.8, 1.5]).max()
                        for r in well])
        order = np.argsort(off)
        chosen = ([well[i]["inst_id"] for i in order[:args.n_typical]]
                  + [well[i]["inst_id"] for i in order[::-1][:args.n_outliers]])

    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "sequences", args.seq)
    print(f"Re-accumulating {len(chosen)} instances: {chosen}")
    buffers, obs, _ = accumulate_instances(seq_dir, 10**9)

    n = len(chosen)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, gi in zip(axes, chosen):
        r = recs[gi]
        acc = voxel_downsample(
            np.vstack(buffers[gi]).astype(np.float64), CONFIG["voxel"])
        egos = np.array([e for _, _, e in obs[gi]])
        cx, cz = r["center_world"][0], r["center_world"][2]

        ax.scatter(acc[:, 0], acc[:, 2], s=0.5, c="#4da6ff", alpha=0.5)
        cor = box_corners_xz([cx, cz], r["yaw_deg"], *r["dims_lwh"][:2])
        ax.plot(cor[:, 0], cor[:, 1], "k-", lw=1.5)
        ax.plot(egos[:, 0], egos[:, 1], ".", c="#ff8800", ms=3)
        L, W, H = r["dims_lwh"]
        ax.set_title(f"inst {gi} | L={L} W={W} H={H}\n"
                     f"{r['n_frames']}f  span={r['azimuth_span_deg']}deg  "
                     f"spread={r['center_spread_xz_m']}m  "
                     f"WO={r['well_observed']}", fontsize=9)
        pad = 4.0
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cz - pad, cz + pad)
        ax.set_aspect("equal")

    for ax in axes[n:]:
        ax.axis("off")

    out = args.out or os.path.join(
        PROJECT_ROOT, "output", args.seq, "amodal_gt_check.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
