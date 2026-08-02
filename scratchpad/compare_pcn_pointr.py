"""PCN vs PoinTr completion quality on real seq-08 (the decider).

Both runs used identical pipeline + input gate + heading (lshape), so the
clean-gated completed track set is the same; only the completion model differs:

  output/08_ab_gated  -> PCN  (pcn_kitti_best.pth)   baseline: 18/26 plausible
  output/08_pointr    -> PoinTr (pointr_kitti_best.pth)

Plausible-car box (same as Finding #27):
  L in [3.3, 4.9], W in [1.5, 2.1], H in [1.1, 1.7]

dims() = desc-sorted global-axis extents. The per-track heading (hence global
rotation) is identical across models, so the proxy is applied apples-to-apples.

  .venv/Scripts/python.exe scratchpad/compare_pcn_pointr.py
"""
import json
import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCN_DIR = os.path.join(ROOT, "output", "08_ab_gated")
PTR_DIR = os.path.join(ROOT, "output", "08_pointr")
GROUND = (0, 2)  # X, Z ground plane (global frame, Y = up)
L_BAND, W_BAND, H_BAND = (3.3, 4.9), (1.5, 2.1), (1.1, 1.7)


def load(out_dir, tid, suffix=""):
    p = os.path.join(out_dir, "objects", f"{tid}{suffix}.ply")
    if not os.path.exists(p):
        return None
    return np.asarray(o3d.io.read_point_cloud(p).points)


def dims(pts):
    return np.sort(pts.max(0) - pts.min(0))[::-1]


def plausible(d):
    L, W, H = d
    return (L_BAND[0] <= L <= L_BAND[1] and
            W_BAND[0] <= W <= W_BAND[1] and
            H_BAND[0] <= H <= H_BAND[1])


def completed_set(out_dir):
    meta = json.load(open(os.path.join(out_dir, "tracks.json")))
    return {t["track_id"]: t for t in meta["tracks"] if t.get("completed")}


def main():
    P, Q = completed_set(PCN_DIR), completed_set(PTR_DIR)
    common = sorted(set(P) & set(Q))
    print(f"completed — PCN: {len(P)}  PoinTr: {len(Q)}  common: {len(common)}\n")

    rows = []
    for tid in common:
        dp, dq = dims(load(PCN_DIR, tid)), dims(load(PTR_DIR, tid))
        rows.append((tid, dp, plausible(dp), dq, plausible(dq)))

    np_ok = sum(r[2] for r in rows)
    nq_ok = sum(r[4] for r in rows)
    print(f"PLAUSIBLE-CAR RATE  PCN: {np_ok}/{len(common)}   PoinTr: {nq_ok}/{len(common)}\n")

    print(f"{'tid':>5} {'PCN L/W/H':>20} {'ok':>3} | {'PoinTr L/W/H':>20} {'ok':>3} {'flip':>6}")
    flips = []
    for tid, dp, op, dq, oq in sorted(rows, key=lambda r: (r[2] == r[4], r[0])):
        flip = "" if op == oq else ("+win" if oq and not op else "-loss")
        if op != oq:
            flips.append((tid, flip))
        print(f"{tid:>5} {str(np.round(dp,2)):>20} {'Y' if op else 'n':>3} | "
              f"{str(np.round(dq,2)):>20} {'Y' if oq else 'n':>3} {flip:>6}")

    print(f"\nflips: {flips}")
    print(f"net plausible change: {nq_ok - np_ok:+d}")

    # Render the tracks where the two models DISAGREE (the decisive cases).
    disagree = [r[0] for r in rows if r[2] != r[4]]
    if not disagree:
        # fall back to PCN failures, to see if PoinTr fixed shape even if box flag same
        disagree = [r[0] for r in rows if not r[2]][:8]
    if disagree:
        gi, gj = GROUND
        n = len(disagree)
        fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.6), squeeze=False)
        for col, tid in enumerate(disagree):
            partial = load(PCN_DIR, tid, "_partial")
            if partial is None:
                partial = load(PTR_DIR, tid, "_partial")
            c = partial[:, [gi, gj]].mean(0) if partial is not None else np.zeros(2)
            for row, (name, d) in enumerate([("PCN", PCN_DIR), ("PoinTr", PTR_DIR)]):
                ax = axes[row][col]
                full = load(d, tid)
                ax.scatter(full[:, gi] - c[0], full[:, gj] - c[1], s=0.5,
                           c="#00cc44", alpha=0.35)
                if partial is not None:
                    ax.scatter(partial[:, gi] - c[0], partial[:, gj] - c[1], s=4,
                               c="#2aa6ff", alpha=0.9)
                dd = dims(full)
                ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
                ax.set_xticklabels([]); ax.set_yticklabels([])
                ok = "Y" if plausible(dd) else "n"
                if row == 0:
                    ax.set_title(f"tid {tid}\nL/W/H {np.round(dd,2)} [{ok}]", fontsize=8)
                else:
                    ax.set_title(f"L/W/H {np.round(dd,2)} [{ok}]", fontsize=8)
                if col == 0:
                    ax.set_ylabel(name, fontsize=11)
        fig.suptitle("PCN (top) vs PoinTr (bottom) — top-down BEV (X-Z), disagreement tracks\n"
                     "partial (blue) + completed car (green)", fontsize=12)
        out = os.path.join(ROOT, "output", "compare_pcn_pointr.png")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print("saved", out)


if __name__ == "__main__":
    main()
