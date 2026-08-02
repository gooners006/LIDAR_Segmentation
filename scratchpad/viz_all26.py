"""All 26 clean-gated completed tracks: PCN vs PoinTr, top-down BEV (X-Z).

For each common completed track, two adjacent panels (PCN | PoinTr): partial
(blue) + completed car (green). Title shows tid, L/W/H and plausibility flag
(green title = plausible car box, red = fails). 4 tracks per row (8 panels).

  .venv/Scripts/python.exe scratchpad/viz_all26.py
"""
import json
import math
import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCN_DIR = os.path.join(ROOT, "output", "08_ab_gated")
PTR_DIR = os.path.join(ROOT, "output", "08_pointr")
GI, GJ = 0, 2  # X, Z ground plane (global frame, Y = up)
L_BAND, W_BAND, H_BAND = (3.3, 4.9), (1.5, 2.1), (1.1, 1.7)
TRACKS_PER_ROW = 4


def load(out_dir, tid, suffix=""):
    p = os.path.join(out_dir, "objects", f"{tid}{suffix}.ply")
    return np.asarray(o3d.io.read_point_cloud(p).points) if os.path.exists(p) else None


def dims(pts):
    return np.sort(pts.max(0) - pts.min(0))[::-1]


def plausible(d):
    L, W, H = d
    return (L_BAND[0] <= L <= L_BAND[1] and W_BAND[0] <= W <= W_BAND[1]
            and H_BAND[0] <= H <= H_BAND[1])


def completed_set(out_dir):
    meta = json.load(open(os.path.join(out_dir, "tracks.json")))
    return {t["track_id"] for t in meta["tracks"] if t.get("completed")}


def panel(ax, full, partial, model, tid):
    c = partial[:, [GI, GJ]].mean(0) if partial is not None else np.zeros(2)
    ax.scatter(full[:, GI] - c[0], full[:, GJ] - c[1], s=0.5, c="#00cc44", alpha=0.35)
    if partial is not None:
        ax.scatter(partial[:, GI] - c[0], partial[:, GJ] - c[1], s=4, c="#2aa6ff", alpha=0.9)
    d = dims(full)
    ok = plausible(d)
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{model} tid {tid}\n{np.round(d, 2)}", fontsize=7,
                 color=("#108010" if ok else "#c01010"))


def main():
    common = sorted(completed_set(PCN_DIR) & completed_set(PTR_DIR))
    print(f"common completed tracks: {len(common)}")

    nrows = math.ceil(len(common) / TRACKS_PER_ROW)
    ncols = TRACKS_PER_ROW * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.5 * nrows),
                             squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for i, tid in enumerate(common):
        r, col = divmod(i, TRACKS_PER_ROW)
        partial = load(PCN_DIR, tid, "_partial")
        if partial is None:
            partial = load(PTR_DIR, tid, "_partial")
        for k, (model, d) in enumerate([("PCN", PCN_DIR), ("PoinTr", PTR_DIR)]):
            ax = axes[r][col * 2 + k]
            ax.axis("on")
            panel(ax, load(d, tid), partial, model, tid)

    fig.suptitle("All 26 clean-gated completions — PCN vs PoinTr, top-down BEV (X-Z)\n"
                 "partial (blue) + completed car (green); green title = plausible-car box, "
                 "red = fails  [L/W/H]", fontsize=12)
    out = os.path.join(ROOT, "output", "all26_pcn_vs_pointr.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
