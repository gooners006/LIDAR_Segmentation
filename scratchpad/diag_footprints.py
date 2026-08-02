"""Show that completion quality tracks BEV footprint elongation, not point count.

Top row: dense-but-poor tracks (many points, low elongation -> ambiguous heading).
Bottom row: sparse-but-good tracks (few points, elongated -> clear heading).
Each cell: BEV (top-down) of partial (blue) + completed car (green), with the
PCA major axis (red) the completer uses to estimate heading.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "08_demo")
DENSE_POOR = [301, 762, 884]
SPARSE_GOOD = [21, 321, 43]


def loadp(tid, suffix=""):
    return np.asarray(o3d.io.read_point_cloud(
        os.path.join(ROOT, "objects", f"{tid}{suffix}.ply")).points)


def major_axis(pts):
    xy = pts[:, :2] - pts[:, :2].mean(0)
    w, v = np.linalg.eigh(xy.T @ xy)
    e = v[:, np.argmax(w)]
    ratio = np.sqrt(np.sort(w)[::-1][0] / max(np.sort(w)[::-1][1], 1e-9))
    return e, ratio


def main():
    meta = {t["track_id"]: t for t in json.load(open(os.path.join(ROOT, "tracks.json")))["tracks"]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for row, (label, tids) in enumerate(
            [("DENSE but POOR (low elongation)", DENSE_POOR),
             ("SPARSE but GOOD (high elongation)", SPARSE_GOOD)]):
        for col, tid in enumerate(tids):
            ax = axes[row, col]
            partial = loadp(tid, "_partial"); full = loadp(tid)
            c = partial[:, :2].mean(0)
            ax.scatter(full[:, 0] - c[0], full[:, 1] - c[1], s=1, c="#00cc44", alpha=0.4)
            ax.scatter(partial[:, 0] - c[0], partial[:, 1] - c[1], s=4, c="#2aa6ff", alpha=0.9)
            e, ratio = major_axis(partial)
            L = 3.0
            ax.plot([-e[0]*L, e[0]*L], [-e[1]*L, e[1]*L], "r-", lw=2, label="PCA heading")
            ax.set_aspect("equal")
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
            ax.set_title(f"tid {tid}: {meta[tid]['completion_input_points']} pts | "
                         f"elong {ratio:.1f}", fontsize=10)
            ax.set_xticklabels([]); ax.set_yticklabels([])
        axes[row, 0].set_ylabel(label, fontsize=11)
    fig.suptitle("BEV top-down: partial (blue) + completed (green) + PCA heading (red)\n"
                 "Poor = square footprint, heading ambiguous; Good = elongated, heading clear",
                 fontsize=12)
    out = os.path.join(ROOT, "diag_footprints.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
