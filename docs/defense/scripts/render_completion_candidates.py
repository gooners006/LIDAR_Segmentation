"""Contact sheet of the pre-registered completion candidates, each rotated so its length runs
left-right (display only: saved clouds keep their scene heading, see check_completion_heading.py).

Reads docs/defense/figures/completion_candidates.json (pick_completion_examples.py), writes
docs/defense/figures/completion_candidates_aligned.png.
Run:  .venv\\Scripts\\python.exe docs/defense/scripts/render_completion_candidates.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "output" / "experiments" / "plausibility_gen"
FIG = REPO / "docs" / "defense" / "figures"
BLUE, GREEN = "#1F6FEB", "#12A150"
plt.rcParams.update({"font.family": "Segoe UI"})


def load(seq, tid, suffix):
    return np.asarray(o3d.io.read_point_cloud(str(GEN / seq / "objects" / f"{tid}{suffix}.ply")).points)


def align(part, comp):
    """Rotate both clouds about the up axis (Y) so the completed cloud's main X-Z axis is +Z,
    then centre on the completed cloud. Returns (length, width, up) columns for plotting."""
    c = comp[:, [0, 2]].mean(0)
    w, v = np.linalg.eigh(np.cov((comp[:, [0, 2]] - c).T))
    major = v[:, 1]                                    # (x, z)
    t = np.arctan2(major[0], major[1])                 # angle from +Z towards +X
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    out = []
    for p in (part, comp):
        xz = (p[:, [0, 2]] - c) @ R.T                  # rotate by -t: major -> +Z
        out.append(np.column_stack([xz[:, 1], xz[:, 0], -p[:, 1]]))   # length, width, up
    return out


def main():
    cands = json.load(open(FIG / "completion_candidates.json"))["candidates"]
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(6, 4, hspace=0.5, wspace=0.12, height_ratios=[1.1, 1] * 3)
    for k, c in enumerate(cands):
        part, comp = align(load(c["seq"], c["track"], "_partial"), load(c["seq"], c["track"], ""))
        r, col = 2 * (k // 4), k % 4
        for j, (a, b, name) in enumerate([(0, 1, "top"), (0, 2, "side")]):
            ax = fig.add_subplot(gs[r + j, col])
            ax.scatter(comp[:, a], comp[:, b], s=1, c=GREEN, alpha=0.45, lw=0)
            ax.scatter(part[:, a], part[:, b], s=2, c=BLUE, lw=0)
            ax.set_aspect("equal"); ax.set_xlim(-2.8, 2.8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#D1D5DB")
            ax.text(0.01, 0.03, name, transform=ax.transAxes, fontsize=9, color="#6B7280")
            if j == 0:
                ax.set_title(f"{c['label']}   seq {c['seq']} · {c['raw_points']} pts · "
                             f"{c['raw_carprob']:.2f} → {c['completed_carprob']:.2f}",
                             fontsize=12, loc="left")
    fig.suptitle("Pre-registered candidates (judge flips, typical density, seed 0), length left-right. "
                 "Blue = single-frame scan, green = completed. Title: input points, judge car-prob raw → completed.",
                 fontsize=13)
    fig.savefig(FIG / "completion_candidates_aligned.png", dpi=100, bbox_inches="tight")
    print(f"wrote {FIG / 'completion_candidates_aligned.png'}")


if __name__ == "__main__":
    main()
