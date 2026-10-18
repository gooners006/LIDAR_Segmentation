"""Backup slide (UCD / UHD): toy top view of why input-referenced fidelity rewards doing nothing.

Toy car footprint 4.0 x 1.8 m; the scan X sees the near side and one end (an L). Each scan point
is joined to its nearest point of the output; UCD = the average of those distances, UHD = the
largest (ch2 sec:th-distances; the reported tab:fidelity values are plain, not squared, distances,
scratchpad/ucd_uhd_compare.py ucd_mean).
  (a) output = a copy of the scan: every distance 0 -> UCD = UHD = 0, the best possible score;
  (b) output = a whole-car outline that fills the hidden side, its points 5-15 cm off the scan
      points: UCD, UHD > 0, a worse score, and the hidden side receives no line at all.
[illustration], numbers are the toy's own, not results.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_b59_ucd_uhd.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parents[1] / "assets" / "b59_ucd_uhd.png"
SCAN, ADDED, INK, MUTED, WORST = "#2B5C8A", "#D9822B", "#1F2933", "#6B7280", "#B42318"
L, W = 4.0, 1.8


def outline(step):
    """Points every `step` m around the footprint rectangle (centre at 0)."""
    xs = np.arange(-L / 2, L / 2 + 1e-9, step)
    ys = np.arange(-W / 2, W / 2 + 1e-9, step)
    return np.unique(np.round(np.vstack([
        np.column_stack([xs, np.full_like(xs, -W / 2)]), np.column_stack([xs, np.full_like(xs, W / 2)]),
        np.column_stack([np.full_like(ys, -L / 2), ys]), np.column_stack([np.full_like(ys, L / 2), ys])]), 6),
        axis=0)


full = outline(0.4)
scan = full[(np.isclose(full[:, 1], -W / 2)) | (np.isclose(full[:, 0], L / 2) & (full[:, 1] < 0.5))]
rng = np.random.default_rng(0)
comp = outline(0.3) + rng.normal(0, 0.07, outline(0.3).shape)       # whole car, a little off the scan

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 16})
fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
for ax, (C, title) in zip(axs, [(scan.copy(), "(a) Output = a copy of the scan"),
                                (comp, "(b) Output = the whole car")]):
    d, j = cKDTree(C).query(scan)
    ax.scatter(C[:, 0], C[:, 1], s=70 if C is not comp else 34, c=ADDED, zorder=2,
               label="output")
    ax.scatter(scan[:, 0], scan[:, 1], s=26, c=SCAN, zorder=3, label="scan (what the sensor saw)")
    for k in range(len(scan)):
        if d[k] > 1e-9:
            worst = k == d.argmax()
            ax.annotate("", C[j[k]], scan[k], arrowprops=dict(arrowstyle="-", lw=3 if worst else 1.6,
                                                               color=WORST if worst else INK))
    ax.set_title(title, fontsize=20, fontweight="bold", loc="left", color=INK)
    ax.text(0.0, -0.08, f"UCD (average line) {d.mean():.2f} m   ·   UHD (longest line) {d.max():.2f} m",
            transform=ax.transAxes, fontsize=17, color=INK, va="top")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.35, 1.35)
axs[0].text(0, 0.05, "every line has length 0:\nthe best possible score", ha="center", va="center",
            fontsize=16, color=MUTED)
axs[1].text(0, 0.35, "hidden side: no line reaches it,\nso it earns no credit", ha="center",
            va="center", fontsize=16, color=MUTED)
axs[0].text(-2.4, -1.3, "sensor side", fontsize=14, color=MUTED)
h, lab = axs[1].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=16)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
