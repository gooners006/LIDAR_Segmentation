"""D13: one real car before and after completion (slide 3.4.1).

Car = candidate C (seq 03, track 192) from the pre-registered draw in pick_completion_examples.py
(judge flip, typical density, seed 0). Rotated so the length runs left-right (display only, same
transform as render_completion_candidates.py). Colours follow the deck: measured = blue,
learned = orange.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d13_completion_example.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_completion_candidates import align, load  # noqa: E402

SEQ, TRACK = "03", 192
OUT = Path(__file__).resolve().parents[1] / "assets" / "d13_completion_example.png"
SCAN, COMP = "#2B5C8A", "#D9822B"
INK, MUTED = "#1F2933", "#6B7280"

part, comp = align(load(SEQ, TRACK, "_partial"), load(SEQ, TRACK, ""))
base = comp[:, 2].min()          # heights are scene heights; put the car's lowest point at 0
part[:, 2] -= base
comp[:, 2] -= base

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 16})
fig, axes = plt.subplots(2, 2, figsize=(16, 7.0), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.0, 0.85]})
views = [(0, 1, "Top view"), (0, 2, "Side view")]
for r, (a, b, view) in enumerate(views):
    for c in range(2):
        ax = axes[r, c]
        if c == 1:
            ax.scatter(comp[:, a], comp[:, b], s=3, c=COMP, alpha=0.55, lw=0, rasterized=True)
        ax.scatter(part[:, a], part[:, b], s=4, c=SCAN, lw=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-1.15, 1.15) if r == 0 else ax.set_ylim(-0.15, 1.75)
        ax.axis("off")
        if c == 0:
            ax.text(-0.02, 0.5, view, transform=ax.transAxes, rotation=90, ha="right", va="center",
                    fontsize=18, fontweight="bold", color=INK)
    if r == 0:  # one 1 m scale bar: both rows share the same scale
        ax = axes[r, 1]
        y0 = ax.get_ylim()[0] + 0.05
        ax.plot([1.4, 2.4], [y0, y0], c=INK, lw=2.5)
        ax.text(1.9, y0 + 0.06, "1 m", ha="center", va="bottom", fontsize=14, color=INK)
# side view: ground = lowest completed point
for c in range(2):
    axes[1, c].axhline(comp[:, 2].min() - 0.03, c="#D1D5DB", lw=1.5, zorder=0)

axes[0, 0].set_title("What the sensor saw (one scan)", fontsize=20, fontweight="bold", loc="left", color=SCAN)
axes[0, 1].set_title("After completion", fontsize=20, fontweight="bold", loc="left", color=COMP)
h = [plt.Line2D([], [], ls="", marker="o", ms=10, mfc=SCAN, mec=SCAN, label="scanned points"),
     plt.Line2D([], [], ls="", marker="o", ms=10, mfc=COMP, mec=COMP, label="points added by the network")]
fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False, fontsize=16)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
