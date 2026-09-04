"""Conceptual schematic of the donor-frame occluded-side principle.

A single bird's-eye (top-down) diagram that defines the metric's core object --
the NOVEL SET -- for a reader who has not yet seen the real-data figure. It is a
SCHEMATIC (clean shapes, illustrative points), not measured data: a parked car
seen from a road, the near flank the input frame observes, the two ends the donor
frames add (= the novel set), and the occluded far flank no frame sees. The real
per-car coverage on measured seq-08 data is the separate figure donor_metric_08.

Design mirrors the metric's semantics: input (blue), novel set (amber), occluded
(grey). Renders output/figures/donor_principle.png.

Run: .venv\\Scripts\\python.exe scratchpad/fig_donor_principle.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "figures", "donor_principle.png")

C_INPUT = "#2f7fd1"    # near flank the input frame sees
C_NOVEL = "#cf8a12"    # ends the donor frames add = novel set
C_OCC = "#8a97a3"      # occluded far flank
C_INK = "#16202b"
rng = np.random.default_rng(3)

# Car footprint (length along x, width along z), centred at origin.
LEN, WID = 4.2, 1.9
xL, xR, zN, zF = -LEN / 2, LEN / 2, -WID / 2, WID / 2   # left/right/near/far


def jitter(pts, s=0.045):
    return pts + rng.normal(0, s, size=np.shape(pts))


fig, ax = plt.subplots(figsize=(9.2, 6.4))

# --- car body ---
ax.add_patch(FancyBboxPatch(
    (xL, zN), LEN, WID, boxstyle="round,pad=0,rounding_size=0.28",
    linewidth=1.1, edgecolor="#c3ccd4", facecolor="#f3f5f7", zorder=1))

# --- occluded far flank (grey dashed) ---
ax.plot([xL + 0.15, xR - 0.15], [zF, zF], color=C_OCC, lw=4,
        dash_capstyle="round", ls=(0, (1.4, 1.6)), zorder=3)

# --- near flank = input (blue) + input points ---
ax.plot([xL + 0.15, xR - 0.15], [zN, zN], color=C_INPUT, lw=4,
        solid_capstyle="round", zorder=3)
xin = np.linspace(xL + 0.35, xR - 0.35, 22)
ax.scatter(jitter(xin, 0.05), jitter(np.full_like(xin, zN + 0.05)),
           s=11, color=C_INPUT, zorder=5)

# --- ends = novel set (amber) + novel points ---
for xe in (xL, xR):
    ax.plot([xe, xe], [zN + 0.15, zF - 0.15], color=C_NOVEL, lw=4,
            solid_capstyle="round", zorder=3)
    ze = np.linspace(zN + 0.30, zF - 0.30, 7)
    ax.scatter(jitter(np.full_like(ze, xe + (0.05 if xe < 0 else -0.05)), 0.04),
               jitter(ze, 0.045), s=11, color=C_NOVEL, zorder=5)

# --- road + ego frames ---
zr = -2.75
ax.plot([-3.7, 3.7], [zr, zr], color="#c3ccd4", lw=1, ls=(0, (1, 3)), zorder=1)
for xe in (-2.9, -1.5, 1.5, 2.9):
    ax.scatter([xe], [zr], s=60, facecolor="white", edgecolor=C_OCC,
               linewidth=1.6, zorder=6)
ax.scatter([0], [zr], s=95, color=C_INPUT, zorder=7)

# input sightlines to the near flank
for xt in (-1.15, 1.15):
    ax.plot([0, xt], [zr, zN - 0.02], color=C_INPUT, lw=0.9, alpha=0.5, zorder=2)
# donor sightlines catching the ends
ax.plot([-2.9, xL - 0.02], [zr, zN + 0.45], color=C_NOVEL, lw=0.9, alpha=0.7, zorder=2)
ax.plot([2.9, xR + 0.02], [zr, zN + 0.45], color=C_NOVEL, lw=0.9, alpha=0.7, zorder=2)

# --- labels ---
ax.text(0, zr - 0.33, "input frame", color=C_INPUT, ha="center", va="top",
        fontsize=11, fontweight="bold")
ax.text(-2.9, zr - 0.33, "donor\nframe", color="#6b7883", ha="center", va="top", fontsize=9)
ax.text(2.9, zr - 0.33, "donor\nframe", color="#6b7883", ha="center", va="top", fontsize=9)
ax.text(0, zN - 0.30, "near flank — input frame sees", color=C_INPUT,
        ha="center", va="top", fontsize=11)
ax.text(0, zF + 0.28, "far flank — occluded, no frame sees", color="#6b7883",
        ha="center", va="bottom", fontsize=11)
ax.annotate("ends: donor frames add\nthese = NOVEL SET",
            xy=(xR + 0.05, 0.15), xytext=(xR + 1.0, 0.9),
            color=C_NOVEL, fontsize=10.5, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-|>", color=C_NOVEL, lw=1.6))
ax.annotate("", xy=(xL - 0.05, 0.15), xytext=(xL - 0.95, 0.85),
            arrowprops=dict(arrowstyle="-|>", color=C_NOVEL, lw=1.6))

ax.text(0, zF + 1.15,
        r"novel set = donor points $\geq\,\tau$ (0.15 m) from every input point;"
        "\nthe raw partial covers none of it by construction",
        ha="center", va="bottom", fontsize=9.5, color=C_INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbf3e2", ec="#e6d3a8"))

# legend proxies
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color=C_INPUT, lw=4, label="input (near flank the input frame saw)"),
       Line2D([0], [0], color=C_NOVEL, lw=4, label="novel set (occluded surface donor frames saw)"),
       Line2D([0], [0], color=C_OCC, lw=4, ls=(0, (1.4, 1.6)), label="occluded (no frame saw)")]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, frameon=False, fontsize=9.5)

ax.set_title("The donor principle: the novel set is occluded surface only other frames observed",
             fontsize=12.5, color=C_INK, pad=12)
ax.set_xlim(-4.2, 4.2)
ax.set_ylim(zr - 0.9, zF + 2.2)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
