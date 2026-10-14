"""
Regenerate the completion-fidelity schematic (fig:bg-fidelity) with the observed
input relabelled P -> X, matching the notation used in ch2_background.tex.

Faithful re-creation of the original bg_fidelity.png style:
  (a) Chamfer distance -- completion C (green, as in panel b) and complete
      reference B (orange), bidirectional nearest-neighbour arrows.
  (b) Unidirectional UCD/UHD -- observed input X (navy filled), completion C
      (green open circles) on a full ring; X only on the observed (left+bottom)
      sides; the added surface (top+right) receives no arrow.

Output: docs/writing/thesis/figures/bg_fidelity_x.png  (new file; original kept)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAVY  = "#1f4e79"
ORANGE = "#e08214"
GREEN = "#2ca02c"
BLUE_ARROW = "#3b6ea5"
RED   = "#c0392b"

def nearest(idx_from, pts_from, pts_to):
    d = np.linalg.norm(pts_to[None, :, :] - pts_from[idx_from][None, None, :], axis=2)[0]
    return int(np.argmin(d))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.2, 6.1))

# ----------------------------------------------------------------------------
# Panel (a): Chamfer distance -- bidirectional nearest-neighbour links
# ----------------------------------------------------------------------------
# Completion C (navy) and complete reference B (orange) as two scattered clusters
# -- the original panel-(a) look. Straight directional arrows: each C point -> its
# nearest B (navy), each B point -> its nearest C (orange). Each cluster has one
# point closest to the other cluster, so the nearest-neighbour arrows converge
# there: this is genuine Chamfer behaviour (every point maps to its single nearest
# neighbour), not an artefact.
Cset = np.array([[0.70, 4.00], [2.30, 3.00], [1.50, 2.50], [1.30, 1.70], [1.80, 3.60]])
Bset = np.array([[3.40, 3.60], [4.30, 3.10], [3.00, 3.00], [3.90, 2.00], [3.60, 1.30]])

for i in range(len(Cset)):
    j = nearest(i, Cset, Bset)
    axa.annotate("", xy=Bset[j], xytext=Cset[i],
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2, alpha=0.75))
for j in range(len(Bset)):
    i = nearest(j, Bset, Cset)
    axa.annotate("", xy=Cset[i], xytext=Bset[j],
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2, alpha=0.75))

axa.scatter(Cset[:, 0], Cset[:, 1], s=110, color=NAVY, zorder=3)
axa.scatter(Bset[:, 0], Bset[:, 1], s=110, color=ORANGE, zorder=3)
axa.set_title("(a) Chamfer distance\n(bidirectional; needs a complete reference $B$)",
              fontsize=13)
axa.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=NAVY,
                           markersize=11, label="completion $C$"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE,
                           markersize=11, label="reference $B$")],
           loc="lower center", ncol=2, frameon=True, fontsize=12)
axa.set_xlim(0.3, 4.7)
axa.set_ylim(0.9, 4.4)

# ----------------------------------------------------------------------------
# Panel (b): Unidirectional UCD / UHD -- observed input X only
# ----------------------------------------------------------------------------
S = 5.0
# Completion C: full ring (navy filled dots, matching C in panel (a))
C_bottom = [(x, 0.0) for x in (0, 1.25, 2.5, 3.75, 5.0)]
C_right  = [(S, y) for y in (1.25, 2.5, 3.75)]
C_top    = [(x, S) for x in (0, 1.25, 2.5, 3.75, 5.0)]
C_left   = [(0.0, y) for y in (1.25, 2.5, 3.75)]
C = np.array(C_bottom + C_right + C_top + C_left)

# Observed input X: only on left + bottom (the two observed sides), just inside
X_left   = [(0.65, 1.25), (0.65, 2.5), (1.15, 3.75), (0.65, 5.0)]  # 3rd = longest (UHD)
X_bottom = [(0.65, 0.65), (1.25, 0.65), (2.5, 0.65), (3.75, 0.65), (5.0, 0.65)]
X = np.array(X_left + X_bottom)

# Nearest-C target for each X, and identify the worst-case (longest) arrow
targets, lengths = [], []
for i in range(len(X)):
    j = nearest(i, X, C)
    targets.append(j)
    lengths.append(np.linalg.norm(C[j] - X[i]))
worst = int(np.argmax(lengths))

for i in range(len(X)):
    col = RED if i == worst else BLUE_ARROW
    lw = 2.4 if i == worst else 1.4
    axb.annotate("", xy=C[targets[i]], xytext=X[i],
                 arrowprops=dict(arrowstyle="->", color=col, lw=lw))

# added-surface pointer -> a right-column C point with no X near it
added = np.array([S, 3.75])
axb.annotate("added surface:\nno arrow reaches it\n$\\Rightarrow$ not scored",
             xy=added, xytext=(2.7, 4.0), fontsize=10.5, color="0.30", ha="center",
             arrowprops=dict(arrowstyle="->", color="0.30", lw=1.6))

axb.text(2.4, 2.25, "every observed point\nhas a near $C$ point\n$\\Rightarrow$ UCD, UHD small",
         fontsize=10.5, color=NAVY, ha="center", va="center")
axb.text(2.5, -0.85, "red = worst-case arrow (UHD)", fontsize=10.5, color=RED, ha="center")

axb.scatter(C[:, 0], C[:, 1], s=100, color=NAVY, zorder=3)
axb.scatter(X[:, 0], X[:, 1], s=105, color=ORANGE, zorder=4)

axb.set_title("(b) Unidirectional UCD / UHD\n(scored only against the observed input $X$)",
              fontsize=13)
axb.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE,
                           markersize=11, label="observed input $X$"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=NAVY,
                           markersize=11, label="completion $C$")],
           loc="upper right", frameon=True, fontsize=11)
axb.set_xlim(-0.6, 6.0)
axb.set_ylim(-1.2, 6.2)

for ax in (axa, axb):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

fig.suptitle("Completion fidelity: scored against the observed input, so added surface earns no credit",
             fontsize=15, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Script lives in figures/scripts/ ; the figure it feeds is one level up in figures/.
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "bg_fidelity_x.png"))
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
