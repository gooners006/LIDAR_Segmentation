"""LOSO diagram (slide 3.2.3): 11 folds x 11 sequences, held-out sequence per fold in orange,
training sequences in blue, with each fold's held-out recall from tab:loso (verbatim).

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_loso_grid.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "assets" / "loso_grid.png"
TRAIN, TEST, INK, MUTED = "#DCE8F5", "#D9822B", "#1F2933", "#6B7280"
SEQS = [f"{i:02d}" for i in range(11)]
RECALL = ["0.758", "0.336", "0.701", "0.623", "0.510", "0.725",
          "0.637", "0.726", "0.736", "0.638", "0.557"]          # tab:loso, held-out fold recall

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 15})
fig, ax = plt.subplots(figsize=(11, 8.2))
for r, k in enumerate(SEQS):
    for c, _ in enumerate(SEQS):
        held = r == c
        ax.add_patch(Rectangle((c, -r), 0.92, 0.82, fc=TEST if held else TRAIN,
                               ec="#2B5C8A" if not held else TEST, lw=1))
        if held:
            ax.text(c + 0.46, -r + 0.41, "test", ha="center", va="center", fontsize=12,
                    color="white", fontweight="bold")
    ax.text(-0.25, -r + 0.41, f"fold {k}", ha="right", va="center", fontsize=15, color=INK)
    ax.text(11.5, -r + 0.41, RECALL[r], ha="left", va="center", fontsize=15, color=INK,
            fontweight="bold" if k == "08" else "normal")
for c, s in enumerate(SEQS):
    ax.text(c + 0.46, 1.2, s, ha="center", va="center", fontsize=15, color=INK)
ax.text(5.5, 1.95, "sequence", ha="center", va="center", fontsize=16, color=MUTED)
ax.text(11.5, 1.2, "recall on\nthe test seq", ha="left", va="center", fontsize=14, color=MUTED)
ax.set_xlim(-2.2, 13.6); ax.set_ylim(-10.6, 2.4); ax.set_aspect("equal"); ax.axis("off")
h = [Rectangle((0, 0), 1, 1, fc=TRAIN, ec="#2B5C8A"), Rectangle((0, 0), 1, 1, fc=TEST, ec=TEST)]
ax.legend(h, ["classifier trained on these 10", "never seen in training or checkpoint choice"],
          loc="lower center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False, fontsize=15)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
