"""D9: what each detection stage does to true and false positives (defense slide 3.3.1).

Numbers are copied verbatim from the submitted thesis, ch4 tab:ablation-stage (full seq 08,
frozen configuration); nothing is re-run. Rows: geometric filter only (per frame, no classifier,
no track filter) -> + classifier (per frame) -> + classifier and offline track filter (= headline).

Usage: .venv/Scripts/python.exe docs/defense/scripts/make_d9_funnel.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/defense/assets/d9_stage_funnel.png"

# tab:ablation-stage: (label, precision, recall, F1, TP, FP)
ROWS = [("Geometric filter", 0.199, 0.773, 0.316, 27002, 108831),
        ("+ classifier", 0.846, 0.676, 0.751, 23600, 4303),
        ("+ track filter", 0.911, 0.732, 0.812, 25568, 2513)]
TP_C, FP_C, INK, MUTED = "#2B5C8A", "#D7263D", "#1F2933", "#6B7280"

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 19})
fig, ax = plt.subplots(figsize=(9.8, 7.4))
x = np.arange(len(ROWS))
w = 0.36
tp = [r[4] for r in ROWS]
fp = [r[5] for r in ROWS]
ax.bar(x - w / 2, tp, w, color=TP_C, label="true positives (correct detections)")
ax.bar(x + w / 2, fp, w, color=FP_C, label="false positives (false detections)")
for xi, v in zip(x - w / 2, tp):
    ax.text(xi, v + 1500, f"{v:,}", ha="center", va="bottom", fontsize=18, color=TP_C, fontweight="bold")
for xi, v in zip(x + w / 2, fp):
    ax.text(xi, v + 1500, f"{v:,}", ha="center", va="bottom", fontsize=18, color=FP_C, fontweight="bold")

# precision / recall under each stage label
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in ROWS], fontsize=20)
ax.get_xticklabels()[-1].set_fontweight("bold")
for i, r in enumerate(ROWS):
    ax.text(x[i], -0.13, f"P {r[1]:.3f} · R {r[2]:.3f}\nF1 {r[3]:.3f}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=18, color=INK, fontweight="bold" if i == len(ROWS) - 1 else "normal")

# the classifier step: ~104,000 false positives removed (thesis wording)
ax.annotate("", xy=(1 + w / 2, 12000), xytext=(0 + w / 2 + 0.2, 95000),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, connectionstyle="arc3,rad=-0.25",
                            mutation_scale=22))
ax.text(1.17, 62000, "classifier removes\n≈104,000 false positives", ha="left", va="center",
        fontsize=19, color=INK, fontweight="bold")

# the track step: true positives recovered (thesis: TP 23,600 -> 25,568)
ax.text(2 - w / 2, 36000, f"track filter wins back\n{ROWS[2][4] - ROWS[1][4]:,} true positives", ha="center",
        va="bottom", fontsize=17, color=TP_C, fontweight="bold")

ax.set_ylim(0, 122000)
ax.set_xlim(-0.6, 2.6)
ax.set_ylabel("detections on seq 08 (full sequence)", color=MUTED)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v else "0"))
ax.tick_params(axis="y", colors=MUTED, labelsize=16)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#9AA0A6"); ax.spines["bottom"].set_color("#9AA0A6")
ax.legend(loc="upper right", frameon=False, fontsize=17)
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
