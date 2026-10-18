"""D14 (slide 2.3.2): what PCN trains on vs what it meets on the road. Illustration, not a result.

Panels (top view + side view each):
  1. simulated scan of a ShapeNet car = the training input, made by the training code's own
     KITTI-like generator (src/train_pcn.py _render_kitti_like: one HDL-64E viewpoint, 0.05 m
     voxels, 0.30 m ground cut);
  2. the complete model sample = the training answer (_sample_gt);
  3. a real single-frame scan = car C of D13 (seq 03, track 192).
Mesh = the thesis fig:kitti-like-partial car. Viewpoint: the first seed (from 0) whose scan shows
a corner (>= 80% of the length and >= 50% of the width visible), so it can be compared with the
real corner view; the generator's own random range (8-30 m) and azimuth are kept.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d14_partials.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pcn import TRAIN_CONFIG, ShapeNetCompletionDataset  # noqa: E402
from render_completion_candidates import align, load  # noqa: E402

MESH = REPO / "dataset/shapenet_data/02958343/2668f2d9b8a2daad95551e0d9e50cb0d/models/model_normalized.obj"
OUT = REPO / "docs/defense/assets/d14_training_vs_real.png"
SCAN, MODEL, INK, MUTED = "#2B5C8A", "#9AA0A6", "#1F2933", "#6B7280"

cfg = dict(TRAIN_CONFIG)
ds = ShapeNetCompletionDataset.__new__(ShapeNetCompletionDataset)   # helpers only, no file scan
ds.config = cfg
mesh = ds._load_and_scale(str(MESH), cfg["category_scale_m"]["02958343"])
np.random.seed(0)                      # reproducible model sample
gt = ds._sample_gt(mesh)


def to_view(p):
    """ShapeNet frame (X width, Y up, Z length) -> (length, width, up), ground at 0."""
    return np.column_stack([p[:, 2], p[:, 0], p[:, 1] - gt[:, 1].min()])


G = to_view(gt)
L, W = np.ptp(G[:, 0]), np.ptp(G[:, 1])
for seed in range(200):
    np.random.seed(seed)
    P = ds._render_kitti_like(mesh, cfg["category_scale_m"]["02958343"])
    if P is None:
        continue
    P = to_view(P)
    if np.ptp(P[:, 0]) >= 0.8 * L and np.ptp(P[:, 1]) >= 0.5 * W:
        break
print(f"seed {seed}: {len(P)} simulated points")

# Put the seen side on top (+width) and the seen end on the right (+length), as in D13.
c = G.mean(0)
for ax in (0, 1):
    seen = P[:, ax].mean() - c[ax]
    if seen < 0:
        P[:, ax] = 2 * c[ax] - P[:, ax]
        G[:, ax] = 2 * c[ax] - G[:, ax]
P[:, :2] -= c[:2]
G[:, :2] -= c[:2]

real, comp = align(load("03", 192, "_partial"), load("03", 192, ""))
real[:, 2] -= comp[:, 2].min()

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 16})
fig, axes = plt.subplots(2, 3, figsize=(18, 6.6), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.0, 0.85]})
panels = [(P, SCAN, 4), (G, MODEL, 3), (real, SCAN, 4)]
for col, (pts, colr, s) in enumerate(panels):
    for row, (a, b) in enumerate([(0, 1), (0, 2)]):
        ax = axes[row, col]
        ax.scatter(pts[:, a], pts[:, b], s=s, c=colr, lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-1.15, 1.15) if row == 0 else ax.set_ylim(-0.15, 1.75)
        ax.axis("off")
        if row == 1:
            ax.axhline(-0.03, c="#D1D5DB", lw=1.5, zorder=0)
for row, name in enumerate(["Top view", "Side view"]):
    axes[row, 0].text(-0.03, 0.5, name, transform=axes[row, 0].transAxes, rotation=90,
                      ha="right", va="center", fontsize=18, fontweight="bold", color=INK)
titles = [("Simulated scan", "training input", SCAN), ("Full 3D model", "training answer", MUTED),
          ("Real scan", "what it meets on the road", SCAN)]
for col, (t, sub, colr) in enumerate(titles):
    axes[0, col].set_title(f"{t}\n", fontsize=20, fontweight="bold", loc="left", color=colr)
    axes[0, col].text(0, 1.02, sub, transform=axes[0, col].transAxes, fontsize=16, color=MUTED)
ax = axes[0, 2]
ax.plot([1.4, 2.4], [-1.05, -1.05], c=INK, lw=2.5)
ax.text(1.9, -0.99, "1 m", ha="center", va="bottom", fontsize=14, color=INK)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
