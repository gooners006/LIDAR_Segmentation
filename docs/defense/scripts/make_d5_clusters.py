"""D5: ground removal then HDBSCAN clusters on one real scan, bird's-eye view (defense slide 2.2.2).

Same seq-08 frame as D4, so slides 2.2.1 -> 2.2.2 follow one scan. Runs the pipeline's own
preprocess_frame -> remove_ground -> cluster_objects (src/pipeline.py, PIPELINE_CONFIG) and plots
(left) ground vs object points and (right) the object points coloured by cluster. Counts are an
illustration on one frame, not a thesis result.

Usage: .venv/Scripts/python.exe docs/defense/scripts/make_d5_clusters.py [frame]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import PIPELINE_CONFIG, cluster_objects, preprocess_frame, remove_ground  # noqa: E402

SEQ, FRAME = "08", int(sys.argv[1]) if len(sys.argv) > 1 else 2500
OUT = ROOT / "docs/defense/assets/d5_ground_then_clusters.png"
GROUND, OBJECT, NOISE = "#C8A464", "#1F4E79", "#D5D9DE"
XLIM, YLIM = (-30, 30), (-15, 15)

cfg = PIPELINE_CONFIG
xyz = np.fromfile(ROOT / f"dataset/sequences/{SEQ}/velodyne/{FRAME:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
pcd = preprocess_frame(xyz, cfg)
ground_pcd, objects_pcd, _, _ = remove_ground(pcd, cfg)
G, O = np.asarray(ground_pcd.points), np.asarray(objects_pcd.points)
labels = cluster_objects(objects_pcd, cfg)
assert len(labels) == len(O)
ids = np.unique(labels[labels >= 0])
print(f"frame {FRAME}: {len(G):,} ground, {len(O):,} object points, {len(ids)} clusters, "
      f"{(labels < 0).sum():,} noise points")

# shuffled qualitative palette so neighbouring cluster ids get distinct colours
palette = np.array(plt.get_cmap("tab20").colors + plt.get_cmap("Dark2").colors)
palette = palette[palette.max(1) - palette.min(1) > 0.15]  # drop greys: they read as noise
rng = np.random.default_rng(0)
colour_of = {c: palette[i % len(palette)] for i, c in enumerate(rng.permutation(ids))}

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 15})
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), constrained_layout=True)

ax = axes[0]
ax.scatter(G[:, 0], G[:, 1], s=0.9, c=GROUND, linewidths=0, rasterized=True)
ax.scatter(O[:, 0], O[:, 1], s=0.9, c=OBJECT, linewidths=0, rasterized=True)
ax.set_title("Ground removed (RANSAC)", fontsize=18, fontweight="bold", loc="left")

ax = axes[1]
noise = labels < 0
ax.scatter(O[noise, 0], O[noise, 1], s=0.6, c=NOISE, linewidths=0, rasterized=True)
cols = np.array([colour_of[c] for c in labels[~noise]])
ax.scatter(O[~noise, 0], O[~noise, 1], s=1.2, c=cols, linewidths=0, rasterized=True)
ax.set_title("Object points grouped by HDBSCAN", fontsize=18, fontweight="bold", loc="left")
ax.text(0.01, 0.97, f"{len(ids)} clusters in the full scan, one colour each", transform=ax.transAxes, va="top",
        fontsize=17, fontweight="bold", color="#3E4C59",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))

for ax in axes:
    ax.scatter([0], [0], s=140, marker="^", c="black", zorder=6)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, left)")

h = [plt.Line2D([], [], ls="", marker="s", ms=10, mfc=GROUND, mec=GROUND, label="ground (removed)"),
     plt.Line2D([], [], ls="", marker="s", ms=10, mfc=OBJECT, mec=OBJECT, label="object points"),
     plt.Line2D([], [], ls="", marker="s", ms=10, mfc=NOISE, mec=NOISE, label="noise (no cluster)"),
     plt.Line2D([], [], ls="", marker="^", ms=11, mfc="black", mec="black", label="sensor")]
fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False, fontsize=15)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
