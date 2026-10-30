"""Thesis figure: training input distribution for one ShapeNet car.

Renders the complete target, the naive pinhole partial, and the KITTI-like partial
of the same car, drawn upright (ShapeNet is Y-up). The two partial generators are
the training code's own. Output: figures/training_partials_final.png
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from train_pcn import TRAIN_CONFIG, ShapeNetCompletionDataset  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "training_partials_final.png")
CAR_SYNSET = "02958343"
MODEL_ID = "2668f2d9b8a2daad95551e0d9e50cb0d"
VIEWS = [("three-quarter", 22, -60), ("side", 4, -90), ("top", 89, -90)]


def main():
    np.random.seed(1)
    cfg = TRAIN_CONFIG.copy()
    cfg["kitti_like_partial"] = True
    ds = ShapeNetCompletionDataset(cfg, split="val")
    scale_m = cfg["category_scale_m"][CAR_SYNSET]
    obj_path = next(p for p, _ in ds.items if MODEL_ID in p)
    mesh = ds._load_and_scale(obj_path, scale_m)
    gt = ds._sample_gt(mesh)
    naive = ds._render_partial(mesh, scale_m)
    kitti = ds._render_kitti_like(mesh, scale_m)

    cols = [("Complete target", gt, "#9a9a9a"),
            ("Naive pinhole partial", naive, "#ff8f33"),
            ("KITTI-like partial", kitti, "#2f8fff")]
    centre = gt.mean(0)
    half = np.ptp(gt, axis=0).max() / 2 * 1.05
    fig = plt.figure(figsize=(10.5, 7.4))
    for r, (vname, elev, azim) in enumerate(VIEWS):
        for c, (cname, pts, color) in enumerate(cols):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")
            show = pts[np.random.choice(len(pts), min(len(pts), 4000), replace=False)] - centre
            # plot axes: length (Z), width (X), up (Y)
            ax.scatter(show[:, 2], show[:, 0], show[:, 1], s=1.2, c=color, alpha=0.8, linewidths=0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"{cname} ({len(pts)} points)", fontsize=11)
            if c == 0:
                ax.text2D(-0.02, 0.5, vname, transform=ax.transAxes, fontsize=10, rotation=90, va="center")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.0, wspace=0.0, hspace=0.0)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
