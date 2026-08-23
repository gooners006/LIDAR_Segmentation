"""Thesis Fig 6.1 -- training-input distribution fix (Finding #26).

Renders the SAME ShapeNet car through:
  GT          : dense uniform surface sample of the complete mesh (the target)
  naive       : plain pinhole depth back-projection (_render_partial) --
                dense, uniform, groundless -- the wrong input distribution
  KITTI-like  : _render_kitti_like -- single HDL-64E viewpoint, scan-ring banded,
                voxelised at 0.05 m, ground-cut -- matches the real cluster

Illustrates why the retrained model (on KITTI-like partials) stops blobbing:
it now trains on the distribution it meets at inference.

Output: output/experiments/fig61_partials/*.png  (freeze-safe: new folder).
Run:    .venv\\Scripts\\python.exe scratchpad/fig61_partial_compare.py --n-cars 4
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pcn import TRAIN_CONFIG, ShapeNetCompletionDataset  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAR_SYNSET = "02958343"
VIEWS = [("iso", 22, -60), ("side", 5, 0), ("top", 89, -90)]


def render_car(gt, naive, kitti, suptitle, out_path):
    cols = [("Complete (GT target)", gt, "#bbbbbb"),
            ("Naive pinhole partial", naive, "#ff9d4d"),
            ("KITTI-like partial", kitti, "#4da6ff")]
    nrow, ncol = len(VIEWS), len(cols)
    fig = plt.figure(figsize=(3.6 * ncol, 3.6 * nrow))
    fig.suptitle(suptitle, fontsize=12)
    center = gt.mean(0)
    rng = max(np.ptp(gt, axis=0).max(), 3.0) / 2
    for r, (vname, elev, azim) in enumerate(VIEWS):
        for c, (cname, pts, color) in enumerate(cols):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            show = pts
            if len(show) > 4000:
                show = show[np.random.choice(len(show), 4000, replace=False)]
            ax.scatter(show[:, 0], show[:, 1], show[:, 2], s=0.7, c=color, alpha=0.75)
            ax.view_init(elev=elev, azim=azim)
            for axis, mid in zip("xyz", center):
                getattr(ax, f"set_{axis}lim")(mid - rng, mid + rng)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if r == 0:
                ax.set_title(f"{cname}\n({len(pts)} pts)", fontsize=9)
            if c == 0:
                ax.text2D(-0.08, 0.5, vname, transform=ax.transAxes,
                          fontsize=10, rotation=90, va="center")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cars", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = os.path.join(PROJECT_ROOT, "output", "experiments", "fig61_partials")
    os.makedirs(out_dir, exist_ok=True)

    cfg = TRAIN_CONFIG.copy()
    cfg["kitti_like_partial"] = True   # enables the kitti-like config keys
    ds = ShapeNetCompletionDataset(cfg, split="val")
    car_items = [i for i, (_, syn) in enumerate(ds.items) if syn == CAR_SYNSET]
    print(f"Val car models: {len(car_items)}")
    if not car_items:
        print("No car meshes found under shapenet_root; check dataset path.")
        return

    scale_m = cfg["category_scale_m"][CAR_SYNSET]
    made = 0
    for i in np.random.permutation(car_items):
        if made >= args.n_cars:
            break
        obj_path, _ = ds.items[i]
        mesh = ds._load_and_scale(obj_path, scale_m)
        if mesh is None or mesh.get_surface_area() < 1e-6:
            continue
        gt = ds._sample_gt(mesh)
        naive = ds._render_partial(mesh, scale_m)
        kitti = ds._render_kitti_like(mesh, scale_m)
        if naive is None or kitti is None or len(naive) < 20 or len(kitti) < 20:
            print(f"  skip {os.path.basename(os.path.dirname(os.path.dirname(obj_path)))}: "
                  f"naive={None if naive is None else len(naive)} "
                  f"kitti={None if kitti is None else len(kitti)}")
            continue
        model_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
        sup = (f"ShapeNet car {model_id} | naive dense/uniform/groundless vs "
               f"KITTI-like single-view/voxelised/ground-cut")
        out_path = os.path.join(out_dir, f"partials_{model_id}.png")
        render_car(gt, naive, kitti, sup, out_path)
        made += 1
        print(f"  saved {out_path}  (naive={len(naive)}, kitti={len(kitti)} pts)")

    print(f"\nRenders: {out_dir}  ({made} cars)")


if __name__ == "__main__":
    main()
