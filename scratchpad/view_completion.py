"""Visualize full-pipeline completion results (before -> after).

Reads a pipeline output dir (output/<seq><tag>) and, for completed car tracks,
overlays the single-frame partial input (blue) with the PCN-completed car (green)
across multiple views. This is the "completion result" of a full pipeline run.

  python scratchpad/view_completion.py --out 08_demo --n 6           # PNG grid
  python scratchpad/view_completion.py --out 08 --interactive --tid 301   # Open3D
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS = [("top", 90, -90), ("side", 5, 0)]


def load(out_dir, tid, suffix=""):
    path = os.path.join(out_dir, "objects", f"{tid}{suffix}.ply")
    if not os.path.exists(path):
        return None
    return np.asarray(o3d.io.read_point_cloud(path).points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="08_demo", help="output/<this> dir")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--tid", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "output", args.out)
    meta = json.load(open(os.path.join(out_dir, "tracks.json")))
    comp = [t for t in meta["tracks"] if t.get("completed")]
    comp.sort(key=lambda t: t.get("completion_input_points", 0))
    print(f"{args.out}: {len(comp)} completed car tracks of {len(meta['tracks'])}")

    if args.interactive:
        tid = args.tid if args.tid is not None else comp[len(comp) // 2]["track_id"]
        partial = load(out_dir, tid, "_partial")
        full = load(out_dir, tid)
        geos = []
        if partial is not None:
            pp = o3d.geometry.PointCloud(); pp.points = o3d.utility.Vector3dVector(partial)
            pp.paint_uniform_color([0.30, 0.65, 1.0]); geos.append(pp)
        fp = o3d.geometry.PointCloud(); fp.points = o3d.utility.Vector3dVector(full)
        fp.paint_uniform_color([0.0, 0.9, 0.3]); geos.append(fp)
        print(f"track {tid}: partial(blue)={0 if partial is None else len(partial)} "
              f"completed(green)={len(full)}")
        o3d.visualization.draw_geometries(geos, window_name=f"Completion track {tid}")
        return

    sel = [comp[int(i)] for i in np.linspace(0, len(comp) - 1, min(args.n, len(comp)))]
    nrow, ncol = len(VIEWS), len(sel)
    fig = plt.figure(figsize=(3.6 * ncol, 3.6 * nrow))
    fig.suptitle(f"Full-pipeline completion (output/{args.out}) — "
                 f"partial input (blue) vs PCN-completed car (green)", fontsize=12)
    for c, t in enumerate(sel):
        tid = t["track_id"]
        partial = load(out_dir, tid, "_partial")
        full = load(out_dir, tid)
        ext = np.sort(full.max(0) - full.min(0))[::-1]
        mid = full.mean(0); rng = max(ext.max(), 3.0) / 2
        for r, (vname, elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            if partial is not None:
                ax.scatter(partial[:, 0], partial[:, 1], partial[:, 2],
                           s=3.0, c="#2aa6ff", alpha=0.9, depthshade=False)
            ax.scatter(full[:, 0], full[:, 1], full[:, 2],
                       s=0.5, c="#00e64d", alpha=0.45, depthshade=False)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 1))
            for axis, m in zip("xyz", mid):
                getattr(ax, f"set_{axis}lim")(m - rng, m + rng)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if r == 0:
                ax.set_title(f"tid {tid} | in {t['completion_input_points']}p\n"
                             f"L/W/H {np.round(ext, 2)}", fontsize=9)
            if c == 0:
                ax.text2D(-0.08, 0.5, vname, transform=ax.transAxes,
                          rotation=90, va="center")
    png = os.path.join(out_dir, "completion_overview.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(png, dpi=120, bbox_inches="tight")
    print("saved", png)


if __name__ == "__main__":
    main()
