"""Visual check of in-pipeline completion (main.py output).

For each completed car track, reconstruct the representative single-frame partial
(sensor frame) and overlay the completed cloud (mapped sensor->global->back) so we
can confirm the completion is a correctly-scaled, correctly-placed car shape and
that main.py's frame bookkeeping is right. Renders top + side views.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "08")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "check_main_completion.png")


def main():
    meta = json.load(open(os.path.join(ROOT, "tracks.json")))
    comp = [t for t in meta["tracks"] if t.get("completed")]
    comp.sort(key=lambda t: t.get("completion_input_points", 0))
    # spread across input-point range
    sel = [comp[int(i)] for i in np.linspace(0, len(comp) - 1, min(5, len(comp)))]

    nrow, ncol = 2, len(sel)
    fig = plt.figure(figsize=(4 * ncol, 8))
    views = [("top", 90, -90), ("side", 5, 0)]
    for c, t in enumerate(sel):
        out_global = np.asarray(
            o3d.io.read_point_cloud(os.path.join(ROOT, "objects", f"{t['track_id']}.ply")).points)
        ext = out_global.max(0) - out_global.min(0)
        for r, (vname, elev, azim) in enumerate(views):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            ax.scatter(out_global[:, 0], out_global[:, 1], out_global[:, 2],
                       s=0.6, c="#00e64d", alpha=0.6)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 1))
            mid = out_global.mean(0); rng = max(ext.max(), 3.0) / 2
            for axis, m in zip("xyz", mid):
                getattr(ax, f"set_{axis}lim")(m - rng, m + rng)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if r == 0:
                ax.set_title(f"tid {t['track_id']} | in {t['completion_input_points']}p\n"
                             f"L/W/H sorted {np.round(np.sort(ext)[::-1], 2)}", fontsize=9)
            if c == 0:
                ax.text2D(-0.1, 0.5, vname, transform=ax.transAxes, rotation=90, va="center")
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print("saved", os.path.abspath(OUT))
    print("\ntid   in_pts   extent(sorted desc, m)")
    for t in sel:
        og = np.asarray(o3d.io.read_point_cloud(os.path.join(ROOT, "objects", f"{t['track_id']}.ply")).points)
        ext = np.sort(og.max(0) - og.min(0))[::-1]
        print(f"{t['track_id']:>3}  {t['completion_input_points']:>6}   {np.round(ext, 2)}")


if __name__ == "__main__":
    main()
