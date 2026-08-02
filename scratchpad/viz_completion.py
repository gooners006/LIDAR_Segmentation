"""Visualize PCN completion: partial input vs completed output per car track.

For each chosen completed track we overlay the partial single-frame input
(blue) and the PCN-completed cloud (green) in two projections:
  - BEV (top-down): horizontal plane = X (width) vs Z (length)
  - side elevation: Z (length) vs height (= -Y, since the saved global frame
    is Y-down: world up = -Y)
Each car is centred on its own centroid for readability. Saves a PNG grid.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ply(path):
    if not os.path.exists(path):
        return np.zeros((0, 3))
    return np.asarray(o3d.io.read_point_cloud(path).points)


def render(seq="08", n_tracks=6, out_name="seq08_completion.png"):
    obj_dir = os.path.join(PROJECT_ROOT, f"output/{seq}/objects")
    meta = json.load(open(os.path.join(PROJECT_ROOT, f"output/{seq}/tracks.json")))
    comp = [t for t in meta["tracks"] if t.get("completed")]
    # spread across the input-density range, not just the densest
    comp = sorted(comp, key=lambda t: t["completion_input_points"], reverse=True)
    idxs = np.linspace(0, len(comp) - 1, n_tracks).astype(int)
    tracks = [comp[i] for i in idxs]

    fig, axes = plt.subplots(n_tracks, 2, figsize=(11, 3.1 * n_tracks))
    axes = np.atleast_2d(axes)

    for r, t in enumerate(tracks):
        tid = t["track_id"]
        partial = load_ply(os.path.join(obj_dir, f"{tid}_partial.ply"))
        comp_pts = load_ply(os.path.join(obj_dir, f"{tid}.ply"))
        center = comp_pts.mean(0) if len(comp_pts) else partial.mean(0)
        p = partial - center
        c = comp_pts - center

        ax_bev, ax_side = axes[r, 0], axes[r, 1]
        # BEV: width (X) vs length (Z)
        ax_bev.scatter(c[:, 2], c[:, 0], s=4, c="#00b33c", alpha=0.5,
                       linewidths=0, label="completed")
        ax_bev.scatter(p[:, 2], p[:, 0], s=4, c="#1f6fff", alpha=0.8,
                       linewidths=0, label="partial input")
        ax_bev.set_aspect("equal")
        ax_bev.set_xlabel("length Z (m)")
        ax_bev.set_ylabel("width X (m)")
        ax_bev.set_title(f"track {tid} — BEV (top-down) | input={t['completion_input_points']} pts",
                         fontsize=9)
        ax_bev.grid(alpha=0.2)

        # side: length (Z) vs height (-Y)
        ax_side.scatter(c[:, 2], -c[:, 1], s=4, c="#00b33c", alpha=0.5,
                        linewidths=0)
        ax_side.scatter(p[:, 2], -p[:, 1], s=4, c="#1f6fff", alpha=0.8,
                        linewidths=0)
        ax_side.set_aspect("equal")
        ax_side.set_xlabel("length Z (m)")
        ax_side.set_ylabel("height -Y (m)")
        # axis-agnostic dims: sort the two horizontal extents so L >= W
        ex_x = c[:, 0].max() - c[:, 0].min()
        ex_z = c[:, 2].max() - c[:, 2].min()
        L, W = max(ex_x, ex_z), min(ex_x, ex_z)
        H = c[:, 1].max() - c[:, 1].min()
        ax_side.set_title(f"side elevation | L={L:.1f} W={W:.1f} H={H:.1f} m (sorted)",
                          fontsize=9)
        ax_side.grid(alpha=0.2)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#1f6fff", label="partial input (single frame)"),
        plt.Line2D([], [], marker="o", ls="", color="#00b33c", label="PCN completed (4096 pts)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Seq {seq} PCN completion: partial input vs completed car shape",
                 y=1.015, fontsize=13)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", out_name)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}  ({len(tracks)} tracks)")


if __name__ == "__main__":
    render()
