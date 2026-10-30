"""Thesis figure: real seq-08 completions across the input density range.

Saved clouds keep their scene heading, so each car is rotated about the vertical
axis for display only, until its longest horizontal axis lies along the plot's
length axis. Output: figures/completion_shape_08_final.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
OUT = os.path.join(os.path.dirname(__file__), "..", "completion_shape_08_final.png")
TRACKS = [10463, 19292, 25533, 8387]  # the four tracks of the original figure


def load_ply(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


def display_rotation(xz):
    """Rotation that puts the principal horizontal axis of xz on the first coordinate."""
    _, vecs = np.linalg.eigh(np.cov((xz - xz.mean(0)).T))
    main = vecs[:, 1]
    theta = np.arctan2(main[1], main[0])
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def main():
    obj_dir = os.path.join(PROJECT_ROOT, "output", "08", "objects")
    tracks = json.load(open(os.path.join(PROJECT_ROOT, "output", "08", "tracks.json")))
    tracks = tracks if isinstance(tracks, list) else tracks.get("tracks", tracks)
    tracks = list(tracks.values()) if isinstance(tracks, dict) else tracks
    meta = {t["track_id"]: t for t in tracks}

    fig, axes = plt.subplots(len(TRACKS), 2, figsize=(11, 2.5 * len(TRACKS)),
                             gridspec_kw={"width_ratios": [1, 1]})
    for r, tid in enumerate(TRACKS):
        partial = load_ply(os.path.join(obj_dir, f"{tid}_partial.ply"))
        comp = load_ply(os.path.join(obj_dir, f"{tid}.ply"))
        centre = comp.mean(0)
        p, c = partial - centre, comp - centre
        rot = display_rotation(c[:, [0, 2]])
        p_h, c_h = p[:, [0, 2]] @ rot.T, c[:, [0, 2]] @ rot.T

        n_in = meta[tid]["completion_input_points"]
        ax_bev, ax_side = axes[r, 0], axes[r, 1]
        for ax, (cu, cv, pu, pv), ylabel in (
            (ax_bev, (c_h[:, 0], c_h[:, 1], p_h[:, 0], p_h[:, 1]), "width (m)"),
            (ax_side, (c_h[:, 0], -c[:, 1], p_h[:, 0], -p[:, 1]), "height (m)"),
        ):
            ax.scatter(cu, cv, s=3, c="#00b33c", alpha=0.5, linewidths=0)
            ax.scatter(pu, pv, s=3, c="#1f6fff", alpha=0.8, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xlim(-3, 3)
            ax.set_xlabel("length (m)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        ax_bev.set_ylim(-1.8, 1.8)
        ax_side.set_ylim(-1.2, 1.2)
        ax_bev.set_title(f"track {tid}: bird's-eye view, {n_in} input points", fontsize=10)
        ax_side.set_title("side elevation", fontsize=10)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#1f6fff", label="single-frame partial input"),
        plt.Line2D([], [], marker="o", ls="", color="#00b33c", label="PCN completion (4096 points)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
