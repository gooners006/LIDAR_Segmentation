"""(2) PCN vs PoinTr on the same tracks.

For tracks completed by BOTH models, show BEV footprints side by side:
left = PCN (partial blue + completed green), right = PoinTr (partial blue +
completed orange). Axis-agnostic dims (sorted horizontal extents) in titles.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    return np.asarray(o3d.io.read_point_cloud(path).points) if os.path.exists(path) else np.zeros((0, 3))


def dims(c):
    """Axis-agnostic car dims: sorted horizontal extents (L>=W) + height(-Y)."""
    ex_x = c[:, 0].max() - c[:, 0].min()
    ex_z = c[:, 2].max() - c[:, 2].min()
    L, W = max(ex_x, ex_z), min(ex_x, ex_z)
    H = c[:, 1].max() - c[:, 1].min()
    return L, W, H


def main(seq="08", n_tracks=6, out_name="seq08_pcn_vs_pointr_shapes.png"):
    pcn = json.load(open(os.path.join(PROJECT_ROOT, f"output/{seq}/tracks.json")))["tracks"]
    ptr = json.load(open(os.path.join(PROJECT_ROOT, f"output/{seq}_pointr/tracks.json")))["tracks"]
    pcn_c = {t["track_id"] for t in pcn if t.get("completed")}
    ptr_c = {t["track_id"] for t in ptr if t.get("completed")}
    both = sorted(pcn_c & ptr_c)
    inp = {t["track_id"]: t.get("completion_input_points", 0) for t in pcn}
    both = sorted(both, key=lambda i: inp.get(i, 0), reverse=True)
    idxs = np.linspace(0, len(both) - 1, n_tracks).astype(int)
    tracks = [both[i] for i in idxs]

    pcn_dir = os.path.join(PROJECT_ROOT, f"output/{seq}/objects")
    ptr_dir = os.path.join(PROJECT_ROOT, f"output/{seq}_pointr/objects")

    fig, axes = plt.subplots(n_tracks, 2, figsize=(10, 3.1 * n_tracks))
    axes = np.atleast_2d(axes)
    for r, tid in enumerate(tracks):
        partial = load(os.path.join(pcn_dir, f"{tid}_partial.ply"))
        cp = load(os.path.join(pcn_dir, f"{tid}.ply"))
        co = load(os.path.join(ptr_dir, f"{tid}.ply"))
        ctr = cp.mean(0) if len(cp) else partial.mean(0)
        for ax, comp, name, col in [
            (axes[r, 0], cp, "PCN", "#00b33c"),
            (axes[r, 1], co, "PoinTr", "#ff7f0e"),
        ]:
            p = partial - ctr
            c = comp - ctr
            ax.scatter(c[:, 0], c[:, 2], s=4, c=col, alpha=0.5, linewidths=0)
            ax.scatter(p[:, 0], p[:, 2], s=4, c="#1f6fff", alpha=0.85, linewidths=0)
            L, W, H = dims(comp)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title(f"track {tid} {name} | L={L:.1f} W={W:.1f} H={H:.1f} "
                         f"(in={inp.get(tid,0)})", fontsize=9)
            ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#1f6fff", label="partial input"),
        plt.Line2D([], [], marker="o", ls="", color="#00b33c", label="PCN completed"),
        plt.Line2D([], [], marker="o", ls="", color="#ff7f0e", label="PoinTr completed"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Seq {seq}: PCN vs PoinTr completion (BEV footprints, same tracks)",
                 y=1.012, fontsize=13)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", out_name)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}  ({len(tracks)} tracks)")


if __name__ == "__main__":
    main()
