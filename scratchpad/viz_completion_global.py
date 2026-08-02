"""(1) Global-scene overlay: completed cars dropped back into a full frame.

Plots one frame's raw cloud (grey, global BEV) and overlays every PCN-completed
car track active in that frame (green), so the completions are seen in context.
Note: completed clouds sit at their completion ref-frame global position, so
moving cars may be slightly offset; parked cars line up.
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)
from pipeline import load_calib, load_poses  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(seq="08", frame=2543, half=45.0, out_name="seq08_completion_global.png"):
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
    bins = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    Tr = load_calib(os.path.join(seq_dir, "calib.txt"))["Tr"]
    T = poses[frame] @ Tr

    raw = np.fromfile(bins[frame], dtype=np.float32).reshape(-1, 4)[:, :3]
    g = (T[:3, :3] @ raw.T).T + T[:3, 3]
    ego = T[:3, 3]

    meta = json.load(open(os.path.join(PROJECT_ROOT, f"output/{seq}/tracks.json")))
    active = [t for t in meta["tracks"]
              if t.get("completed") and t["first_frame"] <= frame <= t["last_frame"]]

    fig, ax = plt.subplots(figsize=(12, 12))
    # raw context: horizontal plane is global X-Z (Y is up/down)
    ax.scatter(g[:, 0], g[:, 2], s=0.4, c="0.8", linewidths=0)

    obj_dir = os.path.join(PROJECT_ROOT, f"output/{seq}/objects")
    n = 0
    for t in active:
        c = o3d.io.read_point_cloud(os.path.join(obj_dir, f"{t['track_id']}.ply"))
        c = np.asarray(c.points)
        if len(c) == 0:
            continue
        ax.scatter(c[:, 0], c[:, 2], s=3, c="#00b33c", linewidths=0)
        cx, cz = c[:, 0].mean(), c[:, 2].mean()
        ax.text(cx, cz + 1.5, str(t["track_id"]), fontsize=7, color="#0a6b2a",
                ha="center")
        n += 1

    ax.scatter([ego[0]], [ego[2]], marker="^", s=160, c="red", zorder=6,
               label="ego")
    ax.set_xlim(ego[0] - half, ego[0] + half)
    ax.set_ylim(ego[2] - half, ego[2] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("global X (m)")
    ax.set_ylabel("global Z (m)")
    ax.set_title(f"Seq {seq} frame {frame}: {n} PCN-completed cars (green) "
                 f"in raw scene (grey)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="0.7", label="raw LiDAR (frame)"),
        plt.Line2D([], [], marker="o", ls="", color="#00b33c", label="completed car"),
        plt.Line2D([], [], marker="^", ls="", color="red", label="ego"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", out_name)
    fig.savefig(out, dpi=130)
    print(f"saved {out}  ({n} completed cars overlaid)")


if __name__ == "__main__":
    main()
