"""(3) 3D multi-angle render of one completed track (headless), plus an
interactive Open3D one-liner the user can run themselves.

Static: partial (blue) + completed (green) from 3 azimuths -> PNG.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


def main(seq="08", tid=9882, out_name=None):
    out_name = out_name or f"seq{seq}_completion_3d_{tid}.png"
    obj = os.path.join(PROJECT_ROOT, f"output/{seq}/objects")
    p = load(os.path.join(obj, f"{tid}_partial.ply"))
    c = load(os.path.join(obj, f"{tid}.ply"))
    ctr = c.mean(0)
    p = p - ctr
    c = c - ctr
    # flip Y so up is up (saved frame is Y-down)
    p[:, 1] *= -1
    c[:, 1] *= -1

    views = [(20, -60), (20, 30), (75, -90)]  # (elev, azim)
    fig = plt.figure(figsize=(16, 5.5))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(c[:, 0], c[:, 2], c[:, 1], s=2, c="#00b33c", alpha=0.5,
                   linewidths=0, label="completed")
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=3, c="#1f6fff", alpha=0.9,
                   linewidths=0, label="partial")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X width")
        ax.set_ylabel("Z length")
        ax.set_zlabel("height")
        ax.set_box_aspect((np.ptp(c[:, 0]), np.ptp(c[:, 2]), np.ptp(c[:, 1])))
        ax.set_title(f"view {i+1} (elev={elev}, azim={azim})", fontsize=10)
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Seq {seq} track {tid}: partial (blue) vs PCN-completed (green), 3D",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", out_name)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 9882
    main(tid=tid)
