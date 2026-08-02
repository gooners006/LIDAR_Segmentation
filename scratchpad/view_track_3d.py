"""Interactive Open3D view of one completed track: partial (blue) + completed
(green). Run yourself (GUI window):
    .venv\\Scripts\\python.exe scratchpad/view_track_3d.py <track_id> [seq]
Drag to rotate, scroll to zoom, close window to exit.
"""

import os
import sys

import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tid = sys.argv[1] if len(sys.argv) > 1 else "9882"
    seq = sys.argv[2] if len(sys.argv) > 2 else "08"
    obj = os.path.join(PROJECT_ROOT, f"output/{seq}/objects")

    partial = o3d.io.read_point_cloud(os.path.join(obj, f"{tid}_partial.ply"))
    comp = o3d.io.read_point_cloud(os.path.join(obj, f"{tid}.ply"))

    # flip Y so gravity points down in the viewer (saved frame is Y-down)
    flip = np.diag([1.0, -1.0, 1.0])
    partial.rotate(flip, center=(0, 0, 0))
    comp.rotate(flip, center=(0, 0, 0))

    partial.paint_uniform_color([0.12, 0.44, 1.0])   # blue
    comp.paint_uniform_color([0.0, 0.70, 0.24])      # green

    print(f"track {tid} (seq {seq}): partial={len(partial.points)} pts (blue), "
          f"completed={len(comp.points)} pts (green)")
    o3d.visualization.draw_geometries(
        [comp, partial], window_name=f"track {tid} — partial(blue) vs completed(green)",
        width=1400, height=900)


if __name__ == "__main__":
    main()
