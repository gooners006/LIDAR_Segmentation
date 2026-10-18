"""Private check (qa_backup, NOT a thesis number): heading of saved completed clouds vs the Z axis.

harmonize.py feeds pairs to the judge assuming Z = length; the ceiling is de-rotated by yaw_deg.
This measures the main-axis (PCA, X-Z plane) angle of every scored completed cloud to Z.
Run: .venv\Scripts\python.exe docs/defense/scripts/check_completion_heading.py
"""
import json
from pathlib import Path
import numpy as np
import open3d as o3d

GEN = Path(__file__).resolve().parents[3] / "output" / "experiments" / "plausibility_gen"
ang = []
for seq in sorted(p.name for p in GEN.iterdir()):
    for t in json.load(open(GEN / seq / "tracks.json"))["tracks"]:
        f = GEN / seq / "objects" / f"{t['track_id']}.ply"
        if not t.get("completed") or not f.exists() or not (GEN / seq / "objects" / f"{t['track_id']}_partial.ply").exists():
            continue
        xz = np.asarray(o3d.io.read_point_cloud(str(f)).points)[:, [0, 2]]
        w, v = np.linalg.eigh(np.cov((xz - xz.mean(0)).T))
        major = v[:, 1]                     # (x, z) of the largest-variance axis
        ang.append(np.degrees(np.arctan2(abs(major[0]), abs(major[1]))))  # 0 = along Z, 90 = along X
ang = np.array(ang)
print(f"N={len(ang)}  within 20 deg of Z: {np.mean(ang <= 20):.3f}   20-70: {np.mean((ang > 20) & (ang < 70)):.3f}"
      f"   within 20 deg of X: {np.mean(ang >= 70):.3f}   median {np.median(ang):.1f} deg")
print("histogram (10-deg bins, 0=Z .. 90=X):", np.histogram(ang, bins=9, range=(0, 90))[0].tolist())
