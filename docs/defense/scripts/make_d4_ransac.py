"""D4: RANSAC ground fit on one real scan, bird's-eye view (defense slide 2.2.1).

Replays remove_ground()'s seeded loop (src/pipeline.py) on one preprocessed seq-08 frame,
recording every try, and plots (left) one early bad try and (right) the best of all tries.
The replay is asserted to pick the same plane as remove_ground() itself. Counts are an
illustration on one frame, not a thesis result.

Usage: .venv/Scripts/python.exe docs/defense/scripts/make_d4_ransac.py [frame]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import PIPELINE_CONFIG, preprocess_frame, remove_ground  # noqa: E402

SEQ, FRAME = "08", int(sys.argv[1]) if len(sys.argv) > 1 else 2500
OUT = ROOT / "docs/defense/assets/d4_ransac_real.png"
GROUND, OTHER, SAMPLE = "#C8A464", "#B8BEC6", "#D7263D"
XLIM, YLIM = (-30, 30), (-15, 15)

cfg = PIPELINE_CONFIG
xyz = np.fromfile(ROOT / f"dataset/sequences/{SEQ}/velodyne/{FRAME:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
pcd = preprocess_frame(xyz, cfg)
P = np.asarray(pcd.points)

# --- replay remove_ground's loop exactly (same RNG, same skip rule, same strict '>')
rng = np.random.default_rng(42)
tries, best = [], None
for it in range(cfg["ransac_iterations"]):
    idx = rng.choice(len(P), size=cfg["ransac_n"], replace=False)
    p0, p1, p2 = P[idx]
    n = np.cross(p1 - p0, p2 - p0)
    if np.linalg.norm(n) < 1e-10:
        continue
    n /= np.linalg.norm(n)
    d = -n.dot(p0)
    mask = np.abs(P @ n + d) < cfg["ransac_distance_threshold"]
    t = dict(it=it + 1, idx=idx, n=n, mask=mask, count=int(mask.sum()))
    tries.append(t)
    if best is None or t["count"] > best["count"]:
        best = t

_, _, _, ref_inliers = remove_ground(pcd, cfg)
assert len(ref_inliers) == best["count"], "replay diverged from remove_ground()"

# one early, clearly-bad try that still passes the 'roughly horizontal' check (a tilted road plane)
bad = next(t for t in tries if t["count"] < 0.3 * best["count"] and 0.5 < abs(t["n"][2]) < 0.995)
print(f"frame {FRAME}: {len(P)} points; best = try #{best['it']} ({best['count']} inliers); "
      f"bad = try #{bad['it']} ({bad['count']} inliers, tilt {np.degrees(np.arccos(abs(bad['n'][2]))):.1f} deg)")

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 15})
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), constrained_layout=True)
panels = [(bad, f"One random try (try #{bad['it']})", f"{bad['count']:,} points agree", "#D7263D"),
          (best, f"Best of {cfg['ransac_iterations']} tries (try #{best['it']})",
           f"{best['count']:,} points agree → removed as ground", "#2E7D32")]
for ax, (t, title, verdict, vcol) in zip(axes, panels):
    m = t["mask"]
    ax.scatter(P[~m, 0], P[~m, 1], s=0.6, c=OTHER, linewidths=0, rasterized=True)
    ax.scatter(P[m, 0], P[m, 1], s=0.9, c=GROUND, linewidths=0, rasterized=True)
    s = P[t["idx"]]
    ax.scatter(s[:, 0], s[:, 1], s=260, marker="o", c=SAMPLE, edgecolors="white", linewidths=2.5, zorder=5)
    ax.scatter([0], [0], s=140, marker="^", c="black", zorder=6)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, left)")
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left")
    ax.text(0.01, 0.97, verdict, transform=ax.transAxes, va="top", fontsize=17, fontweight="bold",
            color=vcol, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))

h = [plt.Line2D([], [], ls="", marker="o", ms=11, mfc=SAMPLE, mec="white", label="3 random points (define a plane)"),
     plt.Line2D([], [], ls="", marker="s", ms=10, mfc=GROUND, mec=GROUND, label="agree: within 0.2 m of that plane"),
     plt.Line2D([], [], ls="", marker="s", ms=10, mfc=OTHER, mec=OTHER, label="do not agree"),
     plt.Line2D([], [], ls="", marker="^", ms=11, mfc="black", mec="black", label="sensor")]
fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False, fontsize=15)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
