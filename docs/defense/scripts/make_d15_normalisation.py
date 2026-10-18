"""D15 (slide 2.3): the four geometry steps before PCN, "Check, Turn, Shift, Size", on car C
(seq 03, track 192, the D13/D14 car). Illustration on one real car, not a result.

Uses the production code itself (src/completion.py PointCloudCompleter.estimate_canonical_frame
and _lshape_axes) on the exact scan production completed: the saved world-frame partial
(output/experiments/plausibility_gen/03/objects/192_partial.ply) mapped back to the sensor frame
of its reference frame with that frame's pose (poses[ref] @ Tr, as src/main.py does).
Check: the recovered scan must reproduce the saved ref_fit_length / ref_fit_width (1 mm).
Length target: the fixed 4.14 m prior. Production used this car's own per-car estimate, which
tracks.json does not store; the push differs by a few centimetres at most for this car.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d15_normalisation.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from completion import (COMPLETION_CAR_WIDTH_PRIOR, COMPLETION_SCALE_CORRECTION,  # noqa: E402
                        COMPLETION_UP_SHIFT, PointCloudCompleter)
from pipeline import load_calib, load_poses  # noqa: E402

SEQ, TRACK = "03", 192
GEN = REPO / "output/experiments/plausibility_gen" / SEQ
OUT = REPO / "docs/defense/assets/d15_check_turn_shift_size.png"
SCAN, INK, MUTED, OK, PUSH = "#2B5C8A", "#1F2933", "#6B7280", "#12A150", "#D9822B"

# ---- the exact production input, back in its sensor frame
meta = [t for t in json.load(open(GEN / "tracks.json"))["tracks"] if t["track_id"] == TRACK][0]
ref = meta["completion_ref_frame"]
T = load_poses(str(REPO / f"dataset/sequences/{SEQ}/poses.txt"))[ref] @ \
    load_calib(str(REPO / f"dataset/sequences/{SEQ}/calib.txt"))["Tr"]
glob = np.asarray(o3d.io.read_point_cloud(str(GEN / "objects" / f"{TRACK}_partial.ply")).points)
pts = (glob - T[:3, 3]) @ T[:3, :3]                     # world -> sensor (Z up, ego at origin)

comp = PointCloudCompleter()                           # geometry only, no network needed
frame, skip = comp.estimate_canonical_frame(pts)
assert skip is None, skip
assert abs(frame["fit_length"] - meta["ref_fit_length"]) < 1e-3, (frame["fit_length"], meta["ref_fit_length"])
assert abs(frame["fit_width"] - meta["ref_fit_width"]) < 1e-3, (frame["fit_width"], meta["ref_fit_width"])
print(f"check OK: fit {frame['fit_length']:.3f} x {frame['fit_width']:.3f} m, ref frame {ref}, {len(pts)} pts")

# ---- quantities for each panel
xy = pts[:, :2]
mid = xy.mean(0)
u_len, u_wid, L, W = comp._lshape_axes(xy - mid)
a, b = (xy - mid) @ u_len, (xy - mid) @ u_wid
rect = np.array([[a.min(), b.min()], [a.max(), b.min()], [a.max(), b.max()], [a.min(), b.max()]])
rect = mid + rect[:, :1] * u_len + rect[:, 1:] * u_wid

basis, center, radius = frame["basis"], frame["center"], frame["radius"]
pc = pts @ basis                                       # canonical: [0] width, [1] up, [2] length
box_mid = 0.5 * (pc.min(0) + pc.max(0))
shift = center - box_mid                               # width push, -0.25 up, length push
norm = (pc - center) / radius

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 15})
fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
heads = [("1  Check", "one whole car?"), ("2  Turn", "length on the training axis"),
         ("3  Shift", "centre of the whole car"), ("4  Size", "scale the network trained on")]
for ax, (h, sub) in zip(axes, heads):
    ax.set_title(f"{h}\n", fontsize=20, fontweight="bold", loc="left", color=INK)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=15, color=MUTED)
    ax.set_aspect("equal"); ax.axis("off")

# 1 Check: footprint in the sensor frame + fitted rectangle + direction to the sensor
ax = axes[0]
p = xy - mid
ax.scatter(p[:, 0], p[:, 1], s=3, c=SCAN, lw=0, rasterized=True)
ax.add_patch(Polygon(rect - mid, closed=True, fill=False, ec=OK, lw=2.5, ls="--"))
to_sensor = -mid / np.linalg.norm(mid)
ax.add_patch(FancyArrowPatch(tuple(1.1 * to_sensor), tuple(2.0 * to_sensor), arrowstyle="-|>",
                             mutation_scale=18, color=MUTED, lw=1.8))
ax.text(*(2.35 * to_sensor), "to sensor", color=MUTED, fontsize=13, ha="center", va="center")
ax.text(0.5, -0.04, f"fitted {L:.1f} m × {W:.1f} m\n≥ 2.7 m long, ≤ 2.3 m wide: passes",
        transform=ax.transAxes, ha="center", va="top", fontsize=15, color=OK)
ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)

# 2 Turn and 3 Shift: canonical top view, length left-right, width up-down (sensor side down)
flip = np.sign(center[0]) if abs(center[0]) > 1e-9 else 1.0     # sensor side at the bottom, as in panel 1
for ax in axes[1:3]:
    ax.scatter(pc[:, 2] - box_mid[2], flip * (pc[:, 0] - box_mid[0]), s=3, c=SCAN, lw=0, rasterized=True)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
    ax.text(0, -2.6, "sensor side", ha="center", fontsize=13, color=MUTED)
ax = axes[2]
cz, cx = shift[2], flip * shift[0]
ax.plot(0, 0, marker="x", ms=13, mew=3, c=MUTED)
ax.plot(cz, cx, marker="o", ms=12, c=OK)
ax.add_patch(FancyArrowPatch((0, 0), (cz, cx), arrowstyle="-|>", mutation_scale=18, color=PUSH, lw=2.5))
ax.text(0.5, -0.04, f"× middle of the scan → ● whole-car centre\n"
        f"{abs(shift[0]):.2f} m away from the sensor (1.9 m width), "
        f"{abs(shift[2]):.2f} m along, {COMPLETION_UP_SHIFT:.2f} m down",
        transform=ax.transAxes, ha="center", va="top", fontsize=13.5, color=INK)

# 4 Size: normalised, inside the unit circle
ax = axes[3]
ax.scatter(norm[:, 2], flip * norm[:, 0], s=3, c=SCAN, lw=0, rasterized=True)
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=MUTED, lw=1.8, ls="--"))
ax.plot(0, 0, marker="o", ms=10, c=OK)
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
ax.text(0.5, -0.04, f"÷ {radius:.2f} m (largest distance ÷ {COMPLETION_SCALE_CORRECTION})\n"
        f"then 256 points go into PCN", transform=ax.transAxes, ha="center", va="top",
        fontsize=15, color=INK)

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"width push {shift[0]:+.3f}, up {shift[1]:+.3f}, length push {shift[2]:+.3f}, radius {radius:.3f}; "
      f"width prior {COMPLETION_CAR_WIDTH_PRIOR}")
print(f"wrote {OUT}")
