"""Backup slide (normalisation ablation): one real car, same PCN checkpoint, two input frames.

Car C (seq 03, track 192; the D13/D14/D15 car), its exact production scan in the sensor frame
(recovered as in make_d15_normalisation.py).
  left:  the original naive path, verbatim from git 896ff9c src/completion.py complete():
         centre on the scan's own centroid, divide by its own radius, rotate into its 3D PCA axes
         (longest axis -> X), run PCN, undo;
  right: the production output, the saved completion of plausibility_gen (192.ply), unchanged.
Same checkpoint (pcn_kitti_best.pth) on both sides. Both are drawn in the car's own frame from the
production L-shape fit (display only). Car fixed in advance (car C), not chosen for this figure.
Illustration of ch4 §Completion Ablation Studies "Inference normalisation"; not a thesis number.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_b57_normalisation_ablation.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from completion import PointCloudCompleter  # noqa: E402
from pcn import PCN  # noqa: E402
from pipeline import load_calib, load_poses  # noqa: E402

SEQ, TRACK = "03", 192
GEN = REPO / "output/experiments/plausibility_gen" / SEQ
CKPT = REPO / "checkpoints" / "pcn_kitti_best.pth"
OUT = REPO / "docs/defense/assets/b57_normalisation_ablation.png"
SCAN, ADDED, INK, MUTED = "#2B5C8A", "#D9822B", "#1F2933", "#6B7280"

meta = [t for t in json.load(open(GEN / "tracks.json"))["tracks"] if t["track_id"] == TRACK][0]
T = load_poses(str(REPO / f"dataset/sequences/{SEQ}/poses.txt"))[meta["completion_ref_frame"]] @ \
    load_calib(str(REPO / f"dataset/sequences/{SEQ}/calib.txt"))["Tr"]


def to_sensor(path):
    g = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    return (g - T[:3, 3]) @ T[:3, :3]


scan = to_sensor(GEN / "objects" / f"{TRACK}_partial.ply")
prod = to_sensor(GEN / "objects" / f"{TRACK}.ply")

# ---- the original naive path (git 896ff9c complete(), verbatim logic)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PCN(num_coarse=1024, grid_size=2).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model_state_dict"])
model.eval()
rng = np.random.default_rng(0)
pts = scan.astype(np.float32)
centroid = pts.mean(0)
pc = pts - centroid
radius = float(np.linalg.norm(pc, axis=1).max())
w, v = np.linalg.eigh(np.cov(pc.T))
axes = v[:, w.argsort()[::-1]]
if np.linalg.det(axes) < 0:
    axes[:, 2] *= -1
norm = (pc @ axes) / radius
idx = rng.choice(len(norm), 256, replace=len(norm) < 256)
with torch.no_grad():
    _, fine = model(torch.from_numpy(norm[idx].astype(np.float32)).unsqueeze(0).to(device))
naive = fine.squeeze(0).cpu().numpy() * radius @ axes.T + centroid

# ---- display frame: production L-shape fit (length left-right, width up-down, up)
frame, skip = PointCloudCompleter().estimate_canonical_frame(scan)
assert skip is None, skip
B = frame["basis"]                                   # columns: width, up, length


def view(p):
    q = p @ B
    return np.column_stack([q[:, 2], q[:, 0], q[:, 1]])  # length, width, up


S, N, P = view(scan), view(naive), view(prod)
mid = 0.5 * (P.min(0) + P.max(0))
ground = P[:, 2].min()
for A in (S, N, P):
    A[:, :2] -= mid[:2]
    A[:, 2] -= ground
ext = lambda A: np.ptp(A, axis=0)                    # noqa: E731
print(f"scan {len(scan)} pts; naive extent L/W/H {ext(N).round(2)}; production {ext(P).round(2)}")

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 16})
fig, axs = plt.subplots(2, 2, figsize=(14, 7.2), constrained_layout=True,
                        gridspec_kw={"height_ratios": [1.0, 0.8]})
cols = [(N, "Scan's own 3D PCA frame", "centroid + radius of the scan; longest axis taken as X"),
        (P, "Car frame (production)", "check, turn, shift, size (slides 19–20)")]
for c, (C, title, sub) in enumerate(cols):
    for r, (a, b) in enumerate([(0, 1), (0, 2)]):
        ax = axs[r, c]
        ax.scatter(C[:, a], C[:, b], s=3, c=ADDED, lw=0, alpha=0.75, rasterized=True)
        ax.scatter(S[:, a], S[:, b], s=5, c=SCAN, lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.4, 2.4) if r == 0 else ax.set_ylim(-0.8, 2.6)
        if r == 1:
            ax.axhline(-0.03, c="#D1D5DB", lw=1.5, zorder=0)
    axs[0, c].set_title(f"{title}\n", fontsize=20, fontweight="bold", loc="left", color=INK)
    axs[0, c].text(0, 1.02, sub, transform=axs[0, c].transAxes, fontsize=15, color=MUTED)
for r, name in enumerate(["Top view", "Side view"]):
    axs[r, 0].text(-0.03, 0.5, name, transform=axs[r, 0].transAxes, rotation=90, ha="right",
                   va="center", fontsize=18, fontweight="bold", color=INK)
axs[0, 1].plot([2.0, 3.0], [-2.25, -2.25], c=INK, lw=2.5)
axs[0, 1].text(2.5, -2.18, "1 m", ha="center", va="bottom", fontsize=14, color=INK)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
