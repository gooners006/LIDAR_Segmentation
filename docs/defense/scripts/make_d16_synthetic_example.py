"""D16 (slide 3.4.3): one held-out synthetic car, simulated scan -> PCN output -> true model.

Protocol = the one behind the thesis synthetic numbers (CD 0.16 m, F@0.1 m 0.76; findings #26,
#28; scratchpad/verify_pcn_step1.py): the first 30 car models of PCN's validation split (never
trained on), each given a KITTI-like simulated scan (src/train_pcn.py _render_kitti_like),
normalised as in training (true centre, true radius), completed by pcn_kitti_best.pth, scored in
metres. Fixed pick rule, not by eye: the car with the MEDIAN Chamfer distance of the 30 (a typical
car, not the best). Seeds fixed (np.random 0, rng 0).
Sanity check (pre-registered): the 30-car mean must land within 0.14-0.18 m CD, else this run
does not reproduce the thesis protocol and the figure is not used.

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/make_d16_synthetic_example.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from completion import chamfer_distance, f_score  # noqa: E402
from pcn import PCN  # noqa: E402
from train_pcn import TRAIN_CONFIG, ShapeNetCompletionDataset  # noqa: E402

CKPT = REPO / "checkpoints" / "pcn_kitti_best.pth"
OUT = REPO / "docs/defense/assets/d16_synthetic_example.png"
CAR, N_EVAL, N_IN = "02958343", 30, 256
SCAN, ADDED, MODEL, INK, MUTED = "#2B5C8A", "#D9822B", "#9AA0A6", "#1F2933", "#6B7280"

np.random.seed(0)
rng = np.random.default_rng(0)
cfg = dict(TRAIN_CONFIG, kitti_like_partial=True)
ds = ShapeNetCompletionDataset(cfg, split="val")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PCN(num_coarse=1024, grid_size=2).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model_state_dict"])
model.eval()


def fix_size(p, n):
    idx = rng.choice(len(p), n, replace=len(p) < n)
    return p[idx].astype(np.float32)


rows = []
for obj, syn in ds.items:
    if syn != CAR:
        continue
    mesh = ds._load_and_scale(obj, cfg["category_scale_m"][syn])
    if mesh is None or mesh.get_surface_area() < 1e-6:
        continue
    gt = ds._sample_gt(mesh)
    part = ds._render_kitti_like(mesh, cfg["category_scale_m"][syn])
    if part is None or len(part) < 16:
        continue
    c = gt.mean(0)
    r = float(np.linalg.norm(gt - c, axis=1).max())
    with torch.no_grad():
        _, fine = model(torch.from_numpy(fix_size((part - c) / r, N_IN)).unsqueeze(0).to(device))
    pred = fine.squeeze(0).cpu().numpy() * r + c
    rows.append({"obj": obj, "gt": gt, "part": part, "pred": pred,
                 "cd": chamfer_distance(pred, gt), "f": f_score(pred, gt, threshold=0.10)})
    if len(rows) == N_EVAL:
        break

cd = np.array([x["cd"] for x in rows])
fs = np.array([x["f"] for x in rows])
print(f"{len(rows)} held-out cars: CD {cd.mean():.3f} +- {cd.std():.3f} m, F@0.1 {fs.mean():.3f}")
assert 0.14 <= cd.mean() <= 0.18, "does not reproduce the thesis protocol; do not use the figure"
pick = rows[int(np.argsort(cd)[len(cd) // 2])]
print(f"median car: {Path(pick['obj']).parts[-3]}  CD {pick['cd']:.3f} m  F {pick['f']:.3f}  "
      f"scan {len(pick['part'])} pts")

gt0 = pick["gt"]


def to_view(p):
    """ShapeNet frame (X width, Y up, Z length) -> (length, width, up), ground at 0."""
    return np.column_stack([p[:, 2], p[:, 0], p[:, 1] - gt0[:, 1].min()])


P, Q, G = to_view(pick["part"]), to_view(pick["pred"]), to_view(gt0)
ctr = G.mean(0)
for ax in (0, 1):                                  # seen side on top, seen end on the right (as D13/D14)
    if P[:, ax].mean() - ctr[ax] < 0:
        for A in (P, Q, G):
            A[:, ax] = 2 * ctr[ax] - A[:, ax]
for A in (P, Q, G):
    A[:, :2] -= ctr[:2]

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 16})
fig, axes = plt.subplots(2, 3, figsize=(18, 6.6), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.0, 0.85]})
for col, (pts, colr, s) in enumerate([(P, SCAN, 5), (Q, ADDED, 3), (G, MODEL, 3)]):
    for row, (a, b) in enumerate([(0, 1), (0, 2)]):
        ax = axes[row, col]
        ax.scatter(pts[:, a], pts[:, b], s=s, c=colr, lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-1.15, 1.15) if row == 0 else ax.set_ylim(-0.15, 1.75)
        ax.axis("off")
        if row == 1:
            ax.axhline(-0.03, c="#D1D5DB", lw=1.5, zorder=0)
for row, name in enumerate(["Top view", "Side view"]):
    axes[row, 0].text(-0.03, 0.5, name, transform=axes[row, 0].transAxes, rotation=90,
                      ha="right", va="center", fontsize=18, fontweight="bold", color=INK)
titles = [("Simulated scan", "input, never seen in training", SCAN),
          ("PCN output", f"Chamfer {pick['cd']:.2f} m, F-score {pick['f']:.2f}", ADDED),
          ("True model", "the answer", MUTED)]
for col, (t, sub, colr) in enumerate(titles):
    axes[0, col].set_title(f"{t}\n", fontsize=20, fontweight="bold", loc="left", color=colr)
    axes[0, col].text(0, 1.02, sub, transform=axes[0, col].transAxes, fontsize=16, color=MUTED)
ax = axes[0, 0]                                     # empty lower area of the scan panel
ax.plot([1.4, 2.4], [-1.05, -1.05], c=INK, lw=2.5)
ax.text(1.9, -0.99, "1 m", ha="center", va="bottom", fontsize=14, color=INK)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
