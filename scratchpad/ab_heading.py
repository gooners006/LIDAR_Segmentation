"""A/B: PCA vs L-shape heading for completion reorientation.

Compares two full-pipeline demo runs that differ ONLY in the heading estimator
used to reorient each partial into the canonical car frame before PCN:

  output/08_ab_pca     --heading-method pca   (A, legacy major-eigenvector)
  output/08_ab_lshape  --heading-method lshape (B, search-based L-shape fit)

Outputs are in the global camera frame (Y = gravity/up), so the top-down BEV is
the X-Z plane. A wrong heading rotates/swaps the completed car's footprint, so we
compare (a) completed L/W/H extents and (b) a BEV overlay for the canonical tids.

  .venv/Scripts/python.exe scratchpad/ab_heading.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_DIR = os.path.join(ROOT, "output", "08_ab_pca")
B_DIR = os.path.join(ROOT, "output", "08_ab_lshape")
DENSE_POOR = [301, 762, 884]
CONTROLS = [21, 321, 43]
GROUND = (0, 2)  # X, Z are the ground plane in the global camera frame (Y = up)


def load(out_dir, tid, suffix=""):
    p = os.path.join(out_dir, "objects", f"{tid}{suffix}.ply")
    if not os.path.exists(p):
        return None
    return np.asarray(o3d.io.read_point_cloud(p).points)


def dims(pts):
    """Length, width, height (desc-sorted extents)."""
    return np.sort(pts.max(0) - pts.min(0))[::-1]


def completed_set(out_dir):
    meta = json.load(open(os.path.join(out_dir, "tracks.json")))
    return {t["track_id"]: t for t in meta["tracks"] if t.get("completed")}


def main():
    for d in (A_DIR, B_DIR):
        if not os.path.isdir(d):
            print(f"MISSING {d} — run the A/B pipeline first."); return

    A, B = completed_set(A_DIR), completed_set(B_DIR)
    common = sorted(set(A) & set(B))
    print(f"completed — pca: {len(A)}  lshape: {len(B)}  common: {len(common)}\n")

    # Aggregate: how much did the completed footprint change, and width sanity.
    rows = []
    for tid in common:
        da = dims(load(A_DIR, tid))
        db = dims(load(B_DIR, tid))
        # heading change ~ how much the L/W split moved (a swap/rotation shows here)
        rows.append((tid, da, db, abs(da[1] - db[1])))
    rows.sort(key=lambda r: -r[3])

    print("Largest completed-footprint changes (width delta):")
    print(f"{'tid':>5} {'pca L/W/H':>20} {'lshape L/W/H':>20} {'dW':>6}")
    for tid, da, db, dw in rows[:15]:
        print(f"{tid:>5} {str(np.round(da,2)):>20} {str(np.round(db,2)):>20} {dw:>6.2f}")

    wa = np.array([r[1][1] for r in rows])
    wb = np.array([r[2][1] for r in rows])
    print(f"\nmean completed width — pca: {wa.mean():.2f}  lshape: {wb.mean():.2f}")
    print(f"wide (>2.2m) cars    — pca: {(wa>2.2).sum()}  lshape: {(wb>2.2).sum()}")

    # BEV overlay for the canonical tids (top-down X-Z).
    gi, gj = GROUND
    tids = DENSE_POOR + CONTROLS
    fig, axes = plt.subplots(2, len(tids), figsize=(2.7 * len(tids), 5.6))
    for col, tid in enumerate(tids):
        partial = load(A_DIR, tid, "_partial")
        for row, (name, out_dir) in enumerate([("PCA", A_DIR), ("L-shape", B_DIR)]):
            ax = axes[row, col]
            full = load(out_dir, tid)
            if full is None:
                ax.set_title(f"tid {tid}\n(not completed)", fontsize=8); continue
            c = partial[:, [gi, gj]].mean(0)
            ax.scatter(full[:, gi] - c[0], full[:, gj] - c[1], s=0.5,
                       c="#00cc44", alpha=0.35)
            ax.scatter(partial[:, gi] - c[0], partial[:, gj] - c[1], s=4,
                       c="#2aa6ff", alpha=0.9)
            d = dims(full)
            ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            grp = "DENSE-POOR" if tid in DENSE_POOR else "control"
            if row == 0:
                ax.set_title(f"tid {tid} ({grp})\nL/W/H {np.round(d,2)}", fontsize=8)
            else:
                ax.set_title(f"L/W/H {np.round(d,2)}", fontsize=8)
            if col == 0:
                ax.set_ylabel(name, fontsize=11)
    fig.suptitle("Completion A/B: top-down BEV (X-Z), partial (blue) + completed car (green)\n"
                 "PCA heading (top) vs L-shape heading (bottom)", fontsize=12)
    out = os.path.join(ROOT, "output", "ab_heading.png")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
