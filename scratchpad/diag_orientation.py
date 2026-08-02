"""Diagnose completion quality vs orientation recoverability (not point count).

For each completed track, measure how 'elongated' the partial's BEV footprint is
(ratio of horizontal PCA eigenvalues). A clean one-sided car view is elongated
(ratio >> 1) so the major axis reliably gives heading. An L-shape (two faces) or
a merge of two cars has a near-1 ratio -> heading is ambiguous -> bad completion.
Cross-reference with the completed car's width (a wide completion = merge or 90deg
orientation error).
"""

import json
import os

import numpy as np
import open3d as o3d

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "08_demo")


def horiz_pca_ratio(pts):
    xy = pts[:, :2] - pts[:, :2].mean(0)
    w, _ = np.linalg.eigh(xy.T @ xy)
    w = np.sort(w)[::-1]
    return float(np.sqrt(w[0] / max(w[1], 1e-9)))  # std ratio along major/minor


def main():
    meta = json.load(open(os.path.join(ROOT, "tracks.json")))
    comp = [t for t in meta["tracks"] if t.get("completed")]
    rows = []
    for t in comp:
        tid = t["track_id"]
        partial = np.asarray(o3d.io.read_point_cloud(
            os.path.join(ROOT, "objects", f"{tid}_partial.ply")).points)
        full = np.asarray(o3d.io.read_point_cloud(
            os.path.join(ROOT, "objects", f"{tid}.ply")).points)
        ext = np.sort(full.max(0) - full.min(0))[::-1]  # L, W, H (desc)
        rows.append(dict(tid=tid, n=t["completion_input_points"],
                         ratio=horiz_pca_ratio(partial),
                         L=ext[0], W=ext[1], H=ext[2]))

    # A completion is "suspect" if the body is too short (mis-orientation) or too
    # wide (merge / 90deg error): real KITTI cars ~3.5-4.5 x 1.6-2.0 m.
    for r in rows:
        r["suspect"] = (r["L"] < 3.2) or (r["W"] > 2.3)

    rows.sort(key=lambda r: r["n"])
    print(f"{'tid':>4} {'in_pts':>6} {'bev_ratio':>9} {'L':>5} {'W':>5} {'H':>5}  flag")
    for r in rows:
        print(f"{r['tid']:>4} {r['n']:>6} {r['ratio']:>9.2f} "
              f"{r['L']:>5.2f} {r['W']:>5.2f} {r['H']:>5.2f}  {'SUSPECT' if r['suspect'] else ''}")

    rr = np.array([r["ratio"] for r in rows])
    nn = np.array([r["n"] for r in rows])
    sus = np.array([r["suspect"] for r in rows])
    print("-" * 50)
    print(f"tracks: {len(rows)}  suspect: {sus.sum()}")
    print(f"mean BEV ratio  — good: {rr[~sus].mean():.2f}   suspect: {rr[sus].mean():.2f}")
    print(f"mean in_pts     — good: {nn[~sus].mean():.0f}    suspect: {nn[sus].mean():.0f}")
    print(f"corr(in_pts, bev_ratio) = {np.corrcoef(nn, rr)[0,1]:+.2f}  "
          f"(negative => more points tend to mean LESS elongated/ambiguous)")


if __name__ == "__main__":
    main()
