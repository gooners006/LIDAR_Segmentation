"""T11 — moving-car completion plausibility on output/08 (numbers + figure only).

Completion validation (#29/#32) covered STATIC cars only, but production
completes movers too. This splits output/08's completed car tracks into
moving vs static by a kinematic criterion and reports the plausible-car-box
rate for each group, using the exact #27/#28 recipe:
  dims = desc-sorted global-axis extents; plausible box
  L in [3.3, 4.9], W in [1.5, 2.1], H in [1.1, 1.7].

Motion criterion (label-free): net horizontal displacement of the track's
per-frame centroid (median of first 5 vs last 5 frames, X-Z ground plane,
Y is vertical per #26/#27):
  static  = net_disp <= 2.0 m   (matches amodal_gt's 2.0 m center-spread guard)
  moving  = net_disp >= 5.0 m   (unambiguous translation; no parked car drifts 5 m)
  ambiguous 2-5 m reported separately, excluded from the headline comparison.

No fixes; observational only.
Run: .venv\\Scripts\\python.exe scratchpad/t11_mover_plausibility.py
"""

import json
import os

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "08")
FIG_DIR = os.path.join(ROOT, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

L_BAND, W_BAND, H_BAND = (3.3, 4.9), (1.5, 2.1), (1.1, 1.7)
STATIC_MAX = 2.0
MOVING_MIN = 5.0


def dims(pts):
    return np.sort(pts.max(0) - pts.min(0))[::-1]


def plausible(d):
    L, W, H = d
    return (L_BAND[0] <= L <= L_BAND[1] and
            W_BAND[0] <= W <= W_BAND[1] and
            H_BAND[0] <= H <= H_BAND[1])


def net_disp(track):
    xz = np.array(track["centroid_history"])[:, [0, 2]]
    k = min(5, len(xz))
    return float(np.linalg.norm(np.median(xz[-k:], 0) - np.median(xz[:k], 0)))


def load(tid, suffix=""):
    p = os.path.join(OUT_DIR, "objects", f"{tid}{suffix}.ply")
    if not os.path.exists(p):
        return None
    return np.asarray(o3d.io.read_point_cloud(p).points)


def summarize(group, label):
    n_ok = 0
    dvals = []
    for t in group:
        pts = load(t["track_id"])
        if pts is None:
            continue
        d = dims(pts)
        dvals.append(d)
        if plausible(d):
            n_ok += 1
    dvals = np.array(dvals)
    n = len(dvals)
    rate = n_ok / n if n else 0.0
    med = dvals.mean(0) if n else [0, 0, 0]  # placeholder; median below
    med = np.median(dvals, 0) if n else np.zeros(3)
    print(f"{label:12s} n={n:4d}  plausible={n_ok:4d} ({rate:.1%})  "
          f"median L/W/H = {med[0]:.2f}/{med[1]:.2f}/{med[2]:.2f}")
    return {"n": n, "n_plausible": n_ok, "rate": round(rate, 3),
            "median_LWH": [round(float(x), 2) for x in med]}


def bev_figure(movers, path):
    """BEV (X-Z ground plane) of mover completions; green=plausible, red=fails."""
    movers = sorted(movers, key=lambda t: -t["_net"])  # fastest first
    n = min(12, len(movers))
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        t = movers[i]
        tid = t["track_id"]
        comp = load(tid)
        part = load(tid, "_partial")
        ax = axes[i]
        ax.axis("on")
        ax.set_aspect("equal")
        if part is not None:
            ax.scatter(part[:, 0], part[:, 2], s=1, c="#1f77b4", alpha=0.5, label="partial")
        d = dims(comp)
        ok = plausible(d)
        ax.scatter(comp[:, 0], comp[:, 2], s=1, c=("#2ca02c" if ok else "#d62728"), alpha=0.4)
        ax.set_title(f"t{tid} d={t['_net']:.0f}m\nL/W/H {d[0]:.1f}/{d[1]:.1f}/{d[2]:.1f}",
                     fontsize=7, color=("#2ca02c" if ok else "#d62728"))
        ax.tick_params(labelsize=5)
    fig.suptitle("T11: moving-car completions, BEV X-Z (green=plausible car box, red=fails)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    print(f"Saved figure: {path}")


def main():
    meta = json.load(open(os.path.join(OUT_DIR, "tracks.json")))
    comp = [t for t in meta["tracks"] if t.get("completed")]
    for t in comp:
        t["_net"] = net_disp(t)

    statics = [t for t in comp if t["_net"] <= STATIC_MAX]
    movers = [t for t in comp if t["_net"] >= MOVING_MIN]
    ambiguous = [t for t in comp if STATIC_MAX < t["_net"] < MOVING_MIN]

    print(f"completed car tracks: {len(comp)}")
    print(f"  static (net<= {STATIC_MAX}m): {len(statics)}")
    print(f"  moving (net>= {MOVING_MIN}m): {len(movers)}")
    print(f"  ambiguous ({STATIC_MAX}-{MOVING_MIN}m): {len(ambiguous)}")
    print("-" * 70)
    res = {
        "static": summarize(statics, "STATIC"),
        "moving": summarize(movers, "MOVING"),
        "ambiguous": summarize(ambiguous, "AMBIGUOUS"),
        "all": summarize(comp, "ALL"),
    }

    bev_path = os.path.join(FIG_DIR, "t11_mover_completions_bev.png")
    bev_figure(movers, bev_path)

    res["thresholds"] = {"static_max_m": STATIC_MAX, "moving_min_m": MOVING_MIN,
                         "plausible_box": {"L": L_BAND, "W": W_BAND, "H": H_BAND}}
    out = os.path.join(ROOT, "output", "experiments", "t10_clustering", "t11_mover_plausibility.json")
    # keep T11 json alongside experiments; use a dedicated path
    out = os.path.join(ROOT, "output", "experiments", "t11_mover_plausibility.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
