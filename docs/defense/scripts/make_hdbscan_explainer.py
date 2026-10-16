"""Backup-slide figure: how HDBSCAN works (toy 2D example, NOT a thesis result).

A bird's-eye toy scene: a near car (dense L-shape) and a far car (sparse L-shape).
Runs the real `hdbscan` library with the pipeline's settings (min_samples 5,
min_cluster_size 10) and draws:
  1. core distance: the circle reaching each point's 5th neighbour (small = dense, big = sparse)
  2. the clusters at three fixed radii, cut from HDBSCAN's own hierarchy
     (single_linkage_tree_ on mutual reachability, i.e. what DBSCAN would give at that radius)
  3. HDBSCAN's final answer, which keeps both cars as separate clusters.

Usage: .venv/Scripts/python.exe docs/defense/scripts/make_hdbscan_explainer.py
"""
from pathlib import Path

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "hdbscan_explainer.png"
MIN_SAMPLES, MIN_CLUSTER = 5, 10            # PIPELINE_CONFIG hdbscan_min_samples / _min_cluster_size
DARK, GREY = "#1F2933", "#9AA5B1"
COL = {"near": "#2563EB", "far": "#EA580C", "merged": "#7C3AED"}


def l_shape(origin, n_long, n_short, jitter, rng):
    """Points along a car's visible L: a 4.2 m side and a 1.8 m end."""
    x0, y0 = origin
    side = np.c_[np.linspace(x0, x0 + 4.2, n_long), np.full(n_long, y0)]
    end = np.c_[np.full(n_short, x0), np.linspace(y0, y0 + 1.8, n_short)]
    return np.r_[side, end[1:]] + rng.normal(0, jitter, (n_long + n_short - 1, 2))


rng = np.random.default_rng(7)
near = l_shape((0.0, 4.8), 60, 26, 0.03, rng)       # dense: ~7 cm spacing
far = l_shape((0.0, 0.0), 18, 8, 0.04, rng)        # sparse: ~25 cm spacing
specks = np.empty((0, 2))   # loose specks dropped: any speck nearer a car than the car-car gap joins that car
P = np.r_[near, far, specks]
truth = np.r_[np.full(len(near), 0), np.full(len(far), 1), np.full(len(specks), -1)]

cl = hdbscan.HDBSCAN(min_samples=MIN_SAMPLES, min_cluster_size=MIN_CLUSTER).fit(P)
core = cl.core_distances_ if hasattr(cl, "core_distances_") else None
if core is None:
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(P).kneighbors(P)
    core = d[:, -1]


def colour(labels):
    """Colour each cluster by which car it holds; a cluster holding both cars = merged."""
    c = np.array([GREY] * len(P), dtype=object)
    for k in set(labels) - {-1}:
        m = labels == k
        has = {t for t in truth[m] if t >= 0}
        c[m] = COL["merged"] if has == {0, 1} else COL["near"] if has == {0} else COL["far"]
    return c


def summary(labels):
    out = []
    for k in sorted(set(labels) - {-1}):
        m = labels == k
        out.append((int(m.sum()), int((truth[m] == 0).sum()), int((truth[m] == 1).sum())))
    return out, int((labels == -1).sum())


slt = cl.single_linkage_tree_
cuts = {r: slt.get_clusters(r, min_cluster_size=MIN_CLUSTER) for r in (0.3, 1.0, 3.2)}
final = cl.labels_

# The figure's story must match what the library actually does; fail loudly if not.
print("core distance, near car median %.2f m, far car median %.2f m"
      % (np.median(core[truth == 0]), np.median(core[truth == 1])))
for r, lab in cuts.items():
    print(f"radius {r} m:", summary(lab))
print("HDBSCAN final:", summary(final))
s_small, s_mid, s_big, s_fin = (summary(cuts[0.3]), summary(cuts[1.0]),
                                summary(cuts[3.2]), summary(final))
assert len(s_small[0]) == 1 and s_small[0][0][2] == 0, "small radius: only the near car should form"
assert len(s_mid[0]) == 2, "medium radius: both cars, separately"
assert len(s_big[0]) == 1 and s_big[0][0][1] > 0 and s_big[0][0][2] > 0, "large radius: one merged blob"
fin = s_fin[0]
assert len(fin) == 2 and all((a == 0) != (b == 0) for _, a, b in fin), "final: two pure car clusters"

plt.rcParams["font.family"] = "Segoe UI"   # the font the deck renders in on this PC
fig = plt.figure(figsize=(16.8, 7.0), dpi=150)
fig.patch.set_facecolor("white")
# 5 panels with a spacer column on each side of step 2, where the separators go
gs = fig.add_gridspec(1, 7, width_ratios=[1.3, 0.12, 1, 1, 1, 0.12, 1.3], left=0.005, right=0.995,
                      top=0.80, bottom=0.27, wspace=0.06)
axes = [fig.add_subplot(gs[0, c]) for c in (0, 2, 3, 4, 6)]
SUBTITLES = [None, "Small radius", "Medium radius", "Large radius", None]
NOTES = ["Circle reaches each point's\n5th neighbour: small = dense,\nbig = sparse",
         "Near car forms;\nfar car is still loose points",
         "Both cars form,\nseparately",
         "Both cars fuse\ninto one blob",
         "Each car is kept whole:\nit holds together longer\nthan its pieces would"]
panels = [None, cuts[0.3], cuts[1.0], cuts[3.2], final]
for i, (ax, lab) in enumerate(zip(axes, panels)):
    if i == 0:
        ax.scatter(P[:, 0], P[:, 1], s=10, c=DARK, zorder=3)
        for j in [10, 40, len(near) + 5, len(near) + 16]:
            ax.add_patch(Circle(P[j], core[j], fill=False, lw=1.6, ec="#DC2626", zorder=2))
            ax.scatter(*P[j], s=26, c="#DC2626", zorder=4)
    else:
        ax.scatter(P[:, 0], P[:, 1], s=12, c=colour(lab), zorder=3)
        if i in (1, 2, 3):
            r = list(cuts)[i - 1]
            ax.plot([5.5 - r, 5.5], [8.8, 8.8], c=DARK, lw=3, solid_capstyle="butt")
            ax.text(5.5 - r / 2, 8.65, f"{r} m", ha="center", va="top", fontsize=15, color=DARK)
    ax.set_xlim(-1.4, 6.3)
    ax.set_ylim(-0.9, 9.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#D1D5DB")
    if SUBTITLES[i]:
        ax.set_title(SUBTITLES[i], fontsize=18, color="#52606D", pad=8)
    ax.text(0.5, -0.04, NOTES[i], transform=ax.transAxes, ha="center", va="top", fontsize=16,
            color=DARK, linespacing=1.35)

# give the two outer panels the middle panels' size and height, centred in their wider columns
for ax in axes:
    ax.apply_aspect()
ref = axes[2].get_position()
for ax in (axes[0], axes[4]):
    b = ax.get_position()
    ax.set_position([b.x0 + (b.width - ref.width) / 2, ref.y0, ref.width, ref.height])
# section headers, and a vertical rule between sections (centre of each spacer column)
pos = [ax.get_position() for ax in axes]
for a, b, title in [(0, 0, "1. Measure density"), (1, 3, "2. Try every radius"),
                    (4, 4, "3. What HDBSCAN keeps")]:
    fig.text((pos[a].x0 + pos[b].x1) / 2, 0.955, title, ha="center", va="top", fontsize=22,
             fontweight="bold", color=DARK)
for left, right in [(0, 1), (3, 4)]:
    x = (pos[left].x1 + pos[right].x0) / 2
    fig.add_artist(plt.Line2D([x, x], [0.13, 0.97], transform=fig.transFigure, color="#9AA5B1",
                              lw=2))
for k, name in [("near", "near car (dense)"), ("far", "far car (sparse)"),
                ("merged", "both cars as one")]:
    axes[0].scatter([], [], c=COL[k], s=60, label=name)
axes[0].scatter([], [], c=GREY, s=60, label="not in a group")
fig.legend(loc="lower center", ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
