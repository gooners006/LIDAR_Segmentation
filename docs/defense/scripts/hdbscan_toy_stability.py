"""Print the condensed tree and per-cluster stability of the toy HDBSCAN example in
make_hdbscan_explainer.py (the numbers quoted in slide 55 notes and S2 notes A4).

Usage: .venv/Scripts/python.exe docs/defense/scripts/hdbscan_toy_stability.py
"""
import sys
from pathlib import Path
import numpy as np
src = (Path(__file__).parent / "make_hdbscan_explainer.py").open(encoding="utf-8").read()
exec(src[:src.index("plt.rcParams")])        # data + hdbscan fit only, no figure
ct = cl.condensed_tree_.to_pandas()
root = ct.parent.min()
kids = ct[(ct.child_size > 1)]
print("cluster nodes (parent -> child, born at lambda, radius, size):")
for _, r in kids.iterrows():
    print(f"  {int(r.parent)} -> {int(r.child)}  lambda={r.lambda_val:.3f}  radius={1/r.lambda_val:.2f} m  size={int(r.child_size)}")
# stability per cluster = sum over points (lambda_leave - lambda_birth)
birth = {root: 0.0}
for _, r in kids.iterrows():
    birth[int(r.child)] = r.lambda_val
stab = {}
for c in birth:
    rows = ct[ct.parent == c]
    stab[c] = float(((rows.lambda_val - birth[c]) * rows.child_size).sum())
for c, s in stab.items():
    rows = ct[(ct.parent == c) & (ct.child_size == 1)]
    pts = rows.lambda_val
    print(f"cluster {c}: born lambda {birth[c]:.3f} (radius {1/birth[c] if birth[c] else float('inf'):.2f} m), "
          f"points leave at radius median {np.median(1/pts) if len(pts) else float('nan'):.2f} m, "
          f"stability {s:.1f}")
print("selected:", sorted(set(cl.labels_) - {-1}), "sizes", [int((cl.labels_ == k).sum()) for k in sorted(set(cl.labels_) - {-1})])
