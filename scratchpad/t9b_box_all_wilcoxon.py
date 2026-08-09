"""T9b helper: pooled (ALL-cars) Wilcoxon for the #29 box metric, raw vs completed.

length_1b_box_eval.py reports per-car medians + per-BAND Wilcoxon but not a
pooled ALL-cars Wilcoxon. The seq-00 held-out pre-registration
(docs/plans/preregistration_heldout.md) names BEV IoU (median, Wilcoxon p<.05)
as a primary refutation-bearing metric over all cars, so compute exactly that
here from the same recomputed donor pairs. No tuning; read-only over the cache.

Run:
    .venv\\Scripts\\python.exe scratchpad\\t9b_box_all_wilcoxon.py --seq 00 \\
        --dir output/experiments/donor_metric_00_lenon
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from completion_box_eval import world_box  # noqa: E402
from completion_box_eval_step2 import bev_iou, yaw_error_deg  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["signed_dL", "adL", "adW", "adH", "bev_iou", "yaw_err", "center_err"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    exp = args.dir if os.path.isabs(args.dir) else os.path.join(PROJECT_ROOT, args.dir)
    with open(os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")) as f:
        amodal = json.load(f)
    well = {int(k): r for k, r in amodal["instances"].items() if r.get("well_observed")}
    with open(os.path.join(exp, f"step1_index_{args.seq}.json")) as f:
        index = json.load(f)
    pairs = [p for p in index["pairs"] if p["skip_reason"] is None]

    by_car = {"raw": {}, "on": {}}
    for p in pairs:
        gt = well[p["inst_id"]]
        gtL, gtW, gtH = gt["dims_lwh"]
        gc = gt["center_world"]
        dd = np.load(os.path.join(exp, f"pairs_{args.seq}", p["file"]))
        T = dd["T"]
        for m, arr in [("raw", dd["raw"]), ("on", dd["completed"])]:
            b = world_box(arr.astype(np.float64) @ T[:3, :3].T + T[:3, 3])
            L, W, H = b["dims_lwh"]
            c = b["center_world"]
            by_car[m].setdefault(p["inst_id"], []).append({
                "signed_dL": L - gtL, "adL": abs(L - gtL),
                "adW": abs(W - gtW), "adH": abs(H - gtH),
                "bev_iou": bev_iou(b, gt),
                "yaw_err": yaw_error_deg(b["yaw_deg"], gt["yaw_deg"]),
                "center_err": float(np.hypot(c[0] - gc[0], c[2] - gc[2])),
            })

    cars = sorted(by_car["raw"])
    med = {m: {i: {k: float(np.median([r[k] for r in by_car[m][i]])) for k in METRICS}
               for i in cars} for m in ["raw", "on"]}
    print(f"ALL cars n={len(cars)} (completed=on vs raw), pooled per-car medians:")
    print(f"{'metric':>11} {'raw':>8} {'on':>8} {'wilcoxon p':>12}")
    for k in METRICS:
        va = np.array([med["raw"][i][k] for i in cars])
        vb = np.array([med["on"][i][k] for i in cars])
        p = float(wilcoxon(va, vb).pvalue) if not np.allclose(va, vb) else None
        ps = f"{p:.2e}" if p is not None else "n/a"
        print(f"{k:>11} {np.median(va):>8.3f} {np.median(vb):>8.3f} {ps:>12}")


if __name__ == "__main__":
    main()
