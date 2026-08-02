"""Direction 2 Step 1b — box metric (#29) for per-car length estimates, BY BAND.

length_prior_box_recheck.py compared exactly two configs (prior off vs the fixed
4.14 m) on the pooled median. That pooled view is what hid the Step-1 defect: the
fixed prior is unbiased on mid-size cars while over-extending compacts (+0.63 m)
and under-extending long cars (-0.59 m), and the two cancel. This script scores
any number of variants and always splits by amodal-GT length band, because the
band split IS the question Step 1b is asking.

Bands follow the probe: compact < 3.6 m (8 cars), normal 3.6-4.6 (27), long >= 4.6
(5). Per-car medians first, then the median across the cars of a band, matching
#29/#32 house style. Wilcoxon is against the shipped fixed prior; the long band
(n=5) is reported for direction only, not significance.

Run:
    .venv\\Scripts\\python.exe scratchpad\\length_1b_box_eval.py \
        --variant off=output/experiments/donor_perf_lenoff \
        --variant on=output/experiments/donor_perf_lenon \
        --variant q95=output/experiments/donor_len1b_q95 \
        --variant q90off=output/experiments/donor_len1b_q90off \
        --variant ols=output/experiments/donor_len1b_ols
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
BANDS = [("compact<3.6", 0.0, 3.6), ("normal", 3.6, 4.6), ("long>=4.6", 4.6, 99.0)]
METRICS = ["signed_dL", "adL", "adW", "adH", "bev_iou", "yaw_err", "center_err"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--variant", action="append", required=True,
                    help="name=dir (repeatable); 'raw' is derived automatically")
    ap.add_argument("--ref", default="on", help="variant to test against")
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output", "experiments", "len_probe", "step1b_box_08.json"))
    args = ap.parse_args()

    variants = {}
    for v in args.variant:
        name, _, d = v.partition("=")
        variants[name] = d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)

    with open(os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")) as f:
        amodal = json.load(f)
    well = {int(k): r for k, r in amodal["instances"].items() if r.get("well_observed")}

    first = next(iter(variants.values()))
    with open(os.path.join(first, f"step1_index_{args.seq}.json")) as f:
        index = json.load(f)
    pairs = [p for p in index["pairs"] if p["skip_reason"] is None]
    print(f"{len(pairs)} gate-passed pairs, {len(variants)} variants "
          f"({', '.join(variants)})\n")

    # per-car lists of per-pair metric dicts, per method ("raw" + each variant)
    methods = ["raw"] + list(variants)
    by_car = {m: {} for m in methods}
    for p in pairs:
        inst = p["inst_id"]
        gt = well[inst]
        gtL, gtW, gtH = gt["dims_lwh"]
        clouds = {}
        for name, d in variants.items():
            dd = np.load(os.path.join(d, f"pairs_{args.seq}", p["file"]))
            if "raw" not in clouds:
                T = dd["T"]
                clouds["raw"] = (dd["raw"].astype(np.float64), T)
            clouds[name] = (dd["completed"].astype(np.float64), dd["T"])
        gt_box = {"yaw_deg": gt["yaw_deg"], "dims_lwh": gt["dims_lwh"],
                  "center_world": gt["center_world"]}
        for m in methods:
            pts, T = clouds[m]
            b = world_box(pts @ T[:3, :3].T + T[:3, 3])
            L, W, H = b["dims_lwh"]
            c, gc = b["center_world"], gt["center_world"]
            by_car[m].setdefault(inst, []).append({
                "signed_dL": L - gtL, "adL": abs(L - gtL),
                "adW": abs(W - gtW), "adH": abs(H - gtH),
                "bev_iou": bev_iou(b, gt_box),
                "yaw_err": yaw_error_deg(b["yaw_deg"], gt["yaw_deg"]),
                "center_err": float(np.hypot(c[0] - gc[0], c[2] - gc[2])),
            })

    # per-car medians
    med = {m: {i: {k: float(np.median([r[k] for r in rs])) for k in METRICS}
               for i, rs in by_car[m].items()} for m in methods}
    gtL_of = {i: well[i]["dims_lwh"][0] for i in med["raw"]}

    results = {}
    for bname, lo, hi in BANDS:
        cars = sorted(i for i, L in gtL_of.items() if lo <= L < hi)
        print(f"--- {bname}  (n={len(cars)} cars)")
        print(f"{'metric':>10} " + " ".join(f"{m:>9}" for m in methods))
        for k in METRICS:
            vals = {m: np.array([med[m][i][k] for i in cars]) for m in methods}
            row = " ".join(f"{np.median(vals[m]):>9.3f}" for m in methods)
            print(f"{k:>10} {row}")
            results.setdefault(bname, {})[k] = {
                m: float(np.median(vals[m])) for m in methods}
            if args.ref in methods and len(cars) >= 6:
                ps = []
                for m in methods:
                    if m == args.ref or np.allclose(vals[m], vals[args.ref]):
                        ps.append("     -   ")
                    else:
                        ps.append(f"{wilcoxon(vals[m], vals[args.ref]).pvalue:>9.1e}")
                print(f"{'  vs ' + args.ref:>10} " + " ".join(ps))
        print()

    # pooled, for continuity with the Step-1 tables
    print(f"--- ALL (n={len(gtL_of)} cars)")
    print(f"{'metric':>10} " + " ".join(f"{m:>9}" for m in methods))
    for k in METRICS:
        vals = {m: np.array([med[m][i][k] for i in sorted(gtL_of)]) for m in methods}
        print(f"{k:>10} " + " ".join(f"{np.median(vals[m]):>9.3f}" for m in methods))
        results.setdefault("ALL", {})[k] = {m: float(np.median(vals[m])) for m in methods}

    with open(args.out, "w") as f:
        json.dump({"bands": results, "variants": {k: os.path.basename(v)
                                                  for k, v in variants.items()}}, f,
                  indent=1)
    print(f"\n-> {args.out}")
    print("signed_dL < 0 = completed box SHORTER than amodal GT (under-extended)")


if __name__ == "__main__":
    main()
