"""Direction 2 — downstream box re-check for the length prior (#29 signed dL).

Finding #29 logged length UNDER-completion as a Direction-2 target: the completed
box was actually shorter than the raw partial (signed dL raw -0.49 -> completed
-0.55 m), i.e. the far end was never extended. The donor metric shows the length
prior fixes far-end coverage; this confirms it also fixes the tangible box
metric — completed length error vs the amodal GT.

Reuses the donor-metric cached clouds (raw + recomputed completed, world
transform T), so no detection sweep: fit the same oriented box (fit_oriented_box_xz,
minmax) the #29 pipeline used, to raw, completed(prior off), completed(prior on),
and compare dims to the amodal GT. Per-car medians + Wilcoxon, as in #29.

Run:
    .venv\\Scripts\\python.exe scratchpad\\length_prior_box_recheck.py \
        --off output/experiments/donor_metric_len_off \
        --on  output/experiments/donor_metric_len414
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_car_median(pairs_by_car, fn):
    out = {}
    for inst, prs in pairs_by_car.items():
        vals = [fn(p) for p in prs]
        vals = [v for v in vals if v is not None]
        if vals:
            out[inst] = float(np.median(vals))
    return out


def wilc(a, b):
    keys = sorted(set(a) & set(b))
    va = np.array([a[k] for k in keys]); vb = np.array([b[k] for k in keys])
    if len(keys) < 5 or np.allclose(va, vb):
        return float(np.median(va)), float(np.median(vb)), None, len(keys)
    return (float(np.median(va)), float(np.median(vb)),
            float(wilcoxon(va, vb).pvalue), len(keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--off", default=os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric_len_off"))
    ap.add_argument("--on", default=os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric_len414"))
    args = ap.parse_args()

    amodal_path = os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")
    with open(amodal_path) as f:
        amodal = json.load(f)
    well = {int(k): r for k, r in amodal["instances"].items()
            if r.get("well_observed")}

    with open(os.path.join(args.off, f"step1_index_{args.seq}.json")) as f:
        index = json.load(f)
    pairs = [p for p in index["pairs"] if p["skip_reason"] is None]
    print(f"{len(pairs)} gate-passed pairs")

    off_pairs = os.path.join(args.off, f"pairs_{args.seq}")
    on_pairs = os.path.join(args.on, f"pairs_{args.seq}")

    by_car = {}
    for p in pairs:
        inst = p["inst_id"]
        gt = well[inst]
        gtL, gtW, gtH = gt["dims_lwh"]
        d_off = np.load(os.path.join(off_pairs, p["file"]))
        d_on = np.load(os.path.join(on_pairs, p["file"]))
        T = d_off["T"]; R, t = T[:3, :3], T[:3, 3]

        raw_b = world_box(d_off["raw"].astype(np.float64) @ R.T + t)
        off_b = world_box(d_off["completed"].astype(np.float64) @ R.T + t)
        on_b = world_box(d_on["completed"].astype(np.float64) @ R.T + t)

        rec = {}
        for name, b in [("raw", raw_b), ("off", off_b), ("on", on_b)]:
            L, W, H = b["dims_lwh"]
            rec[name] = {
                "signed_dL": L - gtL, "adL": abs(L - gtL),
                "adW": abs(W - gtW), "adH": abs(H - gtH),
            }
        by_car.setdefault(inst, []).append(rec)

    print(f"{len(by_car)} cars\n")
    methods = ["raw", "off", "on"]
    print(f"{'metric':>10} " + " ".join(f"{m:>9}" for m in methods))
    for key in ["signed_dL", "adL", "adW", "adH"]:
        med = {m: per_car_median(by_car, lambda p, m=m, k=key: p[m][k])
               for m in methods}
        row = {m: float(np.median(list(med[m].values()))) for m in methods}
        print(f"{key:>10} " + " ".join(f"{row[m]:>9.3f}" for m in methods))
        # key contrasts: on vs off (does the prior help), on vs raw
        for a, b in [("on", "off"), ("on", "raw")]:
            ma, mb, pp, n = wilc(med[a], med[b])
            tag = "p=" + (f"{pp:.2e}" if pp is not None else "n/a")
            print(f"           {a} vs {b}: {ma:.3f} vs {mb:.3f}  {tag} (n={n})")
    print("\nsigned_dL<0 = box shorter than amodal GT (under-extended far end)")
    print("Baseline #29 (fine-tuned ckpt, full sweep): signed dL raw -0.49 -> completed -0.55")


if __name__ == "__main__":
    main()
