"""Direction 1 — Step 3: per-car statistics + validity checks for the donor metric.

Reads Step 2's records and produces the numbers that decide whether the
donor-frame occluded-side metric is valid (the Direction-1 deliverable):

  - per-car medians (frames of a parked car are autocorrelated, as in
    Finding #29) of coverage@0.1 and median novel-distance per method,
    Wilcoxon signed-rank across cars for the three method contrasts;
  - validation gate: (a) raw ranks last, (b) per-car stability (IQR across a
    car's pairs), (c) ranking unchanged across tau in {0.10, 0.15, 0.20},
    (d) hallucination guard (out-of-GT-box fraction) small for completed and
    not worse than mirrored;
  - diagnostics for Direction 2: input-size bins, novel-region breakdown
    (far side / far end / top), both_sides_seen split, symmetry self-CD.

Run:
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_step3.py --seq 08
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["raw", "mirrored", "completed"]
MIN_NOVEL = 100          # pair qualification: novel reference points at primary tau
SIZE_BINS = [(0, 100), (100, 300), (300, 10**9)]
# Amodal-GT length bands for the per-band hallucination guard (gate d2). Cut at
# 3.6 / 4.6 m: seq-08's 40 well-observed cars split 8 / 27 / 5.
LENGTH_BANDS = [("compact<3.6", 0.0, 3.6), ("normal", 3.6, 4.6),
                ("long>=4.6", 4.6, 99.0)]


def per_car_median(pairs_by_car: dict, fn) -> dict:
    """Median of fn(pair) over each car's qualified pairs."""
    out = {}
    for inst, prs in pairs_by_car.items():
        vals = [fn(p) for p in prs]
        vals = [v for v in vals if v is not None]
        if vals:
            out[inst] = float(np.median(vals))
    return out


def paired_wilcoxon(a: dict, b: dict):
    """(median_a, median_b, p) over cars present in both dicts."""
    keys = sorted(set(a) & set(b))
    va = np.array([a[k] for k in keys])
    vb = np.array([b[k] for k in keys])
    if len(keys) < 5 or np.allclose(va, vb):
        return float(np.median(va)), float(np.median(vb)), None, len(keys)
    p = float(wilcoxon(va, vb).pvalue)
    return float(np.median(va)), float(np.median(vb)), p, len(keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    exp_dir = args.dir or os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric")
    rec_path = os.path.join(exp_dir, f"donor_metric_records_{args.seq}.json")
    out_path = os.path.join(exp_dir, f"donor_metric_summary_{args.seq}.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    with open(rec_path) as f:
        payload = json.load(f)
    taus = [f"{t:.2f}" for t in payload["config"]["taus"]]
    tp = f"{payload['config']['tau_primary']:.2f}"

    amodal_path = os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")
    with open(amodal_path) as f:
        amodal = json.load(f)
    both_sides = {int(k) for k, r in amodal["instances"].items()
                  if r.get("well_observed") and r.get("both_sides_seen")}

    records = payload["records"]
    qual = [r for r in records if r["taus"][tp]["n_novel"] >= MIN_NOVEL]
    by_car: dict[int, list] = {}
    for r in qual:
        by_car.setdefault(r["inst_id"], []).append(r)
    print(f"{len(qual)}/{len(records)} pairs qualified (>= {MIN_NOVEL} novel pts "
          f"@ tau={tp}); {len(by_car)} cars")

    summary = {"seq": args.seq, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_pairs_qualified": len(qual), "n_cars": len(by_car),
               "min_novel": MIN_NOVEL}

    # ---- Headline: per-car medians + Wilcoxon at each tau -------------------
    print("\n==== Per-car medians (n cars) + Wilcoxon ====")
    summary["taus"] = {}
    for tau in taus:
        cov = {m: per_car_median(by_car, lambda p, m=m: p["taus"][tau][m]["cov"])
               for m in METHODS}
        med = {m: per_car_median(by_car,
                                 lambda p, m=m: p["taus"][tau][m]["med_dist"])
               for m in METHODS}
        line = {"cov": {m: float(np.median(list(cov[m].values())))
                        for m in METHODS},
                "med_dist": {m: float(np.median(list(med[m].values())))
                             for m in METHODS},
                "wilcoxon": {}}
        for a, b in [("completed", "raw"), ("completed", "mirrored"),
                     ("mirrored", "raw")]:
            for metric, d in [("cov", cov), ("med_dist", med)]:
                ma, mb, p, n = paired_wilcoxon(d[a], d[b])
                line["wilcoxon"][f"{a}_vs_{b}_{metric}"] = {
                    "a": round(ma, 4), "b": round(mb, 4),
                    "p": (round(p, 6) if p is not None else None), "n": n}
        summary["taus"][tau] = line
        flag = " (primary)" if tau == tp else ""
        print(f"\n-- tau = {tau}{flag}")
        print(f"{'method':>10} {'cov@0.1':>8} {'med dist (m)':>13}")
        for m in METHODS:
            print(f"{m:>10} {line['cov'][m]:>8.3f} {line['med_dist'][m]:>13.3f}")
        for name, w in line["wilcoxon"].items():
            if "cov" in name:
                print(f"  {name}: {w['a']:.3f} vs {w['b']:.3f}  "
                      f"p={w['p']}  (n={w['n']})")

    # ---- Validation gate ----------------------------------------------------
    print("\n==== Validation gate ====")
    gate = {}
    prim = summary["taus"][tp]
    # (a) raw ranks last on coverage at every tau
    gate["a_raw_last"] = all(
        summary["taus"][t]["cov"]["raw"] <= min(
            summary["taus"][t]["cov"]["mirrored"],
            summary["taus"][t]["cov"]["completed"]) for t in taus)
    # (b) stability: median per-car IQR of completed cov across pairs
    iqrs = [float(np.subtract(*np.percentile(
        [p["taus"][tp]["completed"]["cov"] for p in prs], [75, 25])))
        for prs in by_car.values() if len(prs) >= 3]
    gate["b_median_car_iqr_completed_cov"] = round(float(np.median(iqrs)), 4)
    # (c) method ranking on coverage identical across taus
    rank = lambda t: tuple(sorted(METHODS, key=lambda m: summary["taus"][t]["cov"][m]))
    gate["c_ranking_stable_across_tau"] = len({rank(t) for t in taus}) == 1
    gate["c_rankings"] = {t: list(rank(t)) for t in taus}
    # (d) hallucination guard
    oob = {m: per_car_median(by_car, lambda p, m=m: p["out_of_box"][m])
           for m in METHODS}
    gate["d_out_of_box_car_median"] = {
        m: round(float(np.median(list(oob[m].values()))), 4) for m in METHODS}
    gate["d_completed_not_worse_than_mirrored"] = (
        gate["d_out_of_box_car_median"]["completed"]
        <= gate["d_out_of_box_car_median"]["mirrored"])
    # (d2) SAME guard split by GT car length. The pooled median above is taken
    # over all cars, so a hallucination failure confined to one size band is
    # invisible: the fixed 4.14 m length prior scores 0.0020 pooled but 0.0433 on
    # compacts (< 3.6 m, 8/40 cars) — 4x the mirrored baseline it is checked
    # against. Any geometry change that pushes the completion outward has to
    # clear this per band, not just pooled (Step 1b, Finding #37).
    gt_len = {int(k): r["dims_lwh"][0] for k, r in amodal["instances"].items()
              if r.get("well_observed")}
    gate["d2_out_of_box_by_gt_length_band"] = {}
    gate["d2_bands_passing"] = {}
    for bname, lo, hi in LENGTH_BANDS:
        cars = [i for i in by_car if lo <= gt_len.get(i, -1) < hi]
        if not cars:
            continue
        band = {m: round(float(np.median([oob[m][i] for i in cars])), 4)
                for m in METHODS}
        band["n_cars"] = len(cars)
        gate["d2_out_of_box_by_gt_length_band"][bname] = band
        gate["d2_bands_passing"][bname] = band["completed"] <= band["mirrored"]
    gate["d2_all_bands_pass"] = all(gate["d2_bands_passing"].values())
    summary["gate"] = gate
    for k, v in gate.items():
        print(f"  {k}: {v}")

    # ---- Diagnostics ---------------------------------------------------------
    print("\n==== Input-size bins (pooled pair medians, primary tau) ====")
    summary["size_bins"] = {}
    for lo, hi in SIZE_BINS:
        sel = [r for r in qual if lo <= r["n_raw_pts"] < hi]
        if not sel:
            continue
        entry = {"n_pairs": len(sel)}
        for m in METHODS:
            entry[m] = round(float(np.median(
                [r["taus"][tp][m]["cov"] for r in sel])), 4)
        summary["size_bins"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = entry
        print(f"  [{lo:>3},{hi if hi < 10**9 else 'inf':>4}) n={len(sel):>4}: "
              + "  ".join(f"{m} {entry[m]:.3f}" for m in METHODS))

    print("\n==== Novel-region coverage (per-car medians, primary tau) ====")
    summary["regions"] = {}
    for region in ["far_side", "far_end", "top"]:
        d = {m: per_car_median(
            by_car, lambda p, m=m, rg=region:
            p.get("regions", {}).get(rg, {}).get(m)) for m in METHODS}
        n_cars = len(d["completed"])
        if n_cars < 5:
            continue
        entry = {"n_cars": n_cars,
                 **{m: round(float(np.median(list(d[m].values()))), 4)
                    for m in METHODS if d[m]}}
        _, _, p, n = paired_wilcoxon(d["completed"], d["mirrored"])
        entry["completed_vs_mirrored_p"] = round(p, 6) if p is not None else None
        summary["regions"][region] = entry
        print(f"  {region:>9} (n={n_cars} cars): "
              + "  ".join(f"{m} {entry.get(m, float('nan')):.3f}"
                          for m in METHODS)
              + f"   comp-vs-mir p={entry['completed_vs_mirrored_p']}")

    print("\n==== both_sides_seen split (far_side region, per-car medians) ====")
    summary["both_sides_split"] = {}
    for label, cars in [("both_sides_seen", both_sides),
                        ("one_side_only", set(by_car) - both_sides)]:
        sub = {i: prs for i, prs in by_car.items() if i in cars}
        if len(sub) < 3:
            continue
        d = {m: per_car_median(
            sub, lambda p, m=m: p.get("regions", {}).get("far_side", {}).get(m))
            for m in METHODS}
        if not d["completed"]:
            continue
        entry = {"n_cars": len(d["completed"]),
                 **{m: round(float(np.median(list(d[m].values()))), 4)
                    for m in METHODS if d[m]}}
        summary["both_sides_split"][label] = entry
        print(f"  {label:>16} (n={entry['n_cars']}): "
              + "  ".join(f"{m} {entry.get(m, float('nan')):.3f}"
                          for m in METHODS))

    sym = per_car_median(by_car, lambda p: p["sym_self_cd"])
    summary["sym_self_cd_car_median"] = round(float(np.median(list(sym.values()))), 4)
    print(f"\nSymmetry self-CD (completed, car median): "
          f"{summary['sym_self_cd_car_median']} m")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
