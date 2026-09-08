"""Prior-art comparison: UCD / UHD vs the donor-frame coverage metric.

Motivation (session 2026-09-05 prior-art audit): the standard *reference-free*
completion metrics on real scans are the Unidirectional Chamfer Distance (UCD)
and Unidirectional Hausdorff Distance (UHD) -- Chen et al., ICLR 2020, use UHD on
KITTI cars; P2C (ICCV 2023) uses UCD. Both, and the region-aware RCD, anchor to the
*observed input* -- none scores genuinely unseen surface. This script measures UCD/UHD
on our own seq-08 completions to show, empirically, that they cannot credit
unseen-surface recovery, whereas the donor coverage metric does. It complements
Finding #26 (raw scores best under bidirectional Chamfer vs the accumulation) by
naming and computing the field's actual reference-free metrics.

Definitions (metres; direction = input -> prediction, the Chen/P2C convention):
  UCD_fwd(method) = mean_{x in input}   min_{y in method} ||x - y||   (also squared, m^2)
  UHD_fwd(method) = max / p95 of the same per-input-point NN distances
The reverse direction (method -> input) is computed only to illustrate that it
*penalises* added surface -- the perverse behaviour that motivates a novel-set metric.

FREEZE-SAFE / READ-ONLY: reuses the cached donor pairs and cloud-reconstruction of
donor_metric_step2.py; touches nothing under src/, checkpoints, or output/08.

Run:
    .venv\\Scripts\\python.exe scratchpad\\ucd_uhd_compare.py --seq 08
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the donor metric's per-car aggregation + Wilcoxon so columns describe the
# same cars/pairs under the same convention.
from donor_metric_step3 import per_car_median, paired_wilcoxon, MIN_NOVEL  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["raw", "mirrored", "completed"]
TAU_PRIMARY = "0.15"   # matches donor_metric_step2 primary visibility-mask tau


def nn_dists(query_pts: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """For each point in query_pts, Euclidean distance to nearest in ref_pts (m)."""
    d, _ = cKDTree(ref_pts).query(query_pts)
    return d


def directional(query_pts: np.ndarray, ref_pts: np.ndarray) -> dict:
    """UCD/UHD statistics for query -> ref (metres)."""
    d = nn_dists(query_pts, ref_pts)
    return {
        "ucd_mean": float(d.mean()),          # UCD (mean-L2, m)
        "ucd_sq": float((d ** 2).mean()),     # UCD (mean-squared, m^2)
        "uhd_max": float(d.max()),            # UHD (max-L2, m)
        "uhd_p95": float(np.percentile(d, 95)),  # UHD (95th-pct-L2, robust, m)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--dir", default=None,
                    help="Donor experiment dir (default output/experiments/donor_metric)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    exp_dir = args.dir or os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric")
    index_path = os.path.join(exp_dir, f"step1_index_{args.seq}.json")
    pairs_dir = os.path.join(exp_dir, f"pairs_{args.seq}")
    rec_path = os.path.join(exp_dir, f"donor_metric_records_{args.seq}.json")

    out_dir = os.path.join(PROJECT_ROOT, "output", "experiments", "ucd_uhd_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ucd_uhd_{args.seq}.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    with open(index_path) as f:
        index = json.load(f)
    with open(rec_path) as f:
        donor = json.load(f)
    # Join key -> donor record (per-pair cov@primary tau + n_novel for qualification).
    drec = {(r["inst_id"], r["frame"]): r for r in donor["records"]}
    pairs = [p for p in index["pairs"] if p["skip_reason"] is None]
    print(f"{len(pairs)} gate-passed pairs; {len(drec)} donor records")

    records = []
    t0 = time.time()
    for k, p in enumerate(pairs):
        inst, fi = p["inst_id"], p["frame"]
        dr = drec.get((inst, fi))
        if dr is None:
            continue  # no donor record (donor set empty) -> not comparable
        data = np.load(os.path.join(pairs_dir, p["file"]))
        raw = data["raw"].astype(np.float64)
        completed = data["completed"].astype(np.float64)
        basis, center = data["basis"], data["center"]
        T = data["T"]
        R, t = T[:3, :3], T[:3, 3]

        # World-frame clouds, reconstructed exactly as donor_metric_step2.py.
        raw_w = raw @ R.T + t
        pts_c = raw @ basis
        mir_c = pts_c.copy()
        mir_c[:, 0] = 2.0 * center[0] - pts_c[:, 0]
        mirrored_w = np.vstack([raw_w, (mir_c @ basis.T) @ R.T + t])
        comp_w = completed @ R.T + t
        method_clouds = {"raw": raw_w, "mirrored": mirrored_w, "completed": comp_w}

        # input == raw partial. Direction input -> method (Chen/P2C convention).
        fwd = {m: directional(raw_w, cloud) for m, cloud in method_clouds.items()}
        rev = {m: directional(cloud, raw_w) for m, cloud in method_clouds.items()}

        tau = dr["taus"][TAU_PRIMARY]
        records.append({
            "inst_id": inst, "frame": fi,
            "n_raw_pts": int(len(raw)),
            "n_novel": int(tau["n_novel"]),
            "fwd": fwd, "rev": rev,
            # side-by-side donor coverage @0.1 at primary tau (0 for raw by design)
            "donor_cov": {m: (tau[m]["cov"] if m in tau else None) for m in METHODS},
        })
        if (k + 1) % 200 == 0 or k == len(pairs) - 1:
            print(f"  pair {k + 1}/{len(pairs)}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- Qualify + aggregate per-car, matching donor_metric_step3 ------------
    qual = [r for r in records if r["n_novel"] >= MIN_NOVEL]
    by_car: dict[int, list] = {}
    for r in qual:
        by_car.setdefault(r["inst_id"], []).append(r)
    print(f"\n{len(qual)}/{len(records)} pairs qualified (>= {MIN_NOVEL} novel pts "
          f"@ tau={TAU_PRIMARY}); {len(by_car)} cars")

    def pooled(fn):
        pc = per_car_median(by_car, fn)
        return float(np.median(list(pc.values()))), pc

    summary = {}
    for m in METHODS:
        row = {}
        for stat in ("ucd_mean", "ucd_sq", "uhd_max", "uhd_p95"):
            row[f"fwd_{stat}"], _ = pooled(lambda p, m=m, s=stat: p["fwd"][m][s])
            row[f"rev_{stat}"], _ = pooled(lambda p, m=m, s=stat: p["rev"][m][s])
        row["donor_cov"], _ = pooled(
            lambda p, m=m: p["donor_cov"][m] if p["donor_cov"][m] is not None else None)
        summary[m] = row

    # Significance: raw vs completed on forward UCD (mean-L2), across cars.
    a = per_car_median(by_car, lambda p: p["fwd"]["raw"]["ucd_mean"])
    b = per_car_median(by_car, lambda p: p["fwd"]["completed"]["ucd_mean"])
    med_raw, med_comp, pval, ncars = paired_wilcoxon(a, b)

    payload = {
        "seq": args.seq,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "convention": {
            "units": "metres (ucd_sq in m^2)",
            "frame": "world (cam0), reconstructed as donor_metric_step2.py",
            "direction_fwd": "input(raw partial) -> method (Chen ICLR2020 / P2C ICCV2023)",
            "direction_rev": "method -> input (illustrative; penalises added surface)",
            "tau_primary": TAU_PRIMARY, "min_novel": MIN_NOVEL,
        },
        "n_pairs": len(records), "n_pairs_qualified": len(qual),
        "n_cars": len(by_car),
        "pooled_per_car_median": summary,
        "wilcoxon_fwd_ucd_mean_raw_vs_completed": {
            "median_raw": med_raw, "median_completed": med_comp,
            "p": pval, "n_cars": ncars},
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved {len(records)} records -> {out_path}\n")

    # ---- Summary table ------------------------------------------------------
    print("Pooled per-car medians (qualified cars). Distances in metres; "
          "direction = input -> method.")
    hdr = (f"{'method':>10} | {'UCD_fwd':>8} {'UCD_sq':>8} {'UHD_max':>8} "
           f"{'UHD_p95':>8} | {'UCD_rev':>8} | {'donor_cov@0.1':>13}")
    print(hdr)
    print("-" * len(hdr))
    for m in METHODS:
        s = summary[m]
        print(f"{m:>10} | {s['fwd_ucd_mean']:8.4f} {s['fwd_ucd_sq']:8.4f} "
              f"{s['fwd_uhd_max']:8.4f} {s['fwd_uhd_p95']:8.4f} | "
              f"{s['rev_ucd_mean']:8.4f} | {s['donor_cov']:13.4f}")
    print(f"\nWilcoxon fwd-UCD raw vs completed: median raw {med_raw:.4f} m, "
          f"completed {med_comp:.4f} m, p={pval}, n_cars={ncars}")
    print("\nReading: UCD/UHD (input->method) reward keeping the input; raw is the "
          "trivial best. Donor cov@0.1 credits recovered unseen surface, where "
          "completed wins. The standard reference-free metrics are input-fidelity "
          "measures and cannot rank unseen-surface recovery.")


if __name__ == "__main__":
    main()
