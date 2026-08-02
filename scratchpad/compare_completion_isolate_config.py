"""Compare published #29/#32 completion metrics vs the isolate-config re-run.

Isolate-config re-run = same classifier per experiment, only the PROMOTED
PIPELINE_CONFIG differs (voxel_before_denoise=True, ransac_iterations=300,
cluster_voxel_size=0.10). Answers: did the runtime-optimization config shift the
published completion findings (#29 amodal-box quality, #32 donor coverage)?

Reads:
  #29  output/experiments/completion_box_eval/step2_metrics_08.json       (published)
       output/experiments/completion_box_eval/step2_metrics_08_perf.json  (re-run)
  #32  output/experiments/donor_metric/donor_metric_summary_08.json       (published)
       output/experiments/donor_metric_perf/donor_metric_summary_08.json  (re-run)

Read-only; prints delta tables. Nothing written.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    with open(p) as f:
        return json.load(f)


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def compare_29():
    base_p = os.path.join(ROOT, "output/experiments/completion_box_eval/step2_metrics_08.json")
    perf_p = os.path.join(ROOT, "output/experiments/completion_box_eval/step2_metrics_08_perf.json")
    if not os.path.exists(perf_p):
        print(f"[#29] re-run file missing: {perf_p}\n")
        return
    b, n = load(base_p), load(perf_p)
    print("=" * 78)
    print("#29 completion_box_eval -- amodal-box quality (per_car), COMPLETED means")
    print(f"    n_pairs: published {b.get('n_pairs')}  ->  re-run {n.get('n_pairs')}")
    print("=" * 78)
    metrics = ["adL", "adW", "adH", "bev_iou", "yaw_err", "center_err"]
    print(f"{'metric':<10} {'base_raw':>9} {'base_comp':>10} | {'perf_raw':>9} {'perf_comp':>10} | {'d_comp':>8}")
    for m in metrics:
        bm, nm = b["per_car"][m], n["per_car"][m]
        d = nm["comp_mean"] - bm["comp_mean"]
        print(f"{m:<10} {bm['raw_mean']:>9.4f} {bm['comp_mean']:>10.4f} | "
              f"{nm['raw_mean']:>9.4f} {nm['comp_mean']:>10.4f} | {d:>+8.4f}")
    print("  (d_comp = re-run comp_mean - published comp_mean; sign per-metric:")
    print("   bev_iou higher=better; adL/adW/adH/center_err lower=better)\n")


def compare_32():
    base_p = os.path.join(ROOT, "output/experiments/donor_metric/donor_metric_summary_08.json")
    perf_p = os.path.join(ROOT, "output/experiments/donor_metric_perf/donor_metric_summary_08.json")
    if not os.path.exists(perf_p):
        print(f"[#32] re-run file missing: {perf_p}\n")
        return
    b, n = load(base_p), load(perf_p)
    print("=" * 78)
    print("#32 donor_metric -- donor coverage / med_dist by tau, COMPLETED")
    print(f"    n_cars: {b.get('n_cars')} -> {n.get('n_cars')}   "
          f"n_pairs_qualified: {b.get('n_pairs_qualified')} -> {n.get('n_pairs_qualified')}")
    print("=" * 78)
    taus = sorted(set(b["taus"]) & set(n["taus"]), key=float)
    print(f"{'tau':>5} | {'cov_base':>9} {'cov_perf':>9} {'d_cov':>8} | "
          f"{'md_base':>8} {'md_perf':>8} {'d_md':>8}")
    for t in taus:
        bc = b["taus"][t]["cov"]["completed"]
        nc = n["taus"][t]["cov"]["completed"]
        bmd = b["taus"][t]["med_dist"]["completed"]
        nmd = n["taus"][t]["med_dist"]["completed"]
        print(f"{t:>5} | {bc:>9.4f} {nc:>9.4f} {nc-bc:>+8.4f} | "
              f"{bmd:>8.4f} {nmd:>8.4f} {nmd-bmd:>+8.4f}")
    print("  (cov completed higher=better; med_dist completed lower=better)")
    print("  raw cov stays 0.0 by construction (occluded-side has no raw points)\n")


if __name__ == "__main__":
    compare_29()
    compare_32()
