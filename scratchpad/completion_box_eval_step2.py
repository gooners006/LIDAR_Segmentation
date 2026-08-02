"""Direction 4a — Step 2: paired raw-vs-completed box metrics against amodal GT.

Reads Step 1's per-frame records (step1_records_<seq>.json) and the amodal GT
boxes (output/<seq>/amodal_gt.json), and computes, for every completed pair,
raw-box and completed-box errors against the GT box:

  - |dL|, |dW|, |dH| dimension errors (m) + signed biases
  - BEV oriented-box IoU (Sutherland-Hodgman rectangle clipping in X-Z)
  - yaw error mod 180, folded to [0, 90] (car symmetry)
  - center error in the ground plane (X-Z distance, m)

Statistics: the primary unit is the CAR (median over each car's completed
frames, then Wilcoxon signed-rank across cars) because frame-level pairs of
the same parked car are heavily autocorrelated. Pooled frame-level numbers are
reported as secondary. W is additionally split by the GT both_sides_seen flag
(amodal W is the least-constrained GT dim), and signed dL is split by GT
length to test the compact-car overshoot hypothesis.

Run:
    .venv\\Scripts\\python.exe scratchpad\\completion_box_eval_step2.py --seq 08
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# BEV oriented-box IoU
# --------------------------------------------------------------------------

def box_corners_xz(center_xz, yaw_deg, length, width) -> np.ndarray:
    """4 corners (CCW) of an oriented rectangle in the X-Z ground plane.
    yaw_deg = length-axis angle from +X toward +Z (amodal_gt convention)."""
    t = np.deg2rad(yaw_deg)
    e_len = np.array([np.cos(t), np.sin(t)])
    e_wid = np.array([-np.sin(t), np.cos(t)])
    c = np.asarray(center_xz, dtype=np.float64)
    hl, hw = length / 2.0, width / 2.0
    return np.array([c + hl * e_len + hw * e_wid,
                     c - hl * e_len + hw * e_wid,
                     c - hl * e_len - hw * e_wid,
                     c + hl * e_len - hw * e_wid])


def _clip_polygon(poly: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: keep the part of poly left of directed edge a->b."""
    edge = b - a
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        side_p = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        side_q = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
        if side_p >= 0:
            out.append(p)
        if (side_p >= 0) != (side_q >= 0):
            denom = side_p - side_q
            if abs(denom) > 1e-12:
                out.append(p + (q - p) * (side_p / denom))
    return np.array(out) if out else np.zeros((0, 2))


def _shoelace_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def bev_iou(box_a: dict, box_b: dict) -> float:
    """Oriented-rectangle IoU in X-Z between two box dicts
    ({yaw_deg, dims_lwh, center_world})."""
    ca = [box_a["center_world"][0], box_a["center_world"][2]]
    cb = [box_b["center_world"][0], box_b["center_world"][2]]
    pa = box_corners_xz(ca, box_a["yaw_deg"], *box_a["dims_lwh"][:2])
    pb = box_corners_xz(cb, box_b["yaw_deg"], *box_b["dims_lwh"][:2])
    inter = pa
    for i in range(4):
        inter = _clip_polygon(inter, pb[i], pb[(i + 1) % 4])
        if len(inter) == 0:
            return 0.0
    ai = _shoelace_area(inter)
    aa, ab = _shoelace_area(pa), _shoelace_area(pb)
    union = aa + ab - ai
    return ai / union if union > 0 else 0.0


def yaw_error_deg(a: float, b: float) -> float:
    """|a - b| mod 180, folded to [0, 90] (car front/back symmetry)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------
# Per-pair metrics
# --------------------------------------------------------------------------

METRICS = ["adL", "adW", "adH", "bev_iou", "yaw_err", "center_err"]
HIGHER_BETTER = {"bev_iou"}


def pair_metrics(box: dict, gt: dict) -> dict:
    dims = np.array(box["dims_lwh"])
    gt_dims = np.array(gt["dims_lwh"])
    d = dims - gt_dims
    cx = box["center_world"][0] - gt["center_world"][0]
    cz = box["center_world"][2] - gt["center_world"][2]
    return {
        "dL": float(d[0]), "dW": float(d[1]), "dH": float(d[2]),
        "adL": abs(float(d[0])), "adW": abs(float(d[1])), "adH": abs(float(d[2])),
        "bev_iou": bev_iou(box, gt),
        "yaw_err": yaw_error_deg(box["yaw_deg"], gt["yaw_deg"]),
        "center_err": float(np.hypot(cx, cz)),
    }


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def _fmt_table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(v).rjust(w) for v, w in zip(header, widths))]
    for r in rows:
        lines.append("  ".join(str(v).rjust(w) for v, w in zip(r, widths)))
    return "\n".join(lines)


def summarize(raw_vals: np.ndarray, comp_vals: np.ndarray, label: str,
              paired_test: bool):
    """Print raw-vs-completed summary for a (n_samples, n_metrics) array pair."""
    rows = []
    stats = {}
    for j, m in enumerate(METRICS):
        r, c = raw_vals[:, j], comp_vals[:, j]
        delta = (c - r) if m in HIGHER_BETTER else (r - c)  # >0 = completion better
        row = [m,
               f"{np.mean(r):.3f}", f"{np.median(r):.3f}",
               f"{np.mean(c):.3f}", f"{np.median(c):.3f}",
               f"{np.median(delta):+.3f}"]
        entry = {
            "raw_mean": float(np.mean(r)), "raw_median": float(np.median(r)),
            "comp_mean": float(np.mean(c)), "comp_median": float(np.median(c)),
            "median_improvement": float(np.median(delta)),
        }
        if paired_test and len(delta) >= 6 and np.any(delta != 0):
            stat_res = wilcoxon(delta)
            row.append(f"{stat_res.pvalue:.2g}")
            entry["wilcoxon_p"] = float(stat_res.pvalue)
        elif paired_test:
            row.append("-")
        rows.append(row)
        stats[m] = entry
    header = ["metric", "raw mean", "raw med", "comp mean", "comp med", "med improv"]
    if paired_test:
        header.append("p (wilcoxon)")
    print(f"\n-- {label} (n={len(raw_vals)}) --")
    print("   (improv > 0 means completion is better; units: m, deg, IoU)")
    print(_fmt_table(rows, header))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--records", default=None,
                    help="Step 1 JSON (default output/experiments/"
                         "completion_box_eval/step1_records_<seq>.json)")
    ap.add_argument("--out", default=None,
                    help="Metrics JSON (default alongside records)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rec_path = args.records or os.path.join(
        PROJECT_ROOT, "output", "experiments", "completion_box_eval",
        f"step1_records_{args.seq}.json")
    out_path = args.out or os.path.join(
        os.path.dirname(rec_path), f"step2_metrics_{args.seq}.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    with open(rec_path) as f:
        step1 = json.load(f)
    with open(os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")) as f:
        amodal = json.load(f)
    gt_by_inst = {int(k): r for k, r in amodal["instances"].items()
                  if r.get("well_observed")}

    # Sanity: GT box against itself must be a perfect match.
    any_gt = next(iter(gt_by_inst.values()))
    assert abs(bev_iou(any_gt, any_gt) - 1.0) < 1e-9, "self-IoU != 1"
    assert yaw_error_deg(179.0, 1.0) == 2.0, "yaw fold broken"

    records = step1["records"]
    completed = [r for r in records if r["skip_reason"] is None]
    print(f"Records: {len(records)} pairs, {len(completed)} completed "
          f"({len(records) - len(completed)} gated)")

    # Per-pair metrics.
    pairs = []
    for r in completed:
        gt = gt_by_inst[r["inst_id"]]
        pairs.append({
            "inst_id": r["inst_id"],
            "n_raw_pts": r["n_raw_pts"],
            "gt_L": gt["dims_lwh"][0],
            "both_sides_seen": gt["both_sides_seen"],
            "raw": pair_metrics(r["raw_box"], gt),
            "comp": pair_metrics(r["comp_box"], gt),
        })

    def arr(group, which):
        return np.array([[p[which][m] for m in METRICS] for p in group])

    out = {"records": rec_path, "n_pairs": len(records),
           "n_completed": len(completed)}

    # --- Primary: per-car medians, Wilcoxon across cars ---
    insts = sorted({p["inst_id"] for p in pairs})
    car_raw, car_comp = [], []
    for gi in insts:
        g = [p for p in pairs if p["inst_id"] == gi]
        car_raw.append(np.median(arr(g, "raw"), axis=0))
        car_comp.append(np.median(arr(g, "comp"), axis=0))
    car_raw, car_comp = np.array(car_raw), np.array(car_comp)
    out["per_car"] = summarize(car_raw, car_comp,
                               f"PRIMARY: per-car medians, {len(insts)} cars",
                               paired_test=True)

    # --- Secondary: pooled frame-level ---
    out["pooled"] = summarize(arr(pairs, "raw"), arr(pairs, "comp"),
                              "secondary: pooled frame-level pairs",
                              paired_test=False)

    # --- W split by GT both_sides_seen (amodal W trustworthiness) ---
    print("\n-- |dW| by GT width evidence (per-car medians) --")
    out["dW_by_sides"] = {}
    for flag in [True, False]:
        sel = [i for i, gi in enumerate(insts)
               if gt_by_inst[gi]["both_sides_seen"] == flag]
        if not sel:
            continue
        j = METRICS.index("adW")
        r, c = car_raw[sel, j], car_comp[sel, j]
        print(f"  both_sides_seen={flag!s:>5} (n={len(sel)} cars): "
              f"raw med {np.median(r):.3f}  comp med {np.median(c):.3f}")
        out["dW_by_sides"][str(flag)] = {
            "n_cars": len(sel),
            "raw_median": float(np.median(r)),
            "comp_median": float(np.median(c)),
        }

    # --- Signed dL by GT length: compact-overshoot hypothesis ---
    print("\n-- signed dL = completed L - GT L, by GT length (frame-level) --")
    out["dL_by_gt_length"] = {}
    for name, lo, hi in [("compact  (< 3.6 m)", 0.0, 3.6),
                         ("normal  (>= 3.6 m)", 3.6, 99.0)]:
        g = [p for p in pairs if lo <= p["gt_L"] < hi]
        if not g:
            continue
        raw_dL = np.array([p["raw"]["dL"] for p in g])
        comp_dL = np.array([p["comp"]["dL"] for p in g])
        n_cars = len({p["inst_id"] for p in g})
        print(f"  {name} ({n_cars} cars, {len(g)} pairs): "
              f"raw {np.mean(raw_dL):+.3f}  completed {np.mean(comp_dL):+.3f}")
        out["dL_by_gt_length"][name.split()[0]] = {
            "n_cars": n_cars, "n_pairs": len(g),
            "raw_mean_signed": float(np.mean(raw_dL)),
            "comp_mean_signed": float(np.mean(comp_dL)),
        }

    # --- By input density ---
    print("\n-- BEV IoU / |dL| by raw cluster size (frame-level means) --")
    out["by_density"] = {}
    for name, lo, hi in [("sparse  <100", 0, 100),
                         ("mid  100-300", 100, 300),
                         ("dense  >=300", 300, 10**9)]:
        g = [p for p in pairs if lo <= p["n_raw_pts"] < hi]
        if not g:
            continue
        ri = np.mean([p["raw"]["bev_iou"] for p in g])
        ci = np.mean([p["comp"]["bev_iou"] for p in g])
        rl = np.mean([p["raw"]["adL"] for p in g])
        cl = np.mean([p["comp"]["adL"] for p in g])
        print(f"  {name:>13} ({len(g):>4} pairs): IoU raw {ri:.3f} -> comp {ci:.3f}"
              f"   |dL| raw {rl:.3f} -> comp {cl:.3f}")
        out["by_density"][name.split()[0]] = {
            "n_pairs": len(g),
            "bev_iou_raw": float(ri), "bev_iou_comp": float(ci),
            "adL_raw": float(rl), "adL_comp": float(cl),
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
