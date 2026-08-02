"""Direction 2 Step 1b — which observable predicts a car's TRUE length?

Step 1 shipped a fixed longitudinal prior (COMPLETION_CAR_LENGTH_PRIOR = 4.14 m,
Finding #35). It over-extends compacts (GT L < 3.6 m: signed dL -0.10 -> +0.25)
while fixing normal cars. Step 1b wants a PER-CAR length estimate instead.

Before spending PCN A/B runs on a guess, measure the estimators offline: for
every gate-passed donor pair we know the amodal-GT length of that car, so an
estimator can be scored directly as "how well does it predict GT L" (MAE / bias),
split by GT length band. No PCN inference needed -- this is pure geometry on the
cached raw clusters.

Candidate signals per pair (single-frame unless noted):
  fit_length   L-shape footprint length (what the prior currently compares to)
  fit_width    L-shape footprint width (complementary occluded dimension)
  obs_height   Z extent in the sensor frame (ground-cut, but not side-occluded)
  n_pts, rng   density / distance context
  far_span     width span of points near the ego-FAR end face, / fit_width.
               ~1 means the far end face is observed (car not truncated there);
               ~0 means the footprint tapers out = truncated far end.
  near_span    same at the ego-near end (control)
  track_max_L  max fit_length over all frames of this car (multi-frame; legal in
               production since main.py completes per track, but see caveat)

Run:
    .venv\\Scripts\\python.exe scratchpad\\length_estimator_probe.py
    .venv\\Scripts\\python.exe scratchpad\\length_estimator_probe.py \
        --dir output/experiments/donor_perf_lenoff --out output/experiments/len_probe
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from completion import PointCloudCompleter  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END_BAND = 0.30   # m: slab thickness at each end used for the face-support span


def end_spans(pts_c: np.ndarray, sign_z: float, fit_width: float) -> tuple:
    """Width span of the points hugging each end face, normalised by fit_width.

    A fully observed end shows a wall of points across the whole width; a
    truncated end tapers, so its slab covers only part of the width.
    """
    z = pts_c[:, 2]
    x = pts_c[:, 0]
    far_z = z.max() if sign_z > 0 else z.min()
    near_z = z.min() if sign_z > 0 else z.max()
    out = []
    for edge in (far_z, near_z):
        m = np.abs(z - edge) <= END_BAND
        out.append(float(np.ptp(x[m]) / fit_width) if m.sum() >= 3 and fit_width > 1e-6
                   else 0.0)
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--dir", default=os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_perf_lenoff"))
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output", "experiments", "len_probe"))
    args = ap.parse_args()

    src = args.dir if os.path.isabs(args.dir) else os.path.join(PROJECT_ROOT, args.dir)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")) as f:
        amodal = json.load(f)
    well = {int(k): r for k, r in amodal["instances"].items() if r.get("well_observed")}

    with open(os.path.join(src, f"step1_index_{args.seq}.json")) as f:
        index = json.load(f)
    pairs = [p for p in index["pairs"] if p["skip_reason"] is None]
    pairs_dir = os.path.join(src, f"pairs_{args.seq}")
    print(f"{len(pairs)} gate-passed pairs from {os.path.basename(src)}")

    # length_prior=None: we only want the gate + canonical frame, no length push
    # (the push moves `center` but never fit_length/fit_width, so this is only
    # about keeping `center[2]` the raw bbox centre for the sign test).
    completer = PointCloudCompleter(length_prior=None)

    recs = []
    for p in pairs:
        raw = np.load(os.path.join(pairs_dir, p["file"]))["raw"].astype(np.float64)
        frame, skip = completer.estimate_canonical_frame(raw)
        if skip is not None:
            continue
        basis, center = frame["basis"], frame["center"]
        pts_c = raw @ basis
        sign_z = np.sign(center[2]) if abs(center[2]) > 1e-9 else 1.0
        far_span, near_span = end_spans(pts_c, sign_z, frame["fit_width"])
        gtL, gtW, gtH = well[p["inst_id"]]["dims_lwh"]
        recs.append({
            "inst": p["inst_id"],
            "fit_length": frame["fit_length"],
            "fit_width": frame["fit_width"],
            "obs_height": float(np.ptp(raw[:, 2])),
            "n_pts": int(len(raw)),
            "rng": float(np.linalg.norm(raw[:, :2].mean(0))),
            "far_span": far_span,
            "near_span": near_span,
            "gtL": gtL, "gtW": gtW, "gtH": gtH,
        })

    with open(os.path.join(out_dir, f"probe_{args.seq}.json"), "w") as f:
        json.dump(recs, f)
    print(f"{len(recs)} records -> {out_dir}\n")

    # ---- per-car track_max_L (max observed footprint length over the car's frames)
    by_inst = {}
    for r in recs:
        by_inst.setdefault(r["inst"], []).append(r)
    for inst, rs in by_inst.items():
        tmax = max(r["fit_length"] for r in rs)
        for r in rs:
            r["track_max_L"] = tmax

    arr = {k: np.array([r[k] for r in recs], dtype=float)
           for k in ("fit_length", "fit_width", "obs_height", "n_pts", "rng",
                     "far_span", "near_span", "gtL", "gtW", "gtH", "track_max_L")}

    # ---- 1. Is there a per-car L<->W / L<->H relationship to exploit at all?
    print("=== GT dimension relationships across the 40 well-observed cars ===")
    cars = {r["inst"]: (r["gtL"], r["gtW"], r["gtH"]) for r in recs}
    cL = np.array([v[0] for v in cars.values()])
    cW = np.array([v[1] for v in cars.values()])
    cH = np.array([v[2] for v in cars.values()])
    print(f"n_cars={len(cars)}  corr(L,W)={np.corrcoef(cL, cW)[0,1]:+.3f}  "
          f"corr(L,H)={np.corrcoef(cL, cH)[0,1]:+.3f}")
    print(f"L/W ratio: median {np.median(cL/cW):.3f}  IQR "
          f"[{np.percentile(cL/cW,25):.3f},{np.percentile(cL/cW,75):.3f}]  "
          f"-> L from a PERFECT W would have MAE "
          f"{np.mean(np.abs(np.median(cL/cW)*cW - cL)):.3f} m\n")

    # ---- 2. Per-pair correlation of every observable with GT length
    print("=== per-pair correlation with GT length (n=%d) ===" % len(recs))
    for k in ("fit_length", "track_max_L", "fit_width", "obs_height", "n_pts",
              "rng", "far_span", "near_span"):
        print(f"  corr(gtL, {k:11s}) = {np.corrcoef(arr['gtL'], arr[k])[0,1]:+.3f}")

    # ---- 3. Does far_span actually detect truncation?
    print("\n=== far_span as a truncation detector ===")
    under = arr["gtL"] - arr["fit_length"]      # how much length is missing
    for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 2.0)):
        m = (arr["far_span"] >= lo) & (arr["far_span"] < hi)
        if m.sum() == 0:
            continue
        print(f"  far_span [{lo:.1f},{hi:.1f}): n={m.sum():4d}  "
              f"median missing length (gtL - fit_length) = {np.median(under[m]):+.3f} m  "
              f"median fit_length {np.median(arr['fit_length'][m]):.2f}")

    # ---- 4. Score candidate estimators of GT L, overall and by GT band
    print("\n=== candidate L_est estimators: error vs GT length ===")
    ratio = float(np.median(cL / cW))
    est = {
        "const_4.14 (shipped)": np.full(len(recs), 4.14),
        "fit_length (prior off)": arr["fit_length"],
        "track_max_L": arr["track_max_L"],
        "ratio*fit_width": np.clip(ratio * arr["fit_width"], 3.0, 5.0),
        "max(fit_len, ratio*fit_w)": np.clip(
            np.maximum(arr["fit_length"], ratio * arr["fit_width"]), 3.0, 5.0),
        "far_span-gated 4.14": np.where(arr["far_span"] >= 0.6,
                                        arr["fit_length"], 4.14),
        "shrink a=0.5": arr["fit_length"] + 0.5 * np.maximum(4.14 - arr["fit_length"], 0),
    }
    bands = [("compact <3.6", arr["gtL"] < 3.6),
             ("normal 3.6-4.6", (arr["gtL"] >= 3.6) & (arr["gtL"] < 4.6)),
             ("long >=4.6", arr["gtL"] >= 4.6)]
    hdr = f"{'estimator':>26} {'MAE':>7} {'bias':>7}"
    for name, _ in bands:
        hdr += f" | {name:>14}"
    print(hdr)
    for name, e in est.items():
        err = e - arr["gtL"]
        row = f"{name:>26} {np.mean(np.abs(err)):>7.3f} {np.mean(err):>+7.3f}"
        for _, m in bands:
            row += f" | {np.mean(err[m]):>+8.3f}({m.sum():4d})"
        print(row)
    print("\nbias > 0 = over-estimates true length (over-extends the completion)")


if __name__ == "__main__":
    main()
