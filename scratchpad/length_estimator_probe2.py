"""Direction 2 Step 1b — probe 2: fit the surviving signals.

Probe 1 (length_estimator_probe.py) killed the elegant ideas: GT L and W are
uncorrelated across the 40 amodal cars (r=+0.02, so an aspect-ratio prior can't
work even with a perfect width), height/range/density carry no signal, and the
far-end face-support test does NOT detect truncation (missing length is larger,
not smaller, when the far end looks well covered).

What survived: the observed footprint length itself, corr(gtL, fit_length)=+0.52.
This probe fits that signal properly -- shrinkage sweep, free linear fit, and
track-level quantiles (max was terrible, +0.95 m bias on compacts; a lower
quantile should denoise it) -- scoring MAE and, more importantly, PER-BAND BIAS,
since the whole point of Step 1b is to kill the compact over-extension without
giving back the normal-car gain.

Reads the cached records from probe 1; no cluster reload.

Run:
    .venv\\Scripts\\python.exe scratchpad\\length_estimator_probe2.py
"""

import argparse
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [("compact<3.6", lambda g: g < 3.6),
         ("normal", lambda g: (g >= 3.6) & (g < 4.6)),
         ("long>=4.6", lambda g: g >= 4.6)]


def report(name, est, gtL, masks, rows):
    err = est - gtL
    rows.append((name, float(np.mean(np.abs(err))), float(np.mean(err)),
                 [float(np.mean(err[m])) for m in masks]))


def show(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'estimator':>30} {'MAE':>7} {'bias':>7} " +
          " ".join(f"{n:>12}" for n, _ in BANDS))
    for name, mae, bias, bnd in rows:
        print(f"{name:>30} {mae:>7.3f} {bias:>+7.3f} " +
              " ".join(f"{b:>+12.3f}" for b in bnd))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default=os.path.join(
        PROJECT_ROOT, "output", "experiments", "len_probe", "probe_08.json"))
    args = ap.parse_args()

    with open(args.probe) as f:
        recs = json.load(f)
    gtL = np.array([r["gtL"] for r in recs])
    fitL = np.array([r["fit_length"] for r in recs])
    inst = np.array([r["inst"] for r in recs])
    masks = [f(gtL) for _, f in BANDS]
    print(f"{len(recs)} pairs, {len(set(inst.tolist()))} cars")

    # ---------------------------------------------------------------- track qs
    # Per-car quantiles of the observed footprint length, broadcast back to pairs.
    qs = {}
    for q in (50, 75, 90, 95, 100):
        v = np.empty(len(recs))
        for i in np.unique(inst):
            m = inst == i
            v[m] = np.percentile(fitL[m], q)
        qs[q] = v

    print("\n=== car-level: how well does a track quantile of fit_length "
          "predict that car's GT L? ===")
    cars = np.unique(inst)
    cg = np.array([gtL[inst == i][0] for i in cars])
    for q in (50, 75, 90, 95, 100):
        cq = np.array([qs[q][inst == i][0] for i in cars])
        # least-squares affine correction a + b*q, fitted on the same 40 cars
        b, a = np.polyfit(cq, cg, 1)
        pred = a + b * cq
        print(f"  q{q:<3d} corr={np.corrcoef(cq, cg)[0,1]:+.3f}  "
              f"raw bias={np.mean(cq - cg):+.3f}  MAE={np.mean(np.abs(cq - cg)):.3f}"
              f"   | affine a={a:+.2f} b={b:+.2f} -> MAE={np.mean(np.abs(pred - cg)):.3f}")

    # ------------------------------------------------------------- shrinkage
    rows = []
    report("const 4.14 (shipped)", np.full(len(recs), 4.14), gtL, masks, rows)
    report("fit_length (prior off)", fitL, gtL, masks, rows)
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        report(f"shrink a={alpha:.1f}",
               fitL + alpha * np.maximum(4.14 - fitL, 0.0), gtL, masks, rows)
    show(rows, "shrinkage toward the 4.14 prior: L = fitL + a*max(4.14-fitL,0)")

    # -------------------------------------------------------- free linear fit
    rows = []
    b1, a1 = np.polyfit(fitL, gtL, 1)
    report(f"OLS {a1:+.2f}{b1:+.2f}*fitL", a1 + b1 * fitL, gtL, masks, rows)
    # extend-only equivalent: the code applies max(L_est-obs,0)/2, so clamp
    report("OLS, extend-only", np.maximum(a1 + b1 * fitL, fitL), gtL, masks, rows)
    for lo, hi in ((3.2, 4.8), (3.4, 4.6), (3.6, 4.6)):
        report(f"OLS clipped [{lo},{hi}]",
               np.clip(a1 + b1 * fitL, lo, hi), gtL, masks, rows)
    show(rows, f"free linear on fit_length (OLS: L = {a1:.3f} + {b1:.3f}*fitL)")

    # ------------------------------------------------- track quantile variants
    rows = []
    report("const 4.14 (shipped)", np.full(len(recs), 4.14), gtL, masks, rows)
    for q in (75, 90, 95, 100):
        report(f"track q{q} raw", qs[q], gtL, masks, rows)
    for q in (75, 90):
        cq = np.array([qs[q][inst == i][0] for i in cars])
        b, a = np.polyfit(cq, np.array([gtL[inst == i][0] for i in cars]), 1)
        report(f"track q{q} affine", a + b * qs[q], gtL, masks, rows)
        report(f"track q{q} affine, extend-only",
               np.maximum(a + b * qs[q], fitL), gtL, masks, rows)
    # blend: per-frame observation floored by a denoised per-car estimate
    for q in (75, 90):
        report(f"max(fitL, track q{q})", np.maximum(fitL, qs[q]), gtL, masks, rows)
    show(rows, "track-level (multi-frame) estimators")

    # ------------------------------------------------ what is achievable at all
    print("\n=== ceiling check ===")
    print(f"  perfect per-car GT L        MAE 0.000")
    print(f"  best single-frame (OLS)     MAE "
          f"{np.mean(np.abs(a1 + b1 * fitL - gtL)):.3f}   "
          f"(corr {np.corrcoef(fitL, gtL)[0,1]:+.3f} caps this)")
    print(f"  per-car mean of fit_length  MAE "
          f"{np.mean(np.abs(qs[50] - gtL)):.3f}")
    print("\nRemember: the pipeline applies push = max(L_est - observed, 0)/2, so"
          "\nany estimator below the observed length is silently a no-op.")


if __name__ == "__main__":
    main()
