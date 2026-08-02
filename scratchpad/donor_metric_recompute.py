"""Direction 2 — recompute donor-metric completed clouds under a new complete().

The donor metric's Step 1 caches, per TP pair, the raw single-frame cluster and
the completed cloud (+ canonical basis/center/radius). Step 2 reads only those
caches. To A/B a `complete()` geometry change (here: the longitudinal length
prior, #32 far-end target) we do NOT re-run the ~2k-frame detection sweep — the
raw clusters and the input gate are unchanged. We reload each cached raw cluster
and re-run only `complete()` / `estimate_canonical_frame()` with the new
geometry, writing a fresh pairs dir that Step 2/3 consume via --dir.

Determinism / clean pairing: the completer RNG is reset per pair, so the
256-point subsample is identical regardless of the length prior — the ONLY thing
that differs between an off-prior and on-prior run is the length push. Run the
script once with no --length-prior (baseline) and once per prior value, then
compare with donor_metric_step2/step3 --dir. `center[0]`/`basis` are untouched by
the Z push, so Step 2's raw and mirrored baselines are byte-identical across runs.

Step 1b adds --length-mode: instead of one constant for every car, derive a
per-car length from the footprint fits the Step-1 index already caches
(`fit_length` per pair), so no cluster reload is needed for the aggregation
either. Modes:
    const        the shipped fixed prior (--length-prior, default 4.14)
    track-q95    per-car 95th percentile of fit_length over gate-passed frames
    track-q90off per-car 90th percentile + 0.12 m bias correction
    ols          single-frame 2.528 + 0.428*fit_length (leakage-free control)
Track modes fall back to the fixed prior on tracks with < 5 gate-passed frames.

CAVEAT (track modes): the per-car quantile is taken over the car's other frames,
which are also the donor metric's reference set — a single scalar (length) leaks
from reference to method. It is legitimate in production (main.py completes each
track once with all its frames in scope) but it means the track-mode far_end
number is "single-frame completion + track-level size prior", not pure
single-frame inference. The `ols` mode is the leakage-free control for exactly
this reason. Note its 2 coefficients are themselves fitted on these 40 cars.

Run:
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_recompute.py \
        --out-dir output/experiments/donor_metric_len_off              # baseline
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_recompute.py \
        --length-prior 4.14 --out-dir output/experiments/donor_metric_len414
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_recompute.py \
        --src-dir output/experiments/donor_perf_lenoff --length-mode track-q95 \
        --out-dir output/experiments/donor_len1b_q95
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from completion import PointCloudCompleter, track_length_estimate  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_SEED = 0   # reset before every complete() so the subsample is prior-invariant

# Single-frame OLS of GT length on observed fit_length, fitted over the 40
# amodal-GT cars (scratchpad/length_estimator_probe2.py). Leakage-free control.
OLS_A, OLS_B = 2.528, 0.428


def per_car_length(pairs, mode: str, fallback: float,
                   offset_override: float = None) -> dict:
    """Per-instance length estimate (m) from the cached Step-1 footprint fits."""
    by_inst = {}
    for p in pairs:
        if p["skip_reason"] is None and p.get("fit_length") is not None:
            by_inst.setdefault(p["inst_id"], []).append(p["fit_length"])
    if mode == "track-q95":
        q, off = 95, 0.0
    elif mode == "track-q90off":
        q, off = 90, 0.12
    else:
        raise ValueError(mode)
    if offset_override is not None:
        off = offset_override
    return {i: track_length_estimate(v, quantile=q, offset=off, fallback=fallback)
            for i, v in by_inst.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--length-prior", type=float, default=None,
                    help="Length prior in m (omit = disabled, i.e. baseline)")
    ap.add_argument("--length-mode", default="const",
                    choices=["const", "track-q95", "track-q90off", "ols"],
                    help="const uses --length-prior for every car; the others "
                         "derive a per-car estimate (Step 1b)")
    ap.add_argument("--fallback-prior", type=float, default=4.14,
                    help="Length used by track modes on tracks with < 5 frames")
    ap.add_argument("--track-offset", type=float, default=None,
                    help="Override the track-mode bias offset (m). Used to probe "
                         "whether donor coverage simply rewards over-extension.")
    ap.add_argument("--src-dir", default=None,
                    help="Source donor_metric dir with the cached Step-1 pairs")
    ap.add_argument("--out-dir", required=True,
                    help="Destination dir for the recomputed pairs + index")
    ap.add_argument("--pcn-ckpt", default=os.path.join(
        PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth"))
    ap.add_argument("--seed", type=int, default=0,
                    help="Per-pair subsample seed (replaces the old PAIR_SEED "
                         "constant; the RNG is reset to it before EVERY "
                         "complete(), which is what makes an A/B differ only by "
                         "the length push). PCN itself is deterministic, but "
                         "_fix_size() draws PCN_N_INPUT (256) points with this "
                         "RNG, so the seed selects WHICH points the network "
                         "sees. Default 0 = every published run. Vary it to "
                         "size the sampling spread of a metric.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = args.src_dir or os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric")
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(PROJECT_ROOT, args.out_dir))
    src_index = os.path.join(src_dir, f"step1_index_{args.seq}.json")
    src_pairs = os.path.join(src_dir, f"pairs_{args.seq}")
    out_index = os.path.join(out_dir, f"step1_index_{args.seq}.json")
    out_pairs = os.path.join(out_dir, f"pairs_{args.seq}")
    if os.path.exists(out_index) and not args.overwrite:
        raise SystemExit(f"{out_index} exists; pass --overwrite to replace it.")
    os.makedirs(out_pairs, exist_ok=True)

    with open(src_index) as f:
        index = json.load(f)

    completer = PointCloudCompleter(model_path=args.pcn_ckpt,
                                    seed=args.seed,
                                    length_prior=args.length_prior)
    pairs = index["pairs"]

    # Per-car estimates for the track modes; `ols` is computed per pair below.
    car_len = {}
    if args.length_mode.startswith("track-"):
        car_len = per_car_length(pairs, args.length_mode, args.fallback_prior,
                                 args.track_offset)
        vals = np.array(list(car_len.values()))
        print(f"length_mode = {args.length_mode}: {len(car_len)} cars, "
              f"L_est median {np.median(vals):.2f} "
              f"range [{vals.min():.2f}, {vals.max():.2f}]")
    elif args.length_mode == "ols":
        print(f"length_mode = ols: L_est = {OLS_A} + {OLS_B}*fit_length (per frame)")
    else:
        print(f"length_prior = {args.length_prior}  "
              f"({'DISABLED (baseline)' if args.length_prior is None else 'ON'})")

    n_comp = 0
    t0 = time.time()
    for k, p in enumerate(pairs):
        data = np.load(os.path.join(src_pairs, p["file"]))
        raw = data["raw"].astype(np.float64)
        arrays = {"raw": raw.astype(np.float32), "T": data["T"]}

        if p["skip_reason"] is None:
            if args.length_mode.startswith("track-"):
                l_est = car_len.get(p["inst_id"])
            elif args.length_mode == "ols":
                l_est = OLS_A + OLS_B * p["fit_length"]
            else:
                l_est = None          # fall through to completer.length_prior
            p["length_estimate"] = l_est

            # Gate runs before the length push -> skip decision must be unchanged.
            completer._rng = np.random.default_rng(args.seed)
            frame_est, frame_skip = completer.estimate_canonical_frame(raw, l_est)
            completer._rng = np.random.default_rng(args.seed)
            completed, skip = completer.complete(raw, "car", l_est)
            assert skip is None and frame_skip is None, (
                f"pair {p['file']} was gate-passed in baseline but now skips: "
                f"{skip!r} — the length prior must not change the gate")
            arrays.update({
                "completed": completed.astype(np.float32),
                "basis": frame_est["basis"].astype(np.float64),
                "center": frame_est["center"].astype(np.float64),
                "radius": np.float64(frame_est["radius"]),
            })
            n_comp += 1
        np.savez_compressed(os.path.join(out_pairs, p["file"]), **arrays)

        if (k + 1) % 200 == 0 or k == len(pairs) - 1:
            print(f"  {k + 1}/{len(pairs)} pairs ({time.time() - t0:.0f}s)",
                  flush=True)

    # Copy the index (skip reasons unchanged) with a provenance note; carry over
    # the donor accumulation cache so Step 2 doesn't re-scan the sequence.
    index["config"]["length_prior"] = args.length_prior
    index["config"]["length_mode"] = args.length_mode
    index["config"]["seed"] = args.seed
    index["config"]["recomputed_from"] = os.path.basename(src_dir)
    with open(out_index, "w") as f:
        json.dump(index, f)
    accum = os.path.join(src_dir, f"accum_cache_{args.seq}.npz")
    if os.path.exists(accum):
        shutil.copy(accum, os.path.join(out_dir, f"accum_cache_{args.seq}.npz"))
        print(f"Copied donor accumulation cache -> {out_dir}")

    print(f"\nRecomputed {n_comp} completed / {len(pairs)} pairs -> {out_pairs}")
    print(f"Index -> {out_index}")


if __name__ == "__main__":
    main()
