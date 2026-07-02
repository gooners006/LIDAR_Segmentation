# Next session — improve completion quality

Goal: raise the real seq-08 plausible-car rate above the current **18/26** (PCN) /
16/26 (PoinTr) and produce crisper car shapes on one-sided real LiDAR clusters.

## What we already know (don't re-derive)

- **It is NOT a decoder-capacity problem.** PoinTr halved synthetic CD (0.063 vs 0.125)
  and hit F@0.1 0.99 vs 0.76, yet was a *wash* on real data (Finding #28). A bigger/
  better model does not help.
- **The real bottlenecks are two (Findings #26, #27, #28):**
  1. **Centroid/scale estimation in `completion.py complete()`** — Finding #26 isolated
     *centroid* as the dominant residual error (scale is solved by ×1.137). Current
     center = bbox-centroid + ego-direction width prior + up-shift. Still imperfect.
  2. **Residual real↔synthetic partiality gap** — both models trained on KITTI-like
     synthetic partials; real clusters differ in occlusion/scan-line structure.
- **Input cleanliness already gated** (Finding #27): fragments/merges are skipped. The
  remaining failures (`110, 2001, 1538, 3194`) are *merges/L-shape two-car blobs* the
  gate lets through — bad inputs, identical failure in both models.
- **Pseudo-GT CD on real static cars is INVALID** (Finding #26): accumulated LiDAR is
  itself one-sided, rewards under-completion. Real assessment must stay qualitative or
  use a different metric. Synthetic (true GT) is the only quantitative ground.

## Highest-leverage ideas (ranked)

1. **Symmetry prior (cheap, classical, attacks the actual problem).** Cars are
   bilaterally symmetric about their longitudinal (length) plane. After reorienting to
   the canonical frame, **mirror the partial across the X=0 (width) plane and union it**
   before feeding the completer — this directly densifies the one-sided input that BOTH
   models choke on. Could even bypass/augment the network. Test: does mirrored-input
   completion beat raw-input on the 8 borderline CLEAN tracks? Likely the single best lever.
2. **Better center estimation.** Replace/augment the bbox+ego-prior center with a
   symmetry-derived center (the mirror plane that best aligns the partial to itself gives
   both heading AND lateral center). Ties into #1. Finding #26 says this is the dominant
   residual error.
3. **Light real-data fine-tuning** — but eval is the catch (pseudo-GT invalid). Options:
   self-supervised symmetry loss on real partials; or treat the mirrored union as a weak
   target. Only worth it if #1/#2 plateau.
4. **Merge-splitting for the bad inputs** (110/2001 are two cars). Overlaps the recall-
   ceiling work (Findings #23–24, mostly negative). Low priority — these are few.

## Evaluation plan for next session

- Quantitative: synthetic val (true GT) for any model/inference change — `train_pcn.py`
  / `train_pointr.py --kitti-like`.
- Real qualitative: re-run `scratchpad/compare_pcn_pointr.py` style — render the 26
  clean-gated tracks, plausible-car-box rate (L∈[3.3,4.9], W∈[1.5,2.1], H∈[1.1,1.7]),
  eyeball BEV (X-Z plane; frame is Y-down). Baselines to beat: PCN 18/26.
- Production model = `pcn_kitti_best.pth` (PoinTr is equivalent, kept for reference;
  `complete()` dispatches either by checkpoint).

## Pointers
- `src/completion.py` — `complete()` (normalization, center/scale/heading, L-shape gate),
  `_load_model()` (dispatch). Constants: SCALE_CORRECTION=1.137, CAR_WIDTH_PRIOR=1.9,
  UP_SHIFT=0.25, FRAGMENT_MIN_LENGTH=2.7, MERGE_MAX_WIDTH=2.3.
- Comparison/viz scripts: `scratchpad/compare_pcn_pointr.py`, `scratchpad/viz_all26.py`.
- Outputs: `output/08_ab_gated` (PCN), `output/08_pointr` (PoinTr),
  `output/all26_pcn_vs_pointr.png`, `output/compare_pcn_pointr.png`.
- Findings #26 (inference fix + calibration), #27 (gate + heading), #28 (PoinTr verdict).
