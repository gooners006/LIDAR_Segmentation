# Point-Cloud Completion — Research Plan

Created: 2026-06-30. Living document. Status markers: [ ] todo, [~] in progress, [x] done.

## Context

Project focus shifted to deepening point-cloud completion, prioritizing
**thesis narrative strength**; retraining is acceptable.

Completion is currently **un-measurable on real data**: the static-car
accumulation pseudo-GT is invalid (the accumulated cloud is itself one-sided, so
Chamfer rewards under-completion — the raw partial scores best; Findings #26/#27).
A valid synthetic metric exists (in-distribution KITTI-like: CD 0.16, F@0.1 0.76).
Solved already: blob bug (#26), inference normalization, L-shape input gate (#27),
model choice PCN≈PoinTr on real → keep PCN (#28).

## Roadmap (4 directions)

1. **Valid real-data completion metric** — donor-frame occluded-side Chamfer
   (split a static car's frames by ego viewpoint; input = set A, reference = set B
   which contains surface unseen in A) + symmetry self-consistency (secondary) +
   curated synthetic bench. Foundational; unblocks measuring 2 and 3 on real data.
2. **Improve complete() geometry** — centroid (dominant residual error, #26),
   90° heading flip; startable now on the valid synthetic metric via controlled
   perturbation experiments.
3. **Close train-vs-real partiality gap** (#28 bottleneck) — better partial
   synthesis and/or masked-Chamfer fine-tuning on real cars. Retraining OK.
   Contingency: negative-result precedent (#16/#17/#19); only after 1/2.
4. **Downstream utility of completion** — (a) completion improves bbox
   dimensions/orientation (measurable now via GT boxes); (b) recover split cars
   to attack the recall ceiling (#23); (c) completion quality as detection signal.

**Chosen order (narrative-first): 4a → 1 → 2 → 3.**

Idea backlog from the pre-plan session handoff (symmetry-mirror input,
symmetry-derived center, plausible-car-rate eval recipe):
`docs/completion/next_ideas.md`.

## Active plan — Direction 4a: "Does completion improve the box?"

**Hypothesis:** PCN completion yields a bbox closer to amodal GT than the raw
single-frame partial, improving occlusion-truncated dims (W, far-end L); heading
neutral (#27).
**Primary metrics:** |ΔL|, |ΔW|, |ΔH| (m), BEV oriented-box IoU.
**Secondary:** yaw error (mod 180°, car symmetry).
**Comparison:** raw-partial box vs completed box, each vs amodal GT box.

**Pseudo-GT trap avoidance:** restrict to static cars (sem=10) so accumulation is
valid; add a viewpoint-azimuth coverage filter so amodal W is trustworthy;
emphasize L/H/yaw as cleanest signals and treat W under the filter explicitly.

### Steps
- [x] **Step 0 — Amodal GT box builder** (`scratchpad/amodal_gt.py`): DONE
  (2026-07-02). Seq 08: 393 sem=10 instances, 388 fitted, **40 well-observed**;
  well-observed dims median L 4.14 / W 1.75 / H 1.47 m (in-range 38/40, 29/40,
  29/40 — misses are genuine compacts ~3.0–3.5 m). Cached
  `output/08/amodal_gt.json`; visual check `output/08/amodal_gt_check.png`
  (`scratchpad/amodal_gt_viz.py`). Beyond the planned azimuth-coverage filter,
  three extra guards proved necessary: (1) reject instance IDs ever labeled
  moving-car (sem=252) — stop-and-go cars keep their ID and smear the
  accumulation; (2) face-support counts (≥40 pts within 0.35 m of each box
  face) — azimuth coverage alone misses far ends occluded by adjacent parked
  cars; (3) overhang truncation flag (> max(15, 0.1%·n) pts beyond a face
  +0.15 m ⇒ the percentile trim cut a sparsely-observed real tail). A
  support-histogram extent estimator was tried and reverted (truncated sparse
  ends harder and inflated W/H by bin quantization); percentile extents
  (0.5/99.5) + the overhang *flag* won. Full write-up:
  **`docs/completion/amodal_gt.md`** (method, guards, results, verification,
  JSON schema).
- [x] **Step 1** (`scratchpad/completion_box_eval.py`): DONE (2026-07-05).
  2,075 TP pairs over all 40 well-observed cars (2,063 candidate frames);
  1,339 completed, 714 fragment-gated + 22 merge-gated (35%, matches #23 split
  rate). Records: `output/experiments/completion_box_eval/step1_records_08.json`.
- [x] **Step 2** (`scratchpad/completion_box_eval_step2.py`): DONE (2026-07-05).
  Per-car medians (n=39), Wilcoxon: BEV IoU 0.707→0.747 (p=.002), |ΔW|
  0.270→0.170 (p=1.5e-4), |ΔH| 0.255→0.131 (p=1.6e-10), center 0.286→0.234
  (p=2.8e-5); L and yaw neutral. Aggregates: `step2_metrics_08.json`.
- [x] **Step 3** (`scratchpad/completion_box_eval_viz.py`): DONE (2026-07-05).
  6-panel overlay figure `output/figures/completion_box_overlays_08.png`;
  **Finding #29** recorded in `docs/findings.md`.

### Decision criterion
- Completed beats raw on dim error / BEV IoU → headline "completion adds value";
  proceed to Direction 1.
- Neutral or worse → fix complete() geometry (Direction 2) before claiming value;
  reorder roadmap. Either outcome is a documented finding.

**DECISION (2026-07-05): criterion met — "completion adds value" established
(Finding #29). Proceed to Direction 1.** Direction-2 targets logged from the
breakdowns: (a) length under-completion on normal cars (signed ΔL −0.49 raw →
−0.55 completed; the far end is not extended), (b) heading errors on sparse
inputs (the one strong IoU regression case). Largest gains on sparse inputs
(<100 pts: IoU 0.461→0.599).

### Reuse note
Step 0's accumulation + viewpoint-coverage infrastructure is the same machinery
Direction 1a (donor-frame Chamfer) needs — build once, use for both.

## Direction 1 — COMPLETE (2026-07-17): donor-frame occluded-side metric, validated

**Finding #32**; full method/results/schema: **`docs/completion/donor_metric.md`**.

- [x] Design locked with user 2026-07-17: visibility-mask novel set (τ=0.15),
  one-directional coverage (novel→method, cov@0.1 m), out-of-GT-box guard,
  pipeline TP inputs, raw + mirrored baselines, per-car medians + Wilcoxon.
- [x] Steps 1–3 + figure (`scratchpad/donor_metric_step{1,2,3}.py`,
  `donor_metric_viz.py`): 2,092 TP pairs / 1,337 gate-passed / 39 cars on
  seq 08 → `output/experiments/donor_metric/`,
  `output/figures/donor_metric_08.png`.
- [x] **Validation gate: all four items PASS** (raw last; per-car IQR 0.14;
  ranking stable across τ; completed out-of-box 0.0003 ≤ mirrored 0.0083).
- [x] Headline (per-car medians, n=39, τ=0.15): cov@0.1 raw 0.000 / mirrored
  0.043 / **completed 0.304**; med novel-dist 0.518 / 0.332 / **0.161 m**;
  all Wilcoxon p < 1e-6.
- [x] Supporting refactor: `estimate_canonical_frame()` extracted from
  `complete()` (`src/completion.py`), verified behavior-preserving.
- Symmetry self-consistency (secondary, A6): implemented (car-median self-CD
  0.122 m); usefulness as a reference-free signal not yet tested — optional.

**Direction-2 targets, now measurable:** far_end cov 0.133 (vs far_side 0.321,
top 0.203) = the #29 length under-completion number to move; heading/center
errors on diagonal/sparse views (worst figure panels + out-of-box flags).

**Next: Direction 2 — improve `complete()` geometry**, measured with this
metric (report per-car-median cov@0.1 @ τ=0.15 + Wilcoxon, far_end split).

## Direction 2 — improve `complete()` geometry

### Step 1 — far-end under-completion: DONE (2026-08-01, Finding #35)

- [x] **Longitudinal length prior** in `estimate_canonical_frame()` (extend-only
  Z push toward the ego-far end; `COMPLETION_CAR_LENGTH_PRIOR = 4.14 m`; mirrors
  the width prior's `sign(center[·])` ego-direction mechanism; inference-only).
  Shipped ON (constructor `length_prior` default = constant; `None` disables).
- [x] **Synthetic mechanism check** (true GT, `length_prior_synth_check.py`):
  far-quarter cov 0.42 → 0.59, CD/F improve, under-reach halved. Prior must be
  near true full length (4.5 ≈ ceiling on 4.5 m synthetic cars).
- [x] **Real donor metric, paired A/B** (`donor_metric_recompute.py` +
  step2/step3 on `donor_metric_len_{off,414,450}`): far_end cov 0.123 → **0.324**
  at 4.14, overall cov 0.307 → 0.428, out_of_box 0.0004 → 0.0014 (guard holds).
  **4.5 rejected** — out_of_box 0.0122 breaks the ≤ mirrored 0.0083 guard
  (over-extends compacts). Decision criterion (far_end ↑, no guard regression)
  met at 4.14 only.
- [x] **Real box metric** (`length_prior_box_recheck.py`): reverses #29's length
  regression — signed ΔL −0.44 → −0.32 (completed now beats raw), |ΔL| 0.44 →
  0.35, width flat, |ΔH| +1.8 cm.

Sign choice (ego direction) self-validated on real data: the donor `far_end`
region is ego-defined, so a wrong sign would have lowered far_end cov — it rose.

**Follow-on (not done):** regenerate production `output/08` completions + the
#29/#32 production-config tables under the shipped prior when refreshing thesis
artifacts.

### Step 1b — per-car length estimate: DONE (2026-08-02, Findings #36/#37)

- [x] **Offline estimator selection against amodal GT, no PCN inference**
  (`length_estimator_probe{,2}.py`). Killed aspect-ratio (corr(GT L, GT W) =
  +0.018 — a perfect width is *worse* than the constant), height/range/density
  (|corr| ≤ 0.13), a far-end face-support truncation test (inverted), and
  track-max (worst estimator, compact bias +0.95 m). Survivor: `fit_length`,
  corr +0.52 per frame / +0.87 per car.
- [x] **Shipped:** `track_length_estimate()` = per-car **q90 of gate-passed
  `fit_length` + 0.12 m**, fallback to 4.14 below 5 frames. Optional
  `length_estimate` arg through `complete()`; `main.py` aggregates per track.
- [x] **Box metric, band split:** compact signed ΔL +0.295 → **+0.063**
  (p = .016); ALL |ΔL| 0.354 → **0.304**, BEV IoU 0.747 → **0.771**, center err
  0.229 → **0.184**. Beat q95 on both metrics and every band.
- [x] **Leakage control** (single-frame OLS): fixes compacts partially
  (+0.141) but regresses normals (|ΔL| 0.345 → 0.388) — only the track estimate
  fixes compacts without a tradeoff.
- [x] **Guard fix (#37):** #32's out-of-box gate is a pooled median and was blind
  to the shipped prior hallucinating at 100× the compact band's mirrored
  baseline. Added per-band gate `d2` to `donor_metric_step3.py`; backfilled.

**Cost:** pooled donor coverage 0.403 → 0.364, far_end 0.346 → 0.316 (worse on
the normal band, better on compact + long).

### Step 1c — the *other* under-extension (NEW, opened by #36's control run)

A deliberate over-extension control (q90 **+0.45**) improved **both** metrics on
normal/long cars — coverage 0.364 → 0.483, far_end 0.316 → 0.509, box |ΔL|
0.304 → 0.210 — so those completions are still genuinely too short even when
`L_est` is unbiased. This is an under-extension *inside* the completion, not in
the center estimate: PCN under-fills its normalized frame, and the center push
also drives `radius` (hence output scale), so the length prior doubles as a
rescale. A per-car length estimate cannot correct it.

Required compensation is strongly length-dependent (≈0.02 / 0.68 / 1.02 m for
compact / normal / long), so a single offset cannot serve all bands and fitting
the trend on 8/27/5 cars would overfit. Candidate framings, cheapest first:
1. Separate the two jobs — estimate `radius` independently of the center push, so
   scale stops riding on the length prior.
2. Calibrate PCN's fill factor on synthetic true-GT cars (where the shrinkage is
   directly measurable) rather than fitting it on the 40 amodal cars.
3. Only then consider a length-dependent target.
Measure with the same band-split box metric + per-band guard `d2`.

### Step 2 — heading/center on diagonal/sparse views (remaining target)

The other #29/#32 weakness (worst donor panels + out-of-box flags): completed
cloud rotated off the box on diagonal/sparse inputs. Candidate levers:
symmetry-derived center/heading, symmetry-mirror input (`next_ideas.md` #1/#2).
Same donor + #29 box metrics.
