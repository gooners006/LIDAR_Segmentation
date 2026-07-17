# Donor-Frame Occluded-Side Metric — Valid Real-Data Completion Evaluation

Direction 1 deliverable. Built and validated 2026-07-17 (Finding #32).
Plan decisions locked with the user on 2026-07-17 (visibility mask,
one-directional coverage, pipeline TP inputs, raw + mirrored baselines).

## Why

Completion had **no valid real-data metric**: static-car accumulation pseudo-GT
is itself one-sided, so plain Chamfer rewards under-completion — the raw
partial scored best on every real example (Finding #26). This blocked measuring
Directions 2 (geometry fixes) and 3 (real-data fine-tuning) on real data.

## Idea

For a static car observed from many ego viewpoints, complete it from **one
frame's** pipeline cluster and score the result only against surfaces that
**donor frames** (all other observations of the same car) saw but the input
frame did not. The raw partial covers none of that surface *by construction*;
any coverage is added surface, and the amodal GT box bounds hallucination.

## Method

Per (frame, instance) TP pair on the 40 well-observed static cars from
`output/08/amodal_gt.json`:

1. **Input** = the production pipeline's cluster (detection + Stage B scratch
   classifier + greedy point-IoU ≥ 0.3 matching, same machinery as
   `completion_box_eval.py`), completed via the production `complete()` path
   (`pcn_kitti_best.pth`, L-shape gate). Sensor frame; world via `poses[i] @ Tr`.
2. **Donor cloud** = the instance's GT (sem=10) points from all observation
   frames *except* the input frame, world frame, 3 cm voxel
   (`amodal_gt.accumulate_instances` machinery).
3. **Novel set (visibility mask)** = donor points ≥ τ from every input point
   (cKDTree). τ = 0.15 m primary; 0.10/0.20 as stability checks.
4. **Score three method clouds** against the novel set (one-directional,
   novel → method):
   - `med_dist` — median novel-point distance to the method cloud (m);
   - `cov` — fraction of novel points within 0.1 m;
   - `out_of_box` — hallucination guard: fraction of method points outside the
     amodal GT box + 0.2 m margin.
   Methods: **raw** partial, **mirrored** partial (union of raw + reflection
   across `complete()`'s own estimated symmetry plane — canonical X, via the
   extracted `estimate_canonical_frame()`), **completed** (PCN).
5. **Qualification:** pair counts iff the input passes the L-shape gate and the
   novel set has ≥ 100 points at the primary τ.
6. **Statistics:** per-car medians + Wilcoxon signed-rank across cars (frames
   of a parked car are autocorrelated — same design as Finding #29).
7. **Secondary:** symmetry self-consistency CD (completed cloud vs its own
   canonical-X mirror) — reference-free plausibility signal.

### Region breakdown (diagnostics for Direction 2)

Novel points at primary τ are split in the GT-box frame, relative to ego:
`far_side` (opposite side of the length axis), `far_end` (opposite end, beyond
L/4), `top` (above H/4; world up = −Y).

## Scripts and outputs

| Step | Script | Output (`output/experiments/donor_metric/`) |
|---|---|---|
| 1. Pair sweep + cache | `scratchpad/donor_metric_step1.py` | `pairs_08/p_f<frame>_i<inst>.npz`, `step1_index_08.json` |
| 2. Metric computation | `scratchpad/donor_metric_step2.py` | `donor_metric_records_08.json`, `accum_cache_08.npz` |
| 3. Statistics + gate | `scratchpad/donor_metric_step3.py` | `donor_metric_summary_08.json` |
| Figure | `scratchpad/donor_metric_viz.py` | `output/figures/donor_metric_08.png` |

All run as `.venv\Scripts\python.exe scratchpad\<script> --seq 08`; outputs are
`--overwrite`-guarded. Supporting refactor: `complete()`'s gate + canonical
frame estimation extracted into `PointCloudCompleter.estimate_canonical_frame()`
(`src/completion.py`), verified bitwise behavior-preserving, so the mirrored
baseline uses *exactly* the production geometry.

## Results (seq 08, 2026-07-17)

Sweep: 2,092 TP pairs on all 40 well-observed cars (2,063 candidate frames);
1,337 gate-passed completions (733 fragment_input, 22 merge_suspected — matches
the #23/#29 split rate). All 1,337 pairs qualified (≥ 100 novel pts @ τ=0.15);
39 cars (one car's pairs were all gate-skipped).

### Headline — per-car medians (n=39), τ = 0.15

| method | cov@0.1 | med novel-dist (m) | out-of-box |
|---|---|---|---|
| raw | 0.000 | 0.518 | 0.000 |
| mirrored | 0.043 | 0.332 | 0.008 |
| completed | **0.304** | **0.161** | 0.000 |

All three pairwise Wilcoxon contrasts (completed>mirrored>raw on cov, both
metrics): p < 1e-6, n = 39.

### Validation gate — all four items PASS

- **(a) raw ranks last** on coverage at every τ ✓
- **(b) stability:** median per-car IQR of completed cov = **0.14** — moderate
  frame-to-frame spread (viewing geometry varies), acceptable for per-car-median
  statistics.
- **(c) ranking stable across τ** ∈ {0.10, 0.15, 0.20} ✓ (completed cov
  0.345 / 0.304 / 0.281 — level shifts, order never inverts).
- **(d) hallucination guard:** completed out-of-box car-median **0.0003** ≤
  mirrored 0.0083 ✓ (center/heading errors make mirrored spill more).

### Breakdowns

- **Input size (pooled pair medians, completed cov):** <100 pts 0.302 (n=49),
  100–300 0.307 (n=281), ≥300 0.331 (n=1007). Completion is size-robust;
  mirrored degrades on sparse inputs (0.010 → 0.083 with size).
- **Regions (per-car medians, completed / mirrored):** far_side 0.321 / 0.083;
  top 0.203 / 0.061; **far_end 0.133 / 0.001** — the far end is the weakest
  region, quantifying #29's length under-completion on the new metric (the
  Direction-2 target now has a number to move). Mirrored's ~0 far_end is a
  sanity check: reflection across the length-axis plane cannot add far-end
  surface.
- **both_sides_seen split (far_side):** cars whose far side the donors actually
  saw richly: completed 0.509 / mirrored 0.268 (n=13); one-side-only cars:
  0.305 / 0.067 (n=26).
- **Symmetry self-CD** (completed, car median): 0.122 m.

### Figure

`output/figures/donor_metric_08.png` — 6 BEV panels, best→worst completed
coverage. Best case (cov 0.89): completion fills the GT box against the novel
donor cloud. Worst cases show the two failure modes: (i) far-end
under-completion — the completed cloud stops short of the unseen end (#29);
(ii) heading/center mis-estimation on diagonal or sparse views — the completed
cloud is rotated off the box (these are also the pairs the out-of-box guard
flags).

### Reproducibility

Step 2 rerun on the cached pairs reproduces the records exactly (byte-identical
modulo timestamp). Step 1 is deterministic given the checkpoints
(rng-consumption order fixed per pair; `estimate_canonical_frame` is rng-free).

## Record schema

`donor_metric_records_08.json`: `config` (taus, cov_thresh 0.1, box_margin 0.2,
donor_voxel 0.03, novel-set definition) + `records[]`, one per gate-passed pair:
`frame`, `inst_id`, `n_raw_pts`, `n_donor_pts`, `taus.{0.10,0.15,0.20}` →
`n_novel` + per-method `{med_dist, cov}`, `out_of_box` per method,
`regions.{far_side,far_end,top}` → `n` + per-method cov, `sym_self_cd`.
`donor_metric_summary_08.json`: per-τ per-car-median tables + Wilcoxon, gate
items, size bins, regions, both-sides split.

## Caveats

- Donor coverage is still viewpoint-limited: for one-side-only cars the novel
  set never contains the truly-unseen far side, so absolute coverage
  understates completion quality there; the metric is a *paired comparison*
  tool, not an absolute recall of the full car surface.
- GT-labeled (sem=10) donor points only — leaks no pipeline error into the
  reference, but assumes SemanticKITTI label quality.
- Static cars only, by construction; seq 08 only so far.
