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
- [ ] **Step 1** (`scratchpad/completion_box_eval.py`): per-frame label-propagated
  detections (reuse `get_frame_detections`), keep TP car clusters matched to a
  well-observed static GT instance; fit raw-partial box and completed box
  (run `completion.complete()`); look up amodal GT box.
- [ ] **Step 2**: paired raw-vs-completed metrics against GT; report within-car
  improvement (paired test).
- [ ] **Step 3**: result table + 4–6 box-overlay figures (GT black / raw blue /
  completed green); record finding in `docs/findings.md`; update
  `docs/project_state.md`.

### Decision criterion
- Completed beats raw on dim error / BEV IoU → headline "completion adds value";
  proceed to Direction 1.
- Neutral or worse → fix complete() geometry (Direction 2) before claiming value;
  reorder roadmap. Either outcome is a documented finding.

### Reuse note
Step 0's accumulation + viewpoint-coverage infrastructure is the same machinery
Direction 1a (donor-frame Chamfer) needs — build once, use for both.
