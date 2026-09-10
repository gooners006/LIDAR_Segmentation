# Project State

Last updated: 2026-09-10

> **Full chronological record: `docs/session_history.md`.** This file was compacted on
> 2026-09-07 — the historical per-session narrative blocks (thesis reframe/polish/review
> rounds, chapter-by-chapter drafting, delegate-brief T1-T13, completion Directions
> 1/2/4a, cross-file audits) were removed from here and a complete pre-compaction snapshot
> was appended to `session_history.md`. This file now holds current state only. Experiment
> detail lives in `docs/findings.md`.

## RESEARCH FREEZE — LIFTED 2026-09-09 (was declared 2026-08-21)

The write-up freeze is **lifted** (user, 2026-09-09): the pipeline may be re-run and models
retrained. Ordinary data-safety hygiene still applies (do not overwrite `output/08` or existing
checkpoints without cause; new work to fresh paths). The committed reference configuration
(`PIPELINE_CONFIG` at `src/pipeline.py:10`; checkpoints `stage_b_scratch_best.pth`,
`pcn_kitti_best.pth`; eval protocol point-level IoU ≥ 0.3 greedy 1-to-1, supported-vehicles,
track filter ON, micro-averaged) is documented in the sections below.

## Thesis Status (2026-09-09) — REFRAME PLANNED (not yet executed)

The manuscript is still fully drafted around the **donor-frame coverage metric** in the `.tex`
files (no thesis edits made this session). A **reviewed, approved reframe plan** now supersedes
that direction; execution is the next task.

- **New direction:** main contribution = the modular detection-and-completion **framework**;
  second = SemanticKITTI + ShapeNet real-data grounding. New title: *"A Modular Detection and
  Completion Framework for Occluded Vehicles in Automotive LiDAR."* Donor metric, amodal-box
  utility, and home-made L/W/H plausibility all **dropped** (all self-made instruments).
- **New completion evaluation:** synthetic CD/F (0.16 / F@0.1 0.76) + Chen-2020
  independent-classifier plausibility (new experiment, pending) + fidelity UCD/UHD reported
  with the under-completion caveat (#52). RQ2 reframed to plausibility + fidelity.
- **Plans (outside repo):** `~/.claude/plans/drop-the-whole-donor-frame-nifty-papert.md`
  (reframe) and `~/.claude/plans/independent-classifier-plausibility.md` (new experiment).
- **Execution order:** run the plausibility experiment FIRST (it supplies the only favorable
  real-data completion number), then reframe Ch1+main → Ch2 → Ch3 → Ch4 → Ch5 → global style
  passes (drop em-dashes, colon-not-period on list leads, minimize bold/italic). **Task G is now
  half-done (judge trained + all-sequence generation complete, 2026-09-10); scoring/stats pending —
  see "Task G" below.**

## Task G — Independent-Classifier Plausibility (in progress, 2026-09-10)

The prerequisite completion experiment (Finding #53 pre-registration; plans
`~/.claude/plans/independent-classifier-plausibility.md`). Judge and data generation done;
scoring, ceiling, and stats remain.

- **Judge — DONE.** In-repo PointNet++ SSG (Option A, built on `src/pointr.py` FPS/index/distance
  primitives + ball query; no new deps, no CUDA). Code `scratchpad/plausibility/{pointnet2_ssg,
  modelnet40,train_judge}.py` (gitignored). Trained from scratch on `modelnet40_ply_hdf5_2048`
  (`dataset/modelnet40/`, gitignored), seed 42, 100 epochs. **Best test top-1 = 0.9214** (gate
  >=0.88 PASS; reference-faithful, no fallback). Checkpoint `checkpoints/modelnet40_pointnet2.pth`.
- **Generation — DONE.** All 11 labeled sequences (00-10) run at production config to
  `output/experiments/plausibility_gen/<seq>/` (isolated from `output/08`). Per track:
  `<id>_partial.ply` (raw PCN input) + `<id>.ply` (completed, 4096 pts). **2,822 completed tracks**
  total (paired-test N). `tracks.json.tracks[]` has `centroid_history` (static/mover), `completed`,
  `ref_fit_length/width`.
- **Harmonization — RESOLVED (measured).** Completed clouds AND ModelNet40 cars are both
  y-up / z-length / x-width, so **no up-axis rotation** is needed (reverses the plan's z-up
  assumption; retires review point 2). Harmonization = unit-sphere + subsample to 1024; only the
  heading sign remains, to be confirmed by an overlay viz.
- **PENDING:** (1) scoring harness → judge car-prob + argmax on the 2,822 pairs, McNemar (argmax)
  + Wilcoxon (car-prob), heading-sign overlay-viz check first; (2) static-accumulation ceiling
  (SemanticKITTI sem=10) for the H3 shape-quality read; (3) record a Finding → feeds the Ch4 reframe.
- **Source layout / build:** unchanged — six files under `docs/writing/thesis/` (`ch0_abstract`,
  `ch1_introduction`, `ch2_background`, `ch3_methodology`, `ch4_evaluation`,
  `ch5_discussion_conclusion`) wired by `main.tex`; clean build, ~76 pp. Still NOT advisor-reviewed.

## Evidence Tasks — all closed

Read-only analyses against the frozen config (freeze-safe). Full detail in `docs/findings.md`.

- **B1** (IoU sensitivity) + **B6** (geometric-only ablation) — DONE (#47).
- **B2** (GT-eligibility count) — DONE (#48): ~25.5% of annotated cars excluded by the
  >=10-surviving-points rule -> implied recall vs all annotated cars ~=0.54.
- **LOSO cross-validation** — DONE (#51, `results/loso/`): leakage was precision-only
  (~2 pts); pooled over 11 seqs P 0.874 / R 0.719 / F1 0.789. Answers the
  "validation-as-test" critique; shipped checkpoint + `PIPELINE_CONFIG` untouched.
- **UCD/UHD reference-free comparison** — DONE (#52): standard input-fidelity metrics rank
  raw partial = mirror = 0 and the completion worst, while donor cov@0.1 credits the
  completion. Strengthens the metric's novel-set-restriction novelty.
- B3 (literature table) / B4 (distance recall) / B5 (pipeline diagram) — writing/figure
  work, folded into the chapters.

## Defense Deck (Phase 8) — planned, not yet built

Planning artifacts committed (`docs/defense/plan.md` + `docs/defense/storyboard.md`),
grounded in the peer template `docs/LVTN.pptx` (same advisor) and the frozen numbers.
Locked: python-pptx onto a cleaned copy of `LVTN.pptx`; 25-30 min / ~40 slides + 7 backups;
completion-forward (donor metric headline); own-the-limitations; reuse thesis figures.
**Resume at plan Section B** to build the `.pptx` (purge peer content from the template copy;
python-pptx slide-clone corruption is the known risk -> fallback fresh 16:9 theme). Deck NOT
yet generated.

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working (recall ceiling confirmed) |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Binary Stage B trained |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | Fixed inference (#26); single-frame completion in `main.py`; L-shape input gate (#27) → completion precision 38%→69%; length prior (#35) → far_end cov 0.13→0.32; per-car length estimate (#36) → box \|ΔL\| 0.354→0.304, compact overshoot fixed |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics + sweep flags), `src/visualize_gt.py` (GT vs pipeline toggle viz), `src/train_classifier.py`, `src/mine_stage_b.py`, `src/analyze_clustering.py` (filter ablation + merge/split), `src/explore_merge_strategies.py` (recall strategy exploration),
`src/test_invariants.py` (pytest invariant tests, T4).

## Classifier — Binary (Complete)

- `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- **Production (since 2026-07-14, Finding #31): `checkpoints/stage_b_scratch_best.pth`** —
  trained on real mined clusters only, from random init. Stage A synthetic
  pretraining dropped from the final pipeline (kept as thesis ablation, #7/#25/#30).
- Stage A (ablation material): ShapeNet car (02958343) as positive, unknown_fraction=0.50. Best val macro F1: 0.9986
- Stage B: Mined from SemanticKITTI (train: seqs 00-07,09-10; val: seq 08; 5000 frames each, purity 0.75)
  - Train: 420,333 clusters (88,600 car / 331,733 not-car)
  - Val: 130,394 clusters (24,968 car / 105,426 not-car)
- Scratch best: epoch 14/15, macro F1 0.9285; fine-tuned (A→B) best: epoch 13/15, macro F1 0.9225
- **Stage A ablation done (Finding #25):** advisor's "too perfect synthetic data" concern resolved — from-scratch Stage B matches pretrained (macro F1 0.9285 vs 0.9225). Synthetic prior is redundant given 420k real clusters; pretrained keeps a weak pipeline precision edge. See `docs/classifier/`.
- **Cross-domain matrix done (Finding #30, advisor-requested):** sim-to-real gap is total and symmetric — car F1 = 0.000 in every off-diagonal cell (synthetic-trained on real, real-trained on synthetic); fine-tuning forgets synthetic entirely. Script: `scratchpad/cross_domain_classifier_eval.py`; results: `output/experiments/cross_domain_classifier/`.

## Eval Metrics

Headline numbers below use the production scratch checkpoint (Finding #31;
prior fine-tuned-checkpoint numbers preserved there).

### Seq 00 (deterministic, 100 frames) — headline baseline

| Metric | Value |
|--------|-------|
| Precision | 0.984 |
| Recall | 0.761 |
| F1 | 0.859 |
| Mean IoU | 0.942 |

TP=1242 FP=20 FN=389. RANSAC is deterministic (`np.random.default_rng(42)`). Results reproducible across runs.
Promoted-config 100-frame eval (2026-07-23, report refresh): P 0.967 / R 0.777 / F1 0.862 / mIoU 0.962. TP=1296 FP=44 FN=371 (TP/FP/FN not separately recorded 2026-07-23; re-run and filled in 2026-08-02, same command/config, metrics unchanged — `.venv\Scripts\python.exe src/evaluate.py`).

### Seq 08 (full, 4071 frames) — generalization check (promoted config, updated 2026-07-23)

| Metric | Value |
|--------|-------|
| Precision | 0.905 |
| Recall | 0.730 |
| F1 | 0.808 |
| Mean IoU | 0.912 |

TP=25478 FP=2676 FN=9444. Promoted `PIPELINE_CONFIG` (voxel_before_denoise,
ransac_iterations=300, cluster_voxel_size=0.10; Finding #34). Pre-opt numbers
(0.903/0.699/0.788/0.895, TP=23823 FP=2565 FN=10240) preserved in #34.
Command: `.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000`.
Confirms the seq-00 story at 40× scale: precision-saturated, recall-limited;
per-frame recall anti-correlated with GT-car density, FP flat ~1/frame.
Figures: `output/figures/seq08_{bev_detections,failure_zooms,timeseries}.png`.

## Recall Bottleneck — Characterized; partly lifted by coarse-voxel clustering (#34)

The recall shortfall was root-caused to HDBSCAN **splitting cars into fragments**
(not a classifier problem). Six repair strategies all failed to generalise
(four with table outcomes + temporal aggregation and threshold lowering in prose;
see Ch 5 reconciliation note above) — **but** coarsening the clustering resolution (cv=0.10, promoted
2026-07-23, Finding #34) closed some intra-car density gaps and lifted recall
0.699→0.730, so the earlier "~0.74 hard limit" was **partly a resolution
artifact**, not fundamental. A smaller structural limit remains (density-based
clustering has no notion of objectness; coarser voxels would start merging
adjacent cars). Extensively investigated across multiple sessions:

### Root cause: HDBSCAN splitting (Finding #23)
- 31-37% of GT cars are split across multiple HDBSCAN clusters
- Large/close cars split most (66% split rate at 0-10m, 4% at 30-50m)
- Merging is negligible (0-0.5%)
- Recoverable ceiling (single-cluster GT cars): 63-68%, matching actual recall

### Geometric filter ablation (Finding #22)
- `min_volume` kills 68% of GT-matching rejected clusters, `min_points` kills 26%
- But these are sub-fragments from split cars, not independent missed detections

### Strategies attempted — all negative (Findings #21, #24)
- **BEV clustering:** F1 0.779 (vs 0.844 baseline). 2D projection merges overlapping objects.
- **Higher min_cluster_size:** MCS=20 → F1 0.852 on seq 00 but 0.801 on seq 08 (overfits).
- **Post-clustering fragment merge:** precision drops outweigh recall gains on held-out data.
- **Distance-adaptive HDBSCAN:** ring boundary artifacts; worse than global.
- **Temporal aggregation (prior session):** HDBSCAN on accumulated points → F1 collapsed to 0.073.
- **Lower cluster thresholds:** zero TP change.

**Conclusion:** Post-hoc (reassemble-after-splitting) interventions exhausted —
all negative. The one lever that worked was clustering *resolution* itself
(cv=0.10, now in production, #34), which recovered part of the loss (recall
0.699→0.730). Residual ~0.73 recall accepted; focus on other thesis contributions.

## Alternative clustering implementations in `pipeline.py`

All disabled by default, CLI-toggleable for documentation:
- `--clustering-method bev` — BEV connected-component clustering
- `--merge-fragments` — post-clustering fragment merge
- `--adaptive-hdbscan` — distance-ring HDBSCAN with per-ring MCS

## Completion — KITTI-like PCN VERIFIED; the blobs were an inference bug (Finding #26)

Root cause of prior PCN failures (#15-19): synthetic partials were OOD from the
real post-pipeline input (voxelized 0.05 m, ground-removed, single-viewpoint).
Built a KITTI-like single-view partial generator (`_render_kitti_like` in
`src/train_pcn.py`, `--kitti-like`; see `docs/pcn/kitti_like_partial.md`) and
trained PCN on it: `checkpoints/pcn_kitti_best.pth`, best val loss 0.1246
(= coarse CD + 0.5·fine CD; val fine-CD 0.066).

**Verdict (Finding #26): the data fix WORKED.** In-distribution synthetic eval is
clean (CD 0.16 m, F@0.1m 0.76 — real cars, not blobs). The "blobs on real data"
were **primarily an inference-normalization bug in `completion.py complete()`**:
it applies **3D PCA alignment + partial-radius/partial-centroid** normalization
that the model never saw in training (this breaks *every* PCN checkpoint, incl.
`pcn_best` — 3.5× worse CD even on in-distribution input). A corrected inference
path (no PCA; reorient gravity→Y, length→Z; scale ×1.137; full-car-center
estimate) de-blobs real seq-08 clusters into car-footprint shapes (see
`output/experiments/verify_pcn_step2/`). Scripts: `scratchpad/verify_pcn_step1.py` (synthetic,
calibration + ablation), `scratchpad/verify_pcn_step2.py` (real, multi-view + pseudo-GT).

Key sub-findings: scale is solved by the ×1.137 factor; **centroid estimation is
the dominant residual error**; training's `_augment_rotation` is roll-invariance
(about the length axis), not yaw. The static-car pseudo-GT metric is **invalid**
for completion — accumulated LiDAR is itself one-sided, so CD rewards
under-completion (raw partial scored lowest CD on every real example).

## Checkpoints

- `checkpoints/stage_b_scratch_best.pth` — binary classifier, real-data-only from scratch (**production**, Finding #31)
- `checkpoints/stage_b_best.pth` — binary Stage B fine-tuned from Stage A (kept for reproducibility)
- `checkpoints/classifier_best.pth` — binary Stage A classifier (ablation material)
- `checkpoints/pcn_kitti_best.pth` — PCN on KITTI-like partials (used by fixed `complete()`)
- `checkpoints/pcn_best.pth` — prior PCN (blobs on real data, #15-19)
- `checkpoints/modelnet40_pointnet2.pth` — ModelNet40 PointNet++ SSG plausibility judge (Task G,
  test top-1 0.9214; gitignored, real-data-independent, not part of the detection/completion pipeline)

## Completion — Frozen Results Summary

Mechanism is described above; full result tables and worked verdicts live in
`docs/findings.md` and `docs/completion/donor_metric.md`. Frozen headline numbers
(seq 08, promoted config, per-car medians):

- **Donor coverage @0.1** (novel-side surface recovery): raw 0.000 / mirrored baseline
  0.043 / **completed 0.304** (#32/#35/#36; ~7x the symmetry-mirror baseline).
- **Amodal-box utility:** completed beats raw partial — BEV IoU 0.725 -> **0.771**,
  |dL| -> 0.304, center err -> 0.184 (#29/#35/#36). Length uses a per-car q90 estimate
  (`COMPLETION_LENGTH_TRACK_QUANTILE=90` + 0.12 m), 4.14 m fallback below 5 frames;
  fallback fires on 119/518 (23%) of completed tracks on seq 08 (#41).
- **Movers** complete as plausibly as statics (57.9% vs 53.7%, #44) — plausibility only,
  not accuracy-validated.
- **Held-out (seq 00) pre-registered replication: PARTIALLY HOLDS** (#42) — both primaries
  (BEV IoU, donor cov@0.1) are decisive wins; the downgrade is a coverage gap (empty long
  band), not a weak metric.
- **T13 Step-1c** (decouple radius / fill_z) tested NEGATIVE — do not ship; flags default
  OFF (#45).
- **Roadmap / open ideas:** `docs/completion/plan.md`, `docs/completion/next_ideas.md`.

## Maintenance / Delegate Brief

Delegate-brief tasks **T1-T13 all complete** (one commit each; full record in the
`session_history.md` snapshot and `docs/findings.md`). Only **T14 (thesis, user-supervised)**
remains.

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. ~~Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN~~ — done (Finding #43, T10)
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
