# Project State

Last updated: 2026-07-23

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working (recall ceiling confirmed) |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Binary Stage B trained |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | Fixed inference (#26); single-frame completion in `main.py`; L-shape input gate (#27) → completion precision 38%→69% |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics + sweep flags), `src/visualize_gt.py` (GT vs pipeline toggle viz), `src/train_classifier.py`, `src/mine_stage_b.py`, `src/analyze_clustering.py` (filter ablation + merge/split), `src/explore_merge_strategies.py` (recall strategy exploration).

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
Promoted-config 100-frame eval (2026-07-23, report refresh): P 0.967 / R 0.777 / F1 0.862 / mIoU 0.962 (exact TP/FP/FN not separately recorded this session).

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
(not a classifier problem). Five *post-hoc* repair strategies all failed on
held-out data — **but** coarsening the clustering resolution (cv=0.10, promoted
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

## Project Focus (updated 2026-06-30): Point-Cloud Completion

Direction shifted to deepen completion, prioritizing **thesis narrative
strength**; retraining acceptable. Four directions:

1. **Valid real-data completion metric** — donor-frame occluded-side Chamfer +
   symmetry self-consistency; curated synthetic bench. Foundational: currently
   NO valid real-data metric (pseudo-GT CD invalid, #26/#27).
2. **Improve `complete()` geometry** — centroid (dominant residual error), 90°
   heading flip. Startable now on the valid synthetic metric.
3. **Close train-vs-real partiality gap** (#28 bottleneck) via masked-Chamfer
   fine-tuning on real cars. Retraining OK; contingency (negative-result
   precedent #16/#17/#19).
4. **Downstream utility** — completion improves bbox dims/orientation
   (measurable now via GT boxes) or recovers split cars (#23).

Chosen order (narrative-first): **4a → 1 → 2 → 3.** Scratchpad viz scripts kept
in scratchpad (Group 1 frozen records / Group 2 reusable tools); not promoted.

Full roadmap and active step plan: **`docs/completion/plan.md`**.

Prior completion milestones (DONE): KITTI-like PCN data fix + inference-bug fix
(#26), single-frame completion wired into `main.py`, L-shape input gate
(precision 38%→69%, #27), full seq-08 regenerated (`output/08`, 518 completed),
PoinTr benchmarked → keep PCN (#28; synthetic table corrected 2026-07-06 via
matched eval `scratchpad/matched_eval_pcn_pointr.py` — PoinTr's edge is small
(CD 0.161→0.153 m, F 0.755→0.782), not the "halves CD" originally recorded;
real-data equivalence and keep-PCN decision unchanged). See `docs/findings.md`.

## Direction 4a — COMPLETE (2026-07-05): "Completion adds value" established

Finding #29. All four steps done (`docs/completion/plan.md` has details):
Step 0 amodal GT (40 well-observed static cars, `output/08/amodal_gt.json`);
Steps 1–3 paired box eval (`scratchpad/completion_box_eval*.py`): 2,075 TP
pairs / 1,339 completed on seq 08.

**Headline (per-car medians, n=39, Wilcoxon):** completed box beats raw partial
on BEV IoU 0.707→**0.747** (p=.002), |ΔW| 0.270→**0.170** (p=1.5e-4), |ΔH|
0.255→**0.131** (p=1.6e-10), center err 0.286→**0.234** (p=2.8e-5); L and yaw
neutral. Gains largest on sparse inputs (<100 pts: IoU 0.461→0.599). Figure:
`output/figures/completion_box_overlays_08.png`.

Direction-2 targets logged (not blockers): length under-completion on normal
cars (far end not extended, signed ΔL −0.49→−0.55) and heading errors on
sparse inputs.

## Direction 1 — COMPLETE (2026-07-17): valid real-data completion metric

Finding #32; method/results: `docs/completion/donor_metric.md`. Donor-frame
occluded-side metric: complete from one frame's pipeline cluster, score
coverage of donor points (other frames of the same static car) ≥ 0.15 m from
every input point, + out-of-amodal-GT-box hallucination guard. Seq 08:
2,092 TP pairs / 1,337 gate-passed / 39 cars. **Validation gate: all four
items pass.** Headline (per-car medians, n=39): cov@0.1 raw 0.000 / mirrored
0.043 / **completed 0.304**, med novel-dist 0.518→**0.161 m**, out-of-box
~0; all Wilcoxon p < 1e-6. First real-data evidence PCN reconstructs unseen
surface (7× the symmetry-mirror baseline). Weakest region = far end (cov
0.133) — #29's length under-completion, now measurable. Supporting refactor:
`estimate_canonical_frame()` extracted from `complete()` (behavior-preserving).
Figure: `output/figures/donor_metric_08.png`.

## Immediate Next Steps — Direction 2: improve `complete()` geometry

Targets (from #29 + #32 breakdowns): (a) far-end under-completion — move
far_end cov 0.133; (b) heading/center errors on diagonal/sparse views.
Measure with the donor metric (per-car-median cov@0.1 @ τ=0.15, Wilcoxon,
far_end split) + the #29 box metrics. Idea backlog:
`docs/completion/next_ideas.md`. Plan: `docs/completion/plan.md`.

Far-end plan (written up in
`docs/report/results_report_2026_07_17_rev2.docx` §6.1): Step 1
inference geometry (L-shape near-corner anchor + longitudinal length prior +
symmetry-derived center; no retraining) → Step 2 visibility-weighted
asymmetric Chamfer retrain on KITTI-like synthetic → Step 3 (contingency)
masked-Chamfer real fine-tuning. Architecture changes ruled out (#28).

Amodal-GT cross-validation against KITTI raw tracklets: **RESOLVED as
impossible (2026-07-17)** — raw drive 2011_09_30_drive_0028 (= odometry seq
08) has no tracklet annotations (verified: 404 on the official S3 bucket
`avg-kitti/raw_data/2011_09_30_drive_0028/…_tracklets.zip`; control drive
2011_09_26_drive_0001 exists; 2011_09_30_drive_0027 and 2011_10_03_drive_0027
also unannotated). Thesis defense of the amodal GT instead rests on: paired
design tolerates reference noise (#29), construction guards (viewpoint
coverage + zero motion), and dimension sanity check (median L 4.14 / W 1.75 /
H 1.47 m vs published car statistics). Written up in rev2 §6.2.

### DONE: pipeline runtime optimization (Findings #33, #34; 2026-07-23)

`docs/perf/plan.md` + locked plan (Section A resolved). Baseline: full seq 08 =
**934.5 ms/frame** (stride-20 harness; 921 ms full-run). Tier 1 (exact-output,
per-frame TP/FP/FN bit-for-bit, #33) + Tier 2/3 (behavior-changing, promoted,
#34) landed. **Promoted combined config** (now the `PIPELINE_CONFIG` defaults):
`voxel_before_denoise=True`, `ransac_iterations=300`, `cluster_voxel_size=0.10`.

- **Runtime: 934.5 → 650.6 ms/frame (−30%, ≈1.54 fps)** — stride-20 benchmark
  `output/experiments/timing/timing_seq08_stride20_combined.json`.
- **Detection improved on full seq 08** (guard metrics, all up, none regressed):
  P 0.903→0.905, R 0.699→**0.730**, F1 0.788→**0.808**, mIoU 0.895→0.912
  (TP 23823→25478, FP 2565→2676, FN 10240→9444). cv=0.10 closes intra-car
  density gaps (#23) → +1655 TP.
- **≤400 ms target NOT reached, and proven unreachable via the authorized tiers:**
  the post-ground object cloud is intrinsically sparse, so coarse-voxel compresses
  it only ~18% and Python HDBSCAN has a ~185 ms floor on it. User decision
  (2026-07-23): **accept 650 ms** (real-time is out of scope); deferred GPU HDBSCAN
  (cuML/WSL2) and Open3D DBSCAN paths NOT pursued.

**Completion re-check DONE (isolate-config, 2026-07-23):** re-ran #29/#32 against
the promoted config, each keeping its own published classifier (#29 `stage_b_best`,
#32 `stage_b_scratch`) so only the config differs. **Neither finding shifts** —
#29 box quality unchanged (BEV IoU 0.747→0.739, |ΔW/H/L| within ±0.008), #32 donor
coverage barely moved (τ=0.10 0.346→0.338, τ=0.15 0.304→0.302), both still far
above their raw/mirrored baselines. Deltas within noise; n grew (n_pairs 2075→2262,
n_cars 39→40) because detection recall improved, so read population-level, not
paired. Detail + tables in Finding #34. Baselines preserved (`*_perf` outputs).
The #29 classifier drift (fine-tuned vs production scratch, pre-#31) is untouched
by design — flagged as a separate pre-existing item.

**`output/08` regenerated DONE (2026-07-23):** re-ran under the promoted config
(1040 tracks, 518 completed, 1558 PLYs); GT artifacts copied in; old output
preserved at `output/08_preperf_backup/` (reversible).

**Report refresh DONE (2026-07-23):** created
`docs/report/results_overview_2026_07_23.docx` — full refresh of §1–§6 + Summary
to the promoted operating point (one consistent operating point throughout), plus
an **honesty pass**: §3 reframed (dropped the obsolete "~0.74 hard ceiling"; the
cv=0.10 recall lift 0.699→0.730 is now first-class, ceiling "smaller than first
claimed"); §5.4 length prose made to match Table 4's "worse" verdict
(|ΔL| 0.428→0.463 m, p=0.034); [22] track-filter ablation given a provenance
caveat (0.762→0.801 is pre-opt; endpoint now 0.808). Item 3b (#29 fine-tuned vs
production scratch checkpoint) left undisclosed by design — immaterial to the
paired raw-vs-completed delta. All Tier-1/2/3 code edits + findings/state/report
remain **uncommitted**.

**Pending (user approval):** correct `docs/findings.md` #24 wording — it still
reads "~0.74 recall ceiling is a hard limit," now contradicted by #34.

### Deferred
- Thesis writing (pipeline description, experiment results, recall-ceiling discussion).
  Advisor progress report done 2026-07-05: `docs/report/progress_report_2026_07_05.md`.
- Advisor results reports done 2026-07-17 (`docs/report/`):
  `results_overview_2026_07_17.docx` (condensed, for the advisor) +
  `results_report_2026_07_17_rev2.docx` (extended justifications /
  defense-prep companion). Overview sent 2026-07-17; advisor asked (2026-07-19)
  whether metrics cover many frames/cars or one car — i.e. test scenarios were
  not explicit. Fixed in `results_overview_2026_07_19.docx`: new "Test scenario
  and evaluation protocol" subsection at top of §2 (seq 08 = 4,071 scans / 393
  distinct cars / ≈34k per-frame car instances, IoU≥0.3 greedy 1-to-1 point-level
  matching, micro-averaged TP/FP/FN; seq 00 100 frames = 41 cars; completion
  §5.3–5.4 unit = 39 static cars, per-car medians). Distinct-car counts from
  label scan (sem ∈ {10,252}, inst>0, ≥10 raw pts).
- Advisor follow-up (2026-07-19): inference time + completion-metric
  explanation. Timing benchmark (`scratchpad/timing_benchmark.py`, Ryzen 7
  7800X3D + RTX 3070 Ti): headline from **full seq 08, all 4,071 frames**
  (`output/experiments/timing/timing_seq08_full_n4068.json`, 3 warmup
  excluded): **921 ms/frame ≈ 1.1 frames/s** — HDBSCAN 502 ms (54%), RANSAC
  163 ms (18%), preprocessing 136 ms, classifier 74 ms (~43 clusters/frame,
  1.7 ms each), tracker 0.3 ms; PCN completion +19 ms per completed car
  (n=12,322; gate rejection 1.5 ms, n=14,823). Scene-density variability:
  ~450-frame block means range ≈770–1,040 ms. Sampling lesson: the
  first-100-contiguous frames understate the mean by 28% (675 ms — sparse
  opening scene); uniform stride-20 sample (201 frames, 934 ms) agreed with
  the full run to 1.4%, so ~100 frames suffice *if drawn across the whole
  drive*. Learned components are cheap; classical CPU stages dominate (72%).
  Both answers added to `results_overview_2026_07_19.docx`: "Inference time"
  section after §1 and "§5.3 Completion evaluation metrics" (Coverage =
  primary quality metric w/ out-of-box hallucination guard; BEV IoU =
  downstream utility; they cross-check each other). Doc restyled 2026-07-19
  from Q&A tone to report tone (question headings removed; completion
  sections renumbered §5.3–§5.5) so it can serve as a school
  submission / paper draft base. Docx and VN reply ready to send (final figure
  921 ms/frame supersedes the ~934 quoted in the earlier chat draft).
- Pipeline diagram for thesis report.

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
