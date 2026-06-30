# Project State

Last updated: 2026-06-30

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
- Stage A: ShapeNet car (02958343) as positive, unknown_fraction=0.50. Best val macro F1: 0.9986
- Stage B: Mined from SemanticKITTI (train: seqs 00-07,09-10; val: seq 08; 5000 frames each, purity 0.75)
  - Train: 420,333 clusters (88,600 car / 331,733 not-car)
  - Val: 130,394 clusters (24,968 car / 105,426 not-car)
- Stage B best: epoch 13/15, macro F1 0.9225 (car P=0.837 R=0.900)
- Checkpoint: `checkpoints/stage_b_best.pth`
- **Stage A ablation done (Finding #25):** advisor's "too perfect synthetic data" concern resolved — from-scratch Stage B matches pretrained (macro F1 0.9285 vs 0.9225). Synthetic prior is redundant given 420k real clusters; pretrained keeps a weak pipeline precision edge. See `docs/classifier/`.

## Eval Metrics

### Seq 00 (deterministic, 100 frames) — headline baseline

| Metric | Value |
|--------|-------|
| Precision | 0.984 |
| Recall | 0.739 |
| F1 | 0.844 |
| Mean IoU | 0.943 |

RANSAC is deterministic (`np.random.default_rng(42)`). Results reproducible across runs.

### Seq 08 (full, 4071 frames) — generalization check (new 2026-06-30)

| Metric | Value |
|--------|-------|
| Precision | 0.913 |
| Recall | 0.693 |
| F1 | 0.788 |
| Mean IoU | 0.895 |

TP=23593 FP=2235 FN=10470. Command: `python src/evaluate.py --seq 08 --frames 5000`.
Confirms the seq-00 story at 40× scale: precision-saturated, recall-limited;
per-frame recall anti-correlated with GT-car density, FP flat ~1/frame.
Figures: `output/seq08_{bev_detections,failure_zooms,timeseries}.png`.

## Recall Bottleneck — Fully Characterized

The ~0.74 recall ceiling is a **hard limit of density-based clustering** on voxelized LiDAR. Extensively investigated across two sessions:

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

**Conclusion:** All clustering-level interventions exhausted. Accept ~0.74 recall and focus on other thesis contributions.

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
trained PCN on it: `checkpoints/pcn_kitti_best.pth`, best val 0.1246.

**Verdict (Finding #26): the data fix WORKED.** In-distribution synthetic eval is
clean (CD 0.16 m, F@0.1m 0.76 — real cars, not blobs). The "blobs on real data"
were **primarily an inference-normalization bug in `completion.py complete()`**:
it applies **3D PCA alignment + partial-radius/partial-centroid** normalization
that the model never saw in training (this breaks *every* PCN checkpoint, incl.
`pcn_best` — 3.5× worse CD even on in-distribution input). A corrected inference
path (no PCA; reorient gravity→Y, length→Z; scale ×1.137; full-car-center
estimate) de-blobs real seq-08 clusters into car-footprint shapes (see
`output/verify_pcn_step2/`). Scripts: `scratchpad/verify_pcn_step1.py` (synthetic,
calibration + ablation), `scratchpad/verify_pcn_step2.py` (real, multi-view + pseudo-GT).

Key sub-findings: scale is solved by the ×1.137 factor; **centroid estimation is
the dominant residual error**; training's `_augment_rotation` is roll-invariance
(about the length axis), not yaw. The static-car pseudo-GT metric is **invalid**
for completion — accumulated LiDAR is itself one-sided, so CD rewards
under-completion (raw partial scored lowest CD on every real example).

## Checkpoints

- `checkpoints/stage_b_best.pth` — binary Stage B classifier (current best)
- `checkpoints/classifier_best.pth` — binary Stage A classifier
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
PoinTr benchmarked → keep PCN (#28). See `docs/findings.md`.

## Immediate Next Steps — Direction 4a: "Does completion improve the box?"

Quantify whether PCN completion yields a bbox closer to amodal GT than the raw
partial. Primary metrics: |ΔL|,|ΔW|,|ΔH|, BEV oriented-box IoU; secondary yaw
error (mod 180°). Hypothesis: completion improves occlusion-truncated dims
(W, far-end L); heading neutral (#27). Detailed plan in `docs/completion/plan.md`.

- **Step 0 — Amodal GT box builder** (`scratchpad/amodal_gt.py`): static cars only
  (sem=10), accumulate instance points across frames via `poses @ Tr`, fit oriented
  box (L-shape fit in X–Z + Y-extent height; frame is Y-down, world up = −Y),
  compute viewpoint-azimuth coverage → `well_observed` flag, cache to
  `output/08/amodal_gt.json`. Validate well-observed static-car count + dim sanity
  (L 3.5–5, W 1.6–2.0, H 1.4–1.6 m). **Start here.**
- **Step 1** (`scratchpad/completion_box_eval.py`): per-frame label-propagated
  detections (reuse `get_frame_detections`), keep TP car clusters matched to
  well-observed static GT; fit raw-partial box and completed box
  (run `completion.complete()`); look up amodal GT box.
- **Step 2**: paired raw-vs-completed-vs-GT metrics; report within-car improvement.
- **Step 3**: result table + 4–6 box-overlay figures (GT black / raw blue /
  completed green); record finding in `docs/findings.md`; update this file.
- **Decision:** completed beats raw → headline "completion adds value"; neutral/worse
  → fix `complete()` geometry (Dir 2) before claiming value. Either way documented.
- **Pseudo-GT trap avoidance:** restrict to static cars + viewpoint-coverage filter
  so amodal W is trustworthy; emphasize L/H/yaw as cleanest. Step 0 infra is reused
  by Direction 1a.

### Deferred
- Thesis writing (pipeline description, experiment results, recall-ceiling discussion).
- Pipeline diagram for thesis report.

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
