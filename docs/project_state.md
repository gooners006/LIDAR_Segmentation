# Project State

Last updated: 2026-06-26

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working (recall ceiling confirmed) |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Binary Stage B trained |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | PCN trained on KITTI-like partials (pcn_kitti_best); real-data quality unverified |

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

## Eval Metrics (Deterministic, seq 00, 100 frames)

| Metric | Value |
|--------|-------|
| Precision | 0.984 |
| Recall | 0.739 |
| F1 | 0.844 |
| Mean IoU | 0.943 |

RANSAC is deterministic (`np.random.default_rng(42)`). Results reproducible across runs.

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

## Completion — KITTI-like PCN trained, verdict pending

Root cause of prior PCN failures (#15-19): synthetic partials were OOD from the
real post-pipeline input (voxelized 0.05 m, ground-removed, single-viewpoint).
Built a KITTI-like single-view partial generator (`_render_kitti_like` in
`src/train_pcn.py`, `--kitti-like`; see `docs/pcn/kitti_like_partial.md`) and
trained PCN on it: `checkpoints/pcn_kitti_best.pth`, best val 0.1246, clean
convergence (plateau ~epoch 55).

**Status: unverified on real data.** A quick `test_single_frame_pcn.py` look on
seq-08 still produced blobs, but on the densest cluster with a single-view render
and no metric — not a valid test. Completion targets SINGLE-FRAME clusters, not
accumulated tracks (those are motion smears).

## Checkpoints

- `checkpoints/stage_b_best.pth` — binary Stage B classifier (current best)
- `checkpoints/classifier_best.pth` — binary Stage A classifier
- `checkpoints/pcn_kitti_best.pth` — PCN on KITTI-like partials (real quality unverified)
- `checkpoints/pcn_best.pth` — prior PCN (blobs on real data, #15-19)

## Immediate Next Steps

1. **Verify KITTI-like PCN on real data** — proper eval: sparse seq-08 clusters
   (40-300 pts), top-down+side views, multiple examples, vs `pcn_best`. Record as
   Finding #26. If still blobs, the data fix failed → reconsider PoinTr. If good →
   wire into `main.py` (single representative frame, not accumulated `all_pts`).
2. **Thesis writing** — pipeline description, experiment results, discussion of recall ceiling
3. **Pipeline diagram** for thesis report

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
