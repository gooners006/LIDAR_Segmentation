# Project State

Last updated: 2026-06-03

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Binary Stage B trained |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | PCN abandoned; PoinTr recommended |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics + sweep flags), `src/visualize_gt.py` (GT vs pipeline toggle viz), `src/train_classifier.py`, `src/mine_stage_b.py`, `src/mine_completion_pairs.py`.

## Classifier — Binary (Complete)

- `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- Stage A: ShapeNet car (02958343) as positive, unknown_fraction=0.50. Best val macro F1: 0.9986
- Stage B: Mined from SemanticKITTI (train: seqs 00-07,09-10; val: seq 08; 5000 frames each, purity 0.75)
  - Train: 420,333 clusters (88,600 car / 331,733 not-car)
  - Val: 130,394 clusters (24,968 car / 105,426 not-car)
- Stage B best: epoch 13/15, macro F1 0.9225 (car P=0.837 R=0.900)
- Checkpoint: `checkpoints/stage_b_best.pth`

## Eval Metrics (Binary Classifier, seq 08, 100 frames)

| Run | Precision | Recall | F1 | Mean IoU |
|-----|-----------|--------|----|----------|
| 1 | 0.967 | 0.724 | 0.828 | 0.897 |
| 2 | 0.956 | 0.724 | 0.824 | 0.882 |
| 3 | 0.959 | 0.692 | 0.804 | 0.881 |
| **Mean** | **0.961** | **0.713** | **0.819** | **0.887** |

RANSAC variance: F1 ± 0.024, Recall ± 0.032. Previous 4-class baseline: F1 0.730.

## Parameter Sweep — No Gains

Exhaustive sweep of post-processing and pipeline parameters (all within RANSAC noise):
- Track filter: min_track_length, min_known_votes, min_known_ratio
- Geometric: min_points_in_cluster, hdbscan_min_cluster_size
- Classifier: unknown_threshold (0.30, 0.40)
- Tracker: max_distance (3.0, 4.0), max_disappeared (8, 10)
- RANSAC: distance_threshold (0.15, 0.25)

Recall ceiling (~0.71) is from single-frame clustering — distant/sparse cars never form clusters.

## Completion — PoinTr Recommended

PCN abandoned (findings #15-19). Reviewed three alternatives:
- Occupancy Networks — ruled out (mesh output, not point cloud)
- PoinTr — recommended: KITTI-proven, transformer architecture, open-source
- SnowflakeNet — better CD numbers but no KITTI evaluation

## Checkpoints

- `checkpoints/stage_b_best.pth` — binary Stage B classifier (current best)
- `checkpoints/classifier_best.pth` — binary Stage A classifier
- `checkpoints/pcn_best.pth` — abandoned
- `checkpoints/pcn_lidar_best.pth` — abandoned

## Immediate Next Steps

1. **Fix RANSAC seed** for reproducible evaluation (currently non-deterministic)
2. **Temporal point aggregation** — accumulate multi-frame points before clustering to break recall ceiling
3. **PoinTr implementation** — replace PCN for point completion
4. **Commit current changes** (evaluate.py sweep flags)

## Medium-Term Backlog

5. Pipeline diagram for thesis report
6. Ablation: Stage A-only vs Stage B on real data
7. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
8. Replace global RANSAC with grid-based ground removal
9. Tracker upgrade — IOU-based matching (SORT-style)
