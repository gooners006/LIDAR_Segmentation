# Project State

Last updated: 2026-06-02

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 (seq 00, geometric only) |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Revamped to binary — training pending |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | PCN abandoned; exploring Occupancy Networks |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/visualize_gt.py` (GT vs pipeline toggle viz), `src/train_classifier.py`, `src/mine_stage_b.py` (real cluster mining), `src/train_pcn.py` (PCN training), `src/mine_completion_pairs.py` (GT-label pair mining).

## Classifier — Binary Revamp

Previous 4-class model (car/bus/motorcycle/unknown) produced false motorcycle detections and low car recall on real data. Simplified to binary (car / not-car):

- `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- Stage A: ShapeNet car (02958343) as positive, all other categories as negative, `unknown_fraction = 0.50`
- Stage B: SemanticKITTI sem labels 10/252 → car, all else → not-car
- Eval: `THING_CLASSES_SUPPORTED = {10, 252}` (car + moving-car)

**Status:** Code changes complete. Stage A training in progress.

## Checkpoints

- **Classifier Stage B** (`checkpoints/stage_b_best.pth`): **STALE** — 4-class model, will be replaced after binary retraining
- **PCN base** (`checkpoints/pcn_best.pth`): ShapeNet depth renders. Blobby on real LiDAR — abandoned.
- **PCN lidar** (`checkpoints/pcn_lidar_best.pth`): Virtual Velodyne fine-tuning. Also blobby — abandoned.

## Eval Metrics (Pre-Revamp, 4-Class Classifier)

**Held-out seq 08 (100 frames):**

| Configuration | Precision | Recall | F1 | Mean IoU |
|---|-----------|--------|----|----------|
| Geometric only (no classifier) | 0.205 | 0.594 | 0.305 | 0.863 |
| + 4-class Stage B classifier + track filtering | 0.728 | 0.732 | 0.730 | 0.887 |

These metrics will be superseded after binary classifier training.

## PCN — Closed

All synthetic approaches failed (findings #15-17, #19). Sparse-input training (32-256 points) also failed — domain gap is structural. User reviewing Occupancy Networks paper as potential alternative.

## Immediate Next Steps

1. **Complete binary classifier training pipeline:** Stage A → mine Stage B → Stage B fine-tune → evaluate
2. **Evaluate binary classifier** on seq 08: compare against 4-class baseline (F1 0.730)
3. **Discuss Occupancy Networks paper** — potential replacement for PCN completion
4. **Commit all uncommitted changes** (substantial uncommitted work spanning multiple sessions)

## Medium-Term Backlog

5. Pipeline diagram for thesis report
6. Ablation: Stage A-only vs Stage B on real data
7. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
8. Replace global RANSAC with grid-based ground removal
9. Threshold calibration sweep for binary classifier
10. Tracker upgrade — IOU-based matching (SORT-style)
