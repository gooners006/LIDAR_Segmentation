# Project State

Last updated: 2026-05-15

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Stage B code ready |
| — | Centroid tracker | `src/tracker.py` | Working |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Trained, dormant |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/analyze_fp.py` (FP analysis), `src/train_classifier.py`, `src/mine_stage_b.py` (real cluster mining).

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): Trained (val_cd_fine 0.066, val_fscore 99.37%). Not yet wired into pipeline.
- **Classifier Stage A** (`checkpoints/classifier_best.pth`): Trained (best val_macro_f1_thresh 0.9964 at epoch 37). Useless on real data — predicts everything as "unknown" (domain gap).
- **Classifier Stage B** (`checkpoints/stage_b_best.pth`): **Not yet trained.** Code ready in `train_classifier.py --stage-b`.

## Stage B Mining (complete)

| | Train (seq 00-07,09-10) | Val (seq 08) |
|---|---|---|
| car | 88,491 | 25,047 |
| unknown | 329,961 | 104,659 |
| motorcycle | 1,650 | 476 |
| bus | 28 | 0 |
| **total** | **420,130** | **130,182** |

## Last Eval Metrics (geometric only, 100 frames seq 00)

| Precision | Recall | F1 | Mean IoU |
|-----------|--------|----|----------|
| 0.654 | 0.704 | 0.678 | 0.944 |

## Evaluation Notes

`evaluate.py` now has `--target` flag:
- `all-things`: all SemanticKITTI thing classes (for geometric baseline)
- `supported-vehicles`: car/bus/motorcycle only (for classifier-filtered runs)

Use `supported-vehicles` when evaluating with classifier to avoid penalizing correct rejection of unsupported classes.

## Immediate Next Steps

1. **Run Stage B training:** `python src/train_classifier.py --stage-b --epochs 15`
2. **Evaluate Stage B classifier on pipeline:**
   - `python src/evaluate.py --seq 00 --frames 100 --classifier-ckpt checkpoints/stage_b_best.pth --target supported-vehicles`
   - Compare to geometric-only baseline
3. **Ablation:** Stage A-only vs Stage B on real data
4. **Threshold sweep** on unknown_threshold for best F1

## Medium-Term Backlog

5. Track-level filtering (min track length, class consistency)
6. Wire PCN completion into `main.py`
7. Qualitative PCN eval — visualize input/coarse/fine on ShapeNet test samples
8. Pipeline diagram for report
9. Classifier quality reporting (confusion matrix on matched clusters, FP/FN semantic breakdown)

## Low Priority

10. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
11. Replace global RANSAC with grid-based ground removal
12. Add EMD loss (Sinkhorn approximation for PCN coarse output)
13. Domain adaptation — fine-tune PCN with `simulate_lidar_noise()` on KITTI data
14. Threshold calibration (sweep 0.3–0.95, select by pipeline F1)
15. Tracker upgrade — IOU-based matching (SORT-style)
