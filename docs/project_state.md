# Project State

Last updated: 2026-05-16

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Stage B trained — F1 0.816 |
| — | Centroid tracker | `src/tracker.py` | Working |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Trained, dormant |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/analyze_fp.py` (FP analysis), `src/train_classifier.py`, `src/mine_stage_b.py` (real cluster mining), `src/train_pcn.py`.

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): Trained (val_cd_fine 0.066, val_fscore 99.37%). Not yet wired into pipeline.
- **Classifier Stage A** (`checkpoints/classifier_best.pth`): Trained (best val_macro_f1_thresh 0.9964 at epoch 37). Useless on real data — predicts everything as "unknown" (domain gap).
- **Classifier Stage B** (`checkpoints/stage_b_best.pth`): Trained (best val_macro_f1_thresh 0.7056). Pipeline F1 0.816 with threshold 0.50.

## Last Eval Metrics (100 frames seq 00)

| Configuration | Precision | Recall | F1 | Mean IoU |
|---|-----------|--------|----|----------|
| Geometric only (`all-things`) | 0.654 | 0.704 | 0.678 | 0.944 |
| + Stage B classifier (`supported-vehicles`, thresh 0.50) | 0.957 | 0.711 | 0.816 | 0.950 |

## Code Quality

Full codebase review completed. All 11 `src/*.py` files reviewed, docstrings on 10/11 files, 2 complexity refactors, 1 bug fix (`analyze_fp.py` stale import), trimesh dependency removed from `train_pcn.py`.

## Immediate Next Steps

1. **Track-level filtering** (min track length, class consistency)
2. **Wire PCN completion into `main.py`**
3. **Qualitative PCN eval** — visualize input/coarse/fine on ShapeNet test samples

## Medium-Term Backlog

4. Pipeline diagram for report
5. Classifier quality reporting (confusion matrix on matched clusters, FP/FN semantic breakdown)
6. Ablation: Stage A-only vs Stage B on real data

## Low Priority

7. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
8. Replace global RANSAC with grid-based ground removal
9. Add EMD loss (Sinkhorn approximation for PCN coarse output)
10. Domain adaptation — fine-tune PCN with `simulate_lidar_noise()` on KITTI data
11. Threshold calibration (sweep 0.3–0.95, select by pipeline F1)
12. Tracker upgrade — IOU-based matching (SORT-style)
13. PointNet bottleneck ablation — 256 vs 512 vs 1024 (Finding #12)
14. Try stronger architectures — PoinTr or SeedFormer if PCN quality is insufficient
15. Explore BEV representation as alternative/complement
