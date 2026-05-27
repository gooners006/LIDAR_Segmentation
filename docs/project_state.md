# Project State

Last updated: 2026-05-27

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Stage B trained — F1 0.816 |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working — F1 0.834 (100fr), 0.801 (full seq) |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Trained, qualitative eval done, dormant |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/analyze_fp.py` (FP analysis), `src/train_classifier.py`, `src/mine_stage_b.py` (real cluster mining), `src/train_pcn.py`, `src/visualize_pcn.py` (PCN qualitative eval).

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): Trained (val_cd_fine 0.066, val_fscore 99.37%). Qualitative eval: mean CD_fine 0.047, F-score 99.2% on val set. Grid-folding artifacts visible in fine output. Not yet wired into pipeline.
- **Classifier Stage A** (`checkpoints/classifier_best.pth`): Trained (best val_macro_f1_thresh 0.9964 at epoch 37). Useless on real data — predicts everything as "unknown" (domain gap).
- **Classifier Stage B** (`checkpoints/stage_b_best.pth`): Trained (best val_macro_f1_thresh 0.7056). Pipeline F1 0.816 with threshold 0.50.

## Last Eval Metrics (seq 00)

| Configuration | Precision | Recall | F1 | Mean IoU | Frames |
|---|-----------|--------|----|----------|--------|
| Geometric only (`all-things`) | 0.654 | 0.704 | 0.678 | 0.944 | 100 |
| + Stage B classifier (`supported-vehicles`) | 0.961 | 0.701 | 0.810 | 0.951 | 100 |
| + Track-level filtering (min_len=2) | 0.964 | 0.734 | 0.834 | 0.945 | 100 |
| + Track-level filtering (min_len=2, full seq) | 0.918 | 0.710 | 0.801 | 0.912 | 4541 |

## Track-Level Filtering

Implemented in `src/main.py` and `src/evaluate.py`. Two mechanisms: min track length + majority class vote. Config in `PIPELINE_CONFIG`:
- `min_track_length: 2`, `track_class_vote: True`, `min_track_known_votes: 2`, `min_track_known_ratio: 0.5`
- Evaluate with: `python src/evaluate.py --seq 00 --frames 100 --target supported-vehicles --classifier-ckpt checkpoints/stage_b_best.pth`
- Disable with: `--no-track-filter`
- Override length: `--min-track-length N`

## Code Quality

Full codebase review completed. All 11 `src/*.py` files reviewed, docstrings on 10/11 files, 2 complexity refactors, 1 bug fix (`analyze_fp.py` stale import), trimesh dependency removed from `train_pcn.py`.

## Immediate Next Steps

1. **Wire PCN completion into `main.py`**
2. **Classifier quality reporting** (confusion matrix on matched clusters, FP/FN semantic breakdown)

## Medium-Term Backlog

3. Pipeline diagram for report
4. Ablation: Stage A-only vs Stage B on real data
5. Consider PoinTr/SeedFormer if PCN visual quality is insufficient

## Low Priority

6. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
7. Replace global RANSAC with grid-based ground removal
8. Add EMD loss (Sinkhorn approximation for PCN coarse output)
9. Domain adaptation — fine-tune PCN with `simulate_lidar_noise()` on KITTI data
10. Threshold calibration (sweep 0.3–0.95, select by pipeline F1)
11. Tracker upgrade — IOU-based matching (SORT-style)
12. PointNet bottleneck ablation — 256 vs 512 vs 1024 (Finding #12)
13. Explore BEV representation as alternative/complement
