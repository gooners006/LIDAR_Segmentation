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
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Wired into pipeline; domain gap blocks usable output (Finding #15) |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/analyze_fp.py` (FP analysis), `src/train_classifier.py`, `src/mine_stage_b.py` (real cluster mining), `src/train_pcn.py`, `src/visualize_pcn.py` (PCN qualitative eval), `src/show_completion.py` (raw vs completed comparison), `src/test_single_frame_pcn.py` (single-frame PCN test).

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): Trained (val_cd_fine 0.066, val_fscore 99.37%). Wired into pipeline but output is blobby on real LiDAR — ShapeNet-to-LiDAR domain gap confirmed (Finding #15). Needs fine-tuning.
- **Classifier Stage A** (`checkpoints/classifier_best.pth`): Trained (best val_macro_f1_thresh 0.9964 at epoch 37). Useless on real data — predicts everything as "unknown" (domain gap).
- **Classifier Stage B** (`checkpoints/stage_b_best.pth`): Trained (best val_macro_f1_thresh 0.7056). Pipeline F1 0.816 with threshold 0.50.

## Last Eval Metrics (seq 00)

| Configuration | Precision | Recall | F1 | Mean IoU | Frames |
|---|-----------|--------|----|----------|--------|
| Geometric only (`all-things`) | 0.654 | 0.704 | 0.678 | 0.944 | 100 |
| + Stage B classifier (`supported-vehicles`) | 0.961 | 0.701 | 0.810 | 0.951 | 100 |
| + Track-level filtering (min_len=2) | 0.964 | 0.734 | 0.834 | 0.945 | 100 |
| + Track-level filtering (min_len=2, full seq) | 0.918 | 0.710 | 0.801 | 0.912 | 4541 |

## PCN Completion

Wired into `src/main.py` post-accumulation output loop. Config in `PIPELINE_CONFIG`:
- `pcn_min_points: 64`, `pcn_completion_classes: ["car", "bus", "motorcycle"]`, `pcn_sample_seed: 0`
- CLI: `--pcn-ckpt checkpoints/pcn_best.pth`, `--no-completion`
- Completion disabled by default until fine-tuning closes the domain gap.
- Domain gap confirmed: both accumulated-track and single-frame completion produce blobby output (Finding #15).

## Track-Level Filtering

Implemented in `src/main.py` and `src/evaluate.py`. Two mechanisms: min track length + majority class vote. Config in `PIPELINE_CONFIG`:
- `min_track_length: 2`, `track_class_vote: True`, `min_track_known_votes: 2`, `min_track_known_ratio: 0.5`
- Evaluate with: `python src/evaluate.py --seq 00 --frames 100 --target supported-vehicles --classifier-ckpt checkpoints/stage_b_best.pth`
- Disable with: `--no-track-filter`
- Override length: `--min-track-length N`

## Code Quality

Full codebase review completed. All 11 `src/*.py` files reviewed, docstrings on 10/11 files, 2 complexity refactors, 1 bug fix (`analyze_fp.py` stale import), trimesh dependency removed from `train_pcn.py`.

## Immediate Next Steps

1. **Classifier quality reporting** (confusion matrix on matched clusters, FP/FN semantic breakdown)
2. **PCN domain-adaptation fine-tuning** (Stage B for PCN, using `simulate_lidar_noise()` + real mined pairs)

## Medium-Term Backlog

3. Pipeline diagram for report
4. Ablation: Stage A-only vs Stage B on real data
5. Consider PoinTr/SeedFormer if PCN fine-tuning quality is insufficient

## Low Priority

6. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
7. Replace global RANSAC with grid-based ground removal
8. Add EMD loss (Sinkhorn approximation for PCN coarse output)
9. Threshold calibration (sweep 0.3–0.95, select by pipeline F1)
10. Tracker upgrade — IOU-based matching (SORT-style)
11. PointNet bottleneck ablation — 256 vs 512 vs 1024 (Finding #12)
12. Explore BEV representation as alternative/complement
