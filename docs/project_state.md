# Project State

Last updated: 2026-05-28

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Stage B trained — F1 0.816 |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working — F1 0.834 (100fr), 0.801 (full seq) |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Domain adaptation in progress (Approach B) |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/analyze_fp.py` (FP analysis), `src/train_classifier.py`, `src/mine_stage_b.py` (real cluster mining), `src/train_pcn.py` (PCN training + fine-tuning), `src/visualize_pcn.py` (PCN qualitative eval), `src/show_completion.py` (raw vs completed comparison), `src/test_single_frame_pcn.py` (single-frame PCN test).

## Checkpoints

- **PCN base** (`checkpoints/pcn_best.pth`): Trained on ShapeNet depth renders (val_cd_fine 0.066, val_fscore 99.37%). Produces blobby output on real LiDAR — domain gap confirmed (Finding #15).
- **PCN lidar** (`checkpoints/pcn_lidar_best.pth`): Training in progress — virtual Velodyne ray-casting fine-tuning (Approach B). ETA ~6-7 hours from session start.
- **Classifier Stage A** (`checkpoints/classifier_best.pth`): Trained (best val_macro_f1_thresh 0.9964 at epoch 37). Useless on real data — predicts everything as "unknown" (domain gap).
- **Classifier Stage B** (`checkpoints/stage_b_best.pth`): Trained (best val_macro_f1_thresh 0.7056). Pipeline F1 0.816 with threshold 0.50.

## Last Eval Metrics (seq 00)

| Configuration | Precision | Recall | F1 | Mean IoU | Frames |
|---|-----------|--------|----|----------|--------|
| Geometric only (`all-things`) | 0.654 | 0.704 | 0.678 | 0.944 | 100 |
| + Stage B classifier (`supported-vehicles`) | 0.961 | 0.701 | 0.810 | 0.951 | 100 |
| + Track-level filtering (min_len=2) | 0.964 | 0.734 | 0.834 | 0.945 | 100 |
| + Track-level filtering (min_len=2, full seq) | 0.918 | 0.710 | 0.801 | 0.912 | 4541 |

## PCN Domain Adaptation

### What failed
- **PCA alignment**: No visible improvement — domain gap is in partiality pattern, not rotation.
- **Approach A** (noise augmentation on depth renders): 30 epochs, clean val CD 0.066→0.065. No improvement on real data. Adding noise doesn't change the pinhole partiality pattern.

### In progress
- **Approach B** (virtual Velodyne HDL-64E ray-casting): 64-beam simulation on ShapeNet meshes, 8-50m range, 0.09° resolution. Training command:
  ```
  .venv\Scripts\python.exe src/train_pcn.py --finetune-lidar --pretrained checkpoints/pcn_best.pth --epochs 30 --lr 1e-5
  ```
- Training log: `checkpoints/pcn_lidar_training_log.csv` (rows 1-30 are from Approach A; Approach B rows append after).

### Evaluation plan
- Run: `python src/test_single_frame_pcn.py --pcn-ckpt checkpoints/pcn_lidar_best.pth`
- Compare output images to `output/single_frame_pcn_pcn_best/` baseline.
- Update Finding #16 with results.

## Track-Level Filtering

Config in `PIPELINE_CONFIG`: `min_track_length: 2`, `track_class_vote: True`, `min_track_known_votes: 2`, `min_track_known_ratio: 0.5`.

## Code Quality

Full codebase review completed. All 11 `src/*.py` files reviewed, docstrings on 10/11 files, 2 complexity refactors, 1 bug fix.

## Immediate Next Steps

1. **Evaluate Approach B** — once training completes, run `test_single_frame_pcn.py` with `pcn_lidar_best.pth` and compare to baseline
2. **Update Finding #16** — add Approach B results
3. **Classifier quality reporting** — confusion matrix on matched clusters, FP/FN semantic breakdown

## Medium-Term Backlog

4. Pipeline diagram for report
5. Ablation: Stage A-only vs Stage B on real data
6. Consider PoinTr/SeedFormer if PCN fine-tuning quality is insufficient

## Low Priority

7. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
8. Replace global RANSAC with grid-based ground removal
9. Add EMD loss (Sinkhorn approximation for PCN coarse output)
10. Threshold calibration (sweep 0.3–0.95, select by pipeline F1)
11. Tracker upgrade — IOU-based matching (SORT-style)
12. PointNet bottleneck ablation — 256 vs 512 vs 1024 (Finding #12)
13. Explore BEV representation as alternative/complement
