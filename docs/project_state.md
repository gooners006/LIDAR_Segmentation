# Project State

Last updated: 2026-05-15

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Stage A retraining (ShapeNet unknowns) |
| — | Centroid tracker | `src/tracker.py` | Working |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Trained, dormant |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics), `src/analyze_fp.py` (FP analysis), `src/train_classifier.py`, `src/train_pcn.py`, `src/mine_stage_b.py` (real cluster mining), `src/download_shapenet.py`.

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): Trained (val_cd_fine 0.066, val_fscore 99.37%). Not yet wired into pipeline.
- **Classifier** (`checkpoints/classifier_best.pth`): **Stage A retraining in progress** with real ShapeNet unknowns (52 non-vehicle categories replacing synthetic primitives). ~10-11 hours estimated.

## Last Eval Metrics (post-filter tuning, 100 frames seq 00)

| Precision | Recall | F1 | Mean IoU |
|-----------|--------|----|----------|
| 0.654 | 0.704 | 0.678 | 0.944 |

1955 detections: 1279 TP, 676 FP. Residual FPs: 64% vegetation, 10% other-object, 10% building. Classifier needed for further improvement.

## In-Progress

- **Classifier Stage A retraining** — ShapeNet unknowns (52 real categories). Running.
- **Stage B mining** — full SemanticKITTI train/val extraction running (`mine_stage_b.py`). Train: seq 00-07,09-10; Val: seq 08.

## Immediate Next Steps

1. **When classifier training completes:** Smoke-test Stage A classifier on pipeline. Compare learned vs heuristic (`--no-learned-classifier`).
2. **When mining completes:** Inspect class distribution across all sequences. Write Stage B training/fine-tuning loop in `train_classifier.py`.
3. **Fine-tune classifier on real clusters** (Stage B). Evaluate before wiring into pipeline.
4. **Evaluate classifier impact on full pipeline.** Compare proposal-only vs proposal+classifier metrics.

## Medium-Term Backlog

5. Track-level filtering (min track length, class consistency)
6. Wire PCN completion into `main.py`
7. Qualitative PCN eval — visualize input/coarse/fine on ShapeNet test samples
8. Pipeline diagram for report

## Low Priority

9. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
10. Replace global RANSAC with grid-based ground removal
11. Add EMD loss (Sinkhorn approximation for PCN coarse output)
12. Domain adaptation — fine-tune PCN with `simulate_lidar_noise()` on KITTI data
13. Try stronger architectures (PoinTr, SeedFormer) if PCN quality insufficient
14. Explore BEV representation as alternative/complement
15. Tracker upgrade — IOU-based matching (SORT-style)
