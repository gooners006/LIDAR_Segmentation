# Project State

Last updated: 2026-05-14

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned — F1 0.678 |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Training in progress (ShapeNet) |
| — | Centroid tracker | `src/tracker.py` | Working |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Retraining in progress, dormant |

Other files: `src/main.py` (pipeline runner), `src/evaluate.py` (metrics vs SemanticKITTI GT), `src/analyze_fp.py` (FP analysis to CSV), `src/train_pcn.py`, `src/train_classifier.py`.

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): previous checkpoint invalidated (camera radius, unit-sphere loss, seeded split fixes). **Retraining in progress.**
- **Classifier** (`checkpoints/classifier_best.pth`): **Training in progress** on ShapeNet-derived data (Stage A). Once complete: smoke-test, then Stage B fine-tuning on real clusters.

## Last Eval Metrics (post-filter tuning, 100 frames seq 00)

| Precision | Recall | F1 | Mean IoU |
|-----------|--------|----|----------|
| 0.654 | 0.704 | 0.678 | 0.944 |

1955 detections: 1279 TP, 676 FP. Residual FPs: 64% vegetation, 10% other-object, 10% building. No further geometric filter can separate these without significant recall loss — classifier needed.

## Active Issues

- **Residual FPs (~680 per 100 frames)** — short compact vegetation clusters that geometrically overlap with vehicles. Requires shape-level discrimination (classifier).
- Classifier training in progress on synthetic ShapeNet data only — will need real-cluster fine-tuning (Stage B).
- PCN retraining in progress — completion remains dormant until detection/classification stabilizes.
- Completion (`completion.py`) is dormant — not wired into `main.py`.

## In-Progress (started prior to this session)

- **Classifier training** (`python src/train_classifier.py`) — ShapeNet-derived Stage A training. Will produce `checkpoints/classifier_best.pth`.
- **PCN retraining** (`python src/train_pcn.py`) — retraining with fixed normalization/split. Will produce updated `checkpoints/pcn_best.pth`.

## Immediate Next Steps (Precision-First)

1. ~~FP analysis script~~ — done (`src/analyze_fp.py`)
2. ~~Tighten geometric filters~~ — done (prec 0.237→0.654)
3. **Evaluate Stage A classifier** (when training completes). Smoke-test on pipeline, compare learned vs heuristic (`--no-learned-classifier`).
4. **Build Stage B real-cluster dataset** — `src/mine_stage_b.py`. Design ready, write next session. Saves centroid-centered .npy per cluster organized by class (car/bus/motorcycle/unknown).
5. **Fine-tune classifier on real clusters.** Retrain using mined dataset. Evaluate classification accuracy before wiring into pipeline.
6. **Evaluate classifier impact on full pipeline.** Compare proposal-only vs proposal+classifier metrics.

## Medium-Term Backlog

7. Track-level filtering (min track length, class consistency) — only after proposal precision improves
8. Wire PCN completion into `main.py` — only after detection/classification is reliable
9. Qualitative PCN eval — visualize input/coarse/fine on ShapeNet test samples
10. Include pipeline diagram in report

## Low Priority

11. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
12. Replace global RANSAC with grid-based ground removal
13. Add EMD loss (Sinkhorn approximation for PCN coarse output)
14. Domain adaptation — fine-tune PCN with `simulate_lidar_noise()` on KITTI data
15. Try stronger architectures (PoinTr, SeedFormer) if PCN quality insufficient
16. Explore BEV representation as alternative/complement
17. Tracker upgrade — IOU-based matching (SORT-style)
