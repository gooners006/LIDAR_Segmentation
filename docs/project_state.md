# Project State

Last updated: 2026-05-14

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Working |
| 6 | Classification (dual-branch PointNet, 80K params) | `src/classifier.py` | Code done, not trained |
| — | Centroid tracker | `src/tracker.py` | Working |
| 7 | Point completion (PCN, 6.87M params) | `src/pcn.py`, `src/completion.py` | Needs retrain |

Other files: `src/main.py` (pipeline runner), `src/evaluate.py` (metrics vs SemanticKITTI GT), `src/train_pcn.py`, `src/train_classifier.py`.

## Checkpoints

- **PCN** (`checkpoints/pcn_best.pth`): trained 100 epochs, F-Score 0.841. **Invalidated** — camera radius, unit-sphere loss normalization, and seeded split were all fixed after training. Must retrain.
- **Classifier**: no checkpoint exists. Pipeline falls back to heuristic classifier (`classify_bbox_heuristic`).

## Last Eval Metrics (HDBSCAN, pre-classifier, 20 frames)

| Precision | Recall | F1 |
|-----------|--------|----|
| 0.154 | 0.868 | 0.261 |

Precision is the bottleneck — geometric filters need tightening.

## Active Issues

- No trained classifier checkpoint — learned classification is untested.
- PCN checkpoint invalid — completion model unusable until retrained.
- Low precision (0.154) — HDBSCAN produces too many false positive clusters.
- Completion (`completion.py`) is dormant — not wired into `main.py`.

## Immediate Next Steps

1. **Train classifier.** `python src/train_classifier.py`
2. **Retrain PCN.** `python src/train_pcn.py`
3. **Pipeline smoke test.** `python src/main.py --no-gui --save-output --seq 00 --frames 20`
4. **Tighten geometric filters.** Increase `min_volume`, `min_points_in_cluster`, or add aspect-ratio filter to improve precision.

## Medium-Term Backlog

5. Wire PCN completion into `main.py` pipeline loop
6. Stage B classifier eval — real LiDAR negatives from SemanticKITTI
7. Compare learned vs heuristic classifier (`--no-learned-classifier`)
8. Qualitative PCN eval — visualize input/coarse/fine on ShapeNet test samples
9. Evaluate PCN on real KITTI partial point clouds (domain transfer test)
10. Include pipeline diagram in report

## Low Priority

11. Add EMD loss (Sinkhorn approximation for PCN coarse output)
12. Domain adaptation — fine-tune PCN with `simulate_lidar_noise()` on KITTI data
13. Try stronger architectures (PoinTr, SeedFormer) if PCN quality insufficient
14. Explore BEV representation as alternative/complement
15. Tracker upgrade — IOU-based matching (SORT-style)
