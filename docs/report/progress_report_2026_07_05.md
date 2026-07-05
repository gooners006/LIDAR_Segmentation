# Progress Report: LiDAR Segmentation & Completion Pipeline

**Ngo Vi Viet Anh — Master's Thesis** · Period covered: May 12 – July 5, 2026

## Summary

Since the last report, the project moved from a geometric-only segmentation pipeline (precision 0.24) to a complete system: learned binary car classifier, track-level filtering, and honest held-out evaluation at full-sequence scale (F1 0.79 on 4,071 frames). The two hardest open questions were both resolved this period: (1) the recall ceiling (~0.74) was root-caused to HDBSCAN cluster splitting and shown to be a hard limit of density-based clustering — all five mitigation strategies were evaluated and documented as negative results; (2) the long-standing "PCN produces blobs on real data" failure turned out to be primarily an **inference-normalization bug**, not a domain-gap failure — after fixing it, the completion network produces recognizable cars on real LiDAR clusters. The project has now pivoted to quantifying whether completion adds measurable downstream value, which is the intended core thesis contribution.

## 1. Learned Classification and Tracking (May–June)

The heuristic bounding-box classifier was replaced with a dual-branch PointNet trained in two stages: Stage A on synthetic ShapeNet renders, Stage B fine-tuned on 420k real clusters mined from SemanticKITTI with GT purity filtering. Key results:

- **Binary reformulation.** The original 4-class design (car/bus/motorcycle/unknown) suffered from extreme class imbalance (28 bus training samples) and produced spurious motorcycle detections. Scope was narrowed to binary car/not-car; Stage B validation macro F1 = **0.92** (car precision 0.84, recall 0.90).
- **Track-level filtering.** A minimum-track-length + majority-class-vote filter over the centroid tracker both cuts transient false positives and recovers detections that flicker between "car" and "unknown" across frames (+0.04 F1 at full-sequence scale).
- **Evaluation integrity.** We identified that seq 00 (previous headline sequence) was inside the Stage B training set. All headline evaluation moved to held-out **seq 08**; earlier seq-00 numbers are retained but marked as leaked.
- **Answering the "too-perfect synthetic data" concern.** A controlled ablation trained Stage B from random init: classifier-level performance is statistically tied with the ShapeNet-pretrained variant (macro F1 0.9285 vs 0.9225). With 420k real clusters, the synthetic prior is redundant — it neither helps nor hurts. No further investment in Stage A realism is warranted.

**Current headline metrics:**

| Evaluation | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|
| Seq 00, 100 frames (leaked, reference) | 0.984 | 0.739 | 0.844 | 0.943 |
| **Seq 08, full 4,071 frames (held-out)** | **0.913** | **0.693** | **0.788** | **0.895** |

The system is precision-saturated (~1 false positive per frame) and recall-limited.

## 2. The Recall Ceiling — Root Cause Established

The ~0.74 recall limit was fully characterized. Per-instance analysis shows **31–37% of GT cars are split across multiple HDBSCAN clusters** (worst for large/close cars: 66% split rate at 0–10 m); the sub-fragments then fail the volume/point-count filters. The recoverable ceiling from single-cluster GT cars is 63–68% — matching actual recall, i.e. the pipeline already recovers essentially every cleanly-clustered car.

Five mitigation strategies were implemented and evaluated; **all were negative on held-out data**: BEV connected-component clustering (merges overlapping objects), larger min-cluster-size (overfits seq 00, F1 drops 0.829→0.801 on seq 08), post-clustering fragment merging (absorbs walls/poles), distance-adaptive HDBSCAN (ring artifacts), and temporal accumulation (F1 collapse). Conclusion for the thesis: the ceiling is a **fundamental limit of density-based clustering without learned object priors** — documented as a characterized finding, and effort was redirected to completion.

## 3. Point-Cloud Completion — Failure Mode Solved

Previous reports documented that PCN trained on ShapeNet produced unusable "blobs" on real LiDAR, resisting three domain-adaptation attempts (noise augmentation, virtual Velodyne ray-casting, sparse-input training). This period resolved the mystery in three steps:

1. **Training data fixed.** A KITTI-like single-view partial generator (voxelized, ground-removed, single-viewpoint — matching real pipeline output) trains a PCN that completes cleanly in-distribution (Chamfer 0.16 m, F-score@0.1m 0.76).
2. **The blobs were an inference bug.** The inference path applied PCA alignment and partial-centroid normalization never seen in training — degrading *every* checkpoint 3.5× even on in-distribution input. A corrected path (canonical reorientation, calibrated scale, full-car-center estimation) de-blobs real seq-08 clusters into car-shaped completions.
3. **Input gating.** Completion quality on real data is bottlenecked by input cleanliness, not model capacity: an L-shape-fit gate that rejects fragments (<2.7 m) and merged clusters (>2.3 m wide) raises plausible-car completion precision from **38% to 69%** with zero loss of good completions.

**Architecture comparison (PoinTr vs PCN).** A transformer completer (PoinTr, 8.9M params) was implemented and trained on identical data. It halves synthetic Chamfer error (0.125 → 0.063) but is **equivalent on real data** (16/26 vs 18/26 plausible completions, differences within threshold noise). The synthetic advantage does not transfer — a clean transfer-gap finding. PCN is retained as production; the result argues that real-data completion is bottlenecked by the residual partiality gap and center estimation, not decoder capacity.

One methodological finding with thesis relevance: **accumulated-LiDAR pseudo-ground-truth is invalid for evaluating completion** — accumulated scans are themselves one-sided, so Chamfer distance rewards *under*-completion (the raw partial always scores best). Valid real-data completion metrics are an open problem we now address directly (below).

## 4. Planned Direction

The thesis narrative is now: *classical segmentation characterized to its limits, plus learned completion with demonstrated downstream value*. Four work packages, in order:

1. **Does completion improve the bounding box?** (in progress) — Compare raw-partial vs completed boxes against amodal GT boxes built from accumulated static cars. Step 0 is done: an amodal GT builder with viewpoint-coverage and motion guards yields 40 well-observed static-car boxes on seq 08 (median L 4.14 / W 1.75 / H 1.47 m — consistent with real car dimensions). Next: paired per-frame evaluation of dimension error and BEV IoU.
2. **A valid real-data completion metric** — donor-frame occluded-side Chamfer: split a static car's observations by ego viewpoint, complete from one side, measure against surfaces only visible from the other. Reuses Step-0 infrastructure.
3. **Improve inference geometry** — center estimation is the dominant residual error (identified by ablation); heading-flip disambiguation secondary.
4. **Real-data fine-tuning** (contingent) — masked-Chamfer fine-tuning on real partials, attempted only if 1–3 show completion value is limited by the remaining train/real gap.

**Decision point:** if completion measurably improves box estimates (WP 1), that is the headline result; if neutral, the geometry fixes (WP 3) come first before re-testing. Either outcome is a documented contribution. Thesis writing (pipeline description, recall-ceiling analysis, completion chapters) begins in parallel once WP 1 concludes.
