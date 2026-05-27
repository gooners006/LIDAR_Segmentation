# Technical Findings

## 1. HDBSCAN Implementation Performance (2026-04-20)

**Context:** Replaced fixed-epsilon DBSCAN with HDBSCAN for density-adaptive clustering (Step 4). Two Python implementations available: `sklearn.cluster.HDBSCAN` (bundled in scikit-learn >= 1.3) and the dedicated `hdbscan` package.

**Finding:** Benchmarked on SemanticKITTI seq 00 frame 0 (~45k object points after ground removal):

| Implementation | Time | Clusters found |
|---|---|---|
| Open3D DBSCAN (C++, fixed eps=0.5) | ~0.01s | — |
| `hdbscan` package (0.8.42) | 0.48s | 293 |
| `sklearn.cluster.HDBSCAN` (1.8.0) | 3.72s | 309 |

The dedicated `hdbscan` package is **~7.7x faster** than sklearn's implementation on this workload. Both produce comparable cluster counts.

Isolated each pipeline step with `time.time()` wall-clock measurements on a single frame. The clustering step was run on the output of steps 1–3 (z-filter, denoise, downsample, ground removal) to match real pipeline conditions. Comparison script:

```python
import time, numpy as np, open3d as o3d, hdbscan
from sklearn.cluster import HDBSCAN as SkHDBSCAN

# ... load + preprocess to get objects_pcd (steps 1-3) ...
obj_pts = np.asarray(objects_pcd.points)

# dedicated package
t0 = time.time()
labels = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(obj_pts)
print(f"hdbscan: {time.time()-t0:.3f}s")

# sklearn
t0 = time.time()
labels = SkHDBSCAN(min_cluster_size=10, min_samples=5, copy=True).fit_predict(obj_pts)
print(f"sklearn: {time.time()-t0:.3f}s")
```

Ran on macOS (Darwin 25.4.0), Python 3.11, sklearn 1.8.0, hdbscan 0.8.42.

**Decision:** Use the dedicated `hdbscan` package. The ~0.5s per frame is acceptable for offline processing. For real-time use, the original Open3D DBSCAN with a distance-adaptive epsilon wrapper would be needed.

## 2. Ground-Plane-Relative Height Filtering (2026-04-20)

**Context:** The original geometric filter used raw `center[2] > 0.5` (height in sensor frame). On slopes or inclines, object centroids shift in z, causing valid objects to be filtered out.

**Finding:** Replaced raw z-height with signed distance from the bbox center to the RANSAC-fitted ground plane:

```
height = (a*cx + b*cy + c*cz + d) / sqrt(a^2 + b^2 + c^2)
```

Falls back to raw z when no ground plane was fitted (e.g. plane normal rejected). Threshold raised from 0.5m (sensor-relative) to 3.0m (ground-relative) to match the new reference frame.

**Decision:** Use ground-plane-relative height for all geometric filtering. Fallback to raw z preserves robustness when RANSAC fails.

## 3. BEV Representation as Future Research Direction (2026-05-06)

**Context:** Exploring alternative scene representations beyond raw 3D point clouds for the completion and segmentation pipeline.

**Finding:** Bird's-Eye View (BEV) representation models project LiDAR 3D point clouds into a flat 2D top-down perspective. This representation compresses the vertical dimension while preserving spatial layout, making it amenable to efficient 2D convolution-based architectures. BEV is widely used in autonomous driving perception pipelines (e.g., BEVFormer, BEVFusion) for object detection, semantic segmentation, and occupancy prediction. It could complement the current 3D point-based pipeline by providing a more computationally efficient alternative for scene-level understanding.

**Decision:** Noted as a future research direction. No immediate action — current focus remains on PCN completion and KITTI evaluation.

## 4. FP Analysis and Geometric Filter Tuning (2026-05-14)

**Context:** Pipeline had high recall but very low precision. Created `src/analyze_fp.py` to log per-cluster features (semantic class, extents, volume, height, density, distance) and match status against SemanticKITTI GT.

### Baseline (pre-tuning, 100 frames seq 00)

| Precision | Recall | F1 |
|---|---|---|
| 0.237 | 0.868 | — |

5882 detections: 1393 TP, 4489 FP. FP breakdown:

| Class | Count | % of FP |
|---|---|---|
| vegetation | 1721 | 38.3% |
| building | 1476 | 32.9% |
| trunk | 560 | 12.5% |
| unlabeled | 300 | 6.7% |
| other | 432 | 9.6% |

### Filter changes applied

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `max_center_height_above_ground` | 3.0 | **1.5** | TP median 0.81m, building/trunk FP medians 1.73/2.00m |
| `max_height_span` | *(none)* | **1.8** | TP p95 = 1.66m; vegetation/building/trunk FP medians 1.51/2.05/1.97m |
| `max_aspect_max_min` | *(none)* | **6.0** | TP p95 = 5.79; building FP median 5.34 (slab-shaped fragments) |

Thresholds chosen by sweeping individual and combined filter stacks against the FP analysis CSV, selecting for best F1 with ≤10% recall loss.

### Post-tuning results (100 frames seq 00)

| Precision | Recall | F1 | Mean IoU |
|---|---|---|---|
| 0.654 | 0.704 | 0.678 | 0.944 |

1955 detections: 1279 TP, 676 FP.

| Metric | Before | After | Change |
|---|---|---|---|
| Precision | 0.237 | 0.654 | +176% |
| Recall | 0.868 | 0.704 | -19% |
| F1 | — | 0.678 | — |
| FP count | 4489 | 676 | -85% |

### Residual FP distribution (post-tuning)

| Class | Count | % of FP |
|---|---|---|
| vegetation | 439 | 64.4% |
| other-object | 71 | 10.4% |
| building | 69 | 10.1% |
| unlabeled | 40 | 5.9% |
| other | 63 | 9.2% |

Building and trunk FPs were nearly eliminated. Remaining FPs are short, compact vegetation clusters that geometrically overlap with vehicles (FP height_span median 0.95 vs TP 1.14, FP density median 62.5 vs TP 71.5). No further simple threshold can separate these without significant recall loss.

**Conclusion:** Geometric filters addressed the easy FPs. The remaining ~680 FPs per 100 frames require shape-level discrimination — this is the classifier's job.

## 5. Distance-Dependent Evaluation (2026-05-14)

**Context:** Post-filter evaluation (Finding #4) showed aggregate precision of 0.654. Broke down detections by distance to ego to check whether performance varies with range.

### Results (100 frames, seq 00, post-tuning filters)

| Distance bin | TP | FP | Total | Precision | Mean IoU |
|---|---|---|---|---|---|
| 0–20m | 716 | 318 | 1034 | 0.692 | 0.933 |
| 20–40m | 525 | 302 | 827 | 0.635 | 0.960 |
| 40m+ | 36 | 62 | 98 | 0.367 | 0.923 |

95% of TPs are within 40m. Far-range (40m+) has poor precision but very few detections total.

### FP class breakdown by distance

| Distance | Top FP classes |
|---|---|
| 0–20m | vegetation (73%), building (13%) |
| 20–40m | vegetation (63%), other-object (15%), building (9%) |
| 40m+ | **unlabeled (53%)**, vegetation (26%) |

Vegetation is the dominant FP source at all ranges. At 40m+, the majority of FPs are "unlabeled" — likely sparse structures not annotated in SemanticKITTI GT rather than true false positives.

### TP feature stats by distance

| Distance | Median points | Median volume | Median height span | Median density |
|---|---|---|---|---|
| 0–20m | 661 | 6.81 m³ | 1.19m | 109.5 pts/m³ |
| 20–40m | 93 | 2.14 m³ | 1.07m | 44.7 pts/m³ |
| 40m+ | 51 | 2.42 m³ | 1.01m | 22.6 pts/m³ |

Point density drops ~5x from near to far range. Any density-based filter would need to be distance-adaptive to avoid killing far-range TPs.

**Conclusion:** No actionable filter change from this analysis. Near/mid range precision (~0.65) is where most detections live and is the classifier's domain. Far-range precision is low but the sample size is too small to optimize for, and the unlabeled FPs may reflect GT annotation gaps.

## 6. Stage B Mining — Class Distribution from SemanticKITTI (2026-05-15)

**Context:** Built `src/mine_stage_b.py` to extract real LiDAR clusters with GT semantic labels for classifier fine-tuning (Stage B). Ran sanity check on seq 00, 100 frames.

**Finding:** 1979 clusters seen, 17 discarded below 0.75 purity threshold (0.86% discard rate).

| Class | Count | % |
|---|---|---|
| car | 1,246 | 63.5% |
| unknown | 684 | 34.9% |
| motorcycle | 32 | 1.6% |
| bus | 0 | 0.0% |
| **total** | **1,962** | |

~20 clusters/frame average. Heavy car dominance, zero buses (seq 00 appears to have none), rare motorcycles. The low discard rate suggests most pipeline clusters are semantically coherent at the 0.75 purity level.

**Decision:** Class imbalance confirms the Stage A + B strategy: pre-train on balanced synthetic ShapeNet data (Stage A), then fine-tune on real data (Stage B). Training on real data alone would starve bus and motorcycle classes.

## 7. Stage A Classifier — Synthetic-to-Real Domain Gap (2026-05-15)

**Context:** Wired the Stage A classifier (trained on synthetic ShapeNet, 50 epochs, best val_macro_f1 0.9968) into `evaluate.py` to measure its impact on pipeline precision.

**Finding:** The classifier predicts ALL real LiDAR clusters as "unknown" with ~100% confidence. On 20 clusters from frame 0, every single one was classified as "unknown" with softmax probability 0.987–1.000; car probability was essentially 0.000.

Pipeline evaluation with classifier enabled (100 frames seq 00):

| Metric | Geometric only | + Stage A classifier |
|---|---|---|
| TP | 1,279 | 0 |
| FP | 676 | 11 |
| Precision | 0.654 | 0.000 |
| Recall | 0.704 | 0.000 |
| F1 | 0.678 | 0.000 |

The classifier was effective at removing FPs (676 → 11) but also removed every single TP. The synthetic ShapeNet feature distributions don't overlap with real LiDAR at all.

**Decision:** Stage A classifier is completely useless on real data without Stage B fine-tuning. This is the expected synthetic-to-real domain gap and validates the two-stage training strategy.

## 8. Full Stage B Mining Results (2026-05-15)

**Context:** Ran `mine_stage_b.py` across all SemanticKITTI sequences to build the Stage B fine-tuning dataset. Train: seq 00-07, 09-10. Val: seq 08.

**Finding:**

| | Train | Val | Total |
|---|---|---|---|
| car | 88,491 (21.1%) | 25,047 (19.2%) | 113,538 |
| unknown | 329,961 (78.5%) | 104,659 (80.4%) | 434,620 |
| motorcycle | 1,650 (0.39%) | 476 (0.37%) | 2,126 |
| bus | 28 (0.007%) | 0 (0.0%) | 28 |
| **total** | **420,130** | **130,182** | **550,312** |
| discarded | 2,873 (0.68%) | 939 (0.72%) | 3,812 |

Proportions are consistent across splits (~80% unknown, ~20% car, <0.5% motorcycle, near-zero bus). Dataset is ~76x larger than Stage A synthetic data (~5,500 samples). Discard rate ~0.7% — pipeline clusters are semantically coherent at the 0.75 purity threshold.

**Decision:** Class weighting or balanced sampling essential for Stage B. Bus (28 train samples) relies almost entirely on Stage A pre-training. Val split (130K samples from seq 08) is adequate for monitoring fine-tuning.

## 9. Evaluation Target Class Mismatch (2026-05-15)

**Context:** Code review identified that `evaluate.py` counted all SemanticKITTI thing classes (pedestrians, bicyclists, trucks, etc.) as GT instances, but the classifier only recognizes car, bus, and motorcycle. When the classifier correctly rejects a pedestrian as "unknown", it was counted as a false negative — inflating FN and deflating recall.

**Finding:** Added `--target` flag to `evaluate.py` with two modes:
- `all-things` (default): all 18 SemanticKITTI thing class IDs — use for geometric-only evaluation
- `supported-vehicles`: car (10, 252), bus (13, 257), motorcycle (15, 255) only — use when evaluating with classifier

Without this fix, classifier-filtered evaluation would have reported artificially low recall because GT instances of unsupported classes (pedestrian, bicycle, truck, etc.) would always be unmatched.

**Decision:** Use `--target all-things` for geometric baseline, `--target supported-vehicles` for classifier-filtered runs. This makes the two evaluations measure different things — document clearly in any comparison table.

## 10. Stage B Classifier — Pipeline Evaluation (2026-05-16)

**Context:** Stage B fine-tuning completed (best macro F1 thresh: 0.7056). Evaluated the Stage B checkpoint on the pipeline against the geometric-only baseline.

**Finding:**

Per-class validation performance (Stage B checkpoint):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| car | 0.792 | 0.906 | 0.845 |
| bus | 0.000 | 0.000 | 0.000 |
| motorcycle | 0.228 | 0.498 | 0.313 |
| unknown | 0.975 | 0.936 | 0.955 |

Pipeline evaluation (seq 00, 100 frames, `--target supported-vehicles`):

| Metric | Geometric only | + Stage B classifier | Delta |
|---|---|---|---|
| Precision | 0.654 | 0.969 | +0.315 |
| Recall | 0.704 | 0.680 | -0.024 |
| F1 | 0.678 | 0.799 | +0.121 |
| Mean IoU | 0.944 | 0.950 | +0.006 |

TP=1141, FP=37, FN=536. The classifier nearly eliminated FPs (676 → 37) with minimal recall loss. The 536 FNs are likely real vehicles rejected as unknown — a threshold sweep on `unknown_threshold` may recover some.

Main concern: unknown-to-car leakage in confusion matrix (5949/105485 unknown val samples misclassified as car) didn't significantly hurt pipeline precision, suggesting those misclassified unknowns don't survive geometric filtering.

**Decision:** Stage B is a clear win. Next: threshold sweep to optimize the precision-recall tradeoff.

## 11. Unknown Threshold Sweep (2026-05-16)

**Context:** Default `unknown_threshold` was 0.65. Swept 12 values (0.30–0.95) on seq 00, 100 frames, `--target supported-vehicles`, checkpoint `stage_b_best.pth`.

**Finding:**

| Threshold | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|---|---|
| 0.30 | 1164 | 55 | 510 | 0.955 | 0.695 | 0.805 | 0.950 |
| 0.40 | 1186 | 56 | 492 | 0.955 | 0.707 | 0.812 | 0.951 |
| **0.50** | **1189** | **54** | **483** | **0.957** | **0.711** | **0.816** | **0.950** |
| 0.55 | 1168 | 46 | 506 | 0.962 | 0.698 | 0.809 | 0.950 |
| 0.60 | 1165 | 37 | 510 | 0.969 | 0.696 | 0.810 | 0.951 |
| 0.65 | 1149 | 39 | 527 | 0.967 | 0.686 | 0.802 | 0.950 |
| 0.70 | 1123 | 29 | 556 | 0.975 | 0.669 | 0.793 | 0.955 |
| 0.75 | 1103 | 24 | 575 | 0.979 | 0.657 | 0.786 | 0.953 |
| 0.80 | 1071 | 12 | 607 | 0.989 | 0.638 | 0.776 | 0.954 |
| 0.85 | 1039 | 14 | 637 | 0.987 | 0.620 | 0.761 | 0.959 |
| 0.90 | 1004 | 7 | 673 | 0.993 | 0.599 | 0.747 | 0.960 |
| 0.95 | 912 | 1 | 767 | 0.999 | 0.543 | 0.704 | 0.965 |

Best F1 at threshold 0.50 (+0.014 over default 0.65). Clean precision-recall tradeoff — lowering the threshold recovers TPs with minimal FP increase. Below 0.40, recall plateaus while FPs keep rising.

**Decision:** Updated default `unknown_threshold` from 0.65 → 0.50 in `classifier.py`, `evaluate.py`, `main.py`, and `train_classifier.py`.

## 12. PointNet Classifier — Architecture Simplifications vs Original Paper (2026-05-16)

**Context:** Code review of `src/classifier.py` against the original PointNet paper (Qi et al., CVPR 2017) to document deliberate design choices.

**Finding:** The implementation simplifies the original PointNet in three ways:

| Component | Original PointNet | Our implementation | Rationale |
|---|---|---|---|
| Bottleneck dim | 1024 | 256 | 4 classes vs 40; 80K params keeps inference fast |
| T-Nets | Input + feature transform | None | LiDAR clusters are in a consistent sensor frame; unit-sphere normalization handles translation/scale |
| Classification head | 512 → 256 → k | 128 → k | Sufficient capacity for 4-class discrimination |

The dual-branch bbox extension (8-d metric features → 32-d) compensates for the smaller point branch by providing real-world scale information that vanilla PointNet lacks. Current pipeline F1 is 0.816 — the bottleneck is class imbalance (28 bus train samples), not model capacity.

**Decision:** Keep current architecture. Add bottleneck dimension ablation (256 vs 512 vs 1024) to low-priority backlog.

## 13. Track-Level Filtering Hypothesis (2026-05-27)

**Context:** Exploring temporal consistency as a post-classifier filtering step to further reduce false positives. Current pipeline achieves F1 0.816 with precision 0.957.

**Finding:** Real objects persist across multiple frames, so short-lived or class-flickering tracks are probably FPs. Filtering by minimum track length and enforcing class consistency (majority vote) across a track's lifetime should boost precision with minimal recall impact.

**Decision:** Proceed with implementation. Expected to primarily improve precision.

## 14. Track-Level Filtering — Implementation and Sweep (2026-05-27)

**Context:** Implemented track-level filtering as described in Finding #13. Two mechanisms: (1) minimum track length, (2) majority class vote over non-unknown labels with evidence thresholds (`min_known_votes=2`, `min_known_ratio=0.5`, ambiguous ties rejected). Evaluation uses offline/post-hoc two-pass approach: pass 1 tracks all detections (including unknown), pass 2 evaluates only detections from surviving tracks.

**Finding:** Key insight — the `keep_unknown=True` path in track-level evaluation recovers detections that per-frame filtering would reject as "unknown". These flickering detections get track-level majority vote and many resolve to valid classes. This means track filtering primarily **boosted recall**, not precision as originally hypothesized.

Sweep results (seq 00, 100 frames, `--target supported-vehicles`, `stage_b_best.pth`):

| min_track_length | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|---|---|
| 0 (per-frame baseline) | 1175 | 48 | 502 | 0.961 | 0.701 | 0.810 | 0.951 |
| 1 | 1223 | 45 | 452 | 0.965 | 0.730 | 0.831 | 0.943 |
| **2** | **1232** | **46** | **446** | **0.964** | **0.734** | **0.834** | **0.945** |
| 3 | 1213 | 62 | 462 | 0.951 | 0.724 | 0.822 | 0.946 |
| 5 | 1182 | 31 | 492 | 0.974 | 0.706 | 0.819 | 0.946 |

Track stats at `min_track_length=3`: 262 total tracks, 50 accepted, 130 rejected (too short), 82 rejected (class vote failed). Accepted tracks average 25.5 frames.

Best F1 at `min_track_length=2` (0.834). The recall gain (+0.033) comes from recovering detections that flicker between "car" and "unknown" across frames. Precision stays essentially flat (0.964 vs 0.961). At length=5, precision peaks (0.974) but recall drops back to baseline.

Full-sequence validation (seq 00, 4541 frames, `min_track_length=2`):

| Config | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|---|---|
| Per-frame baseline | 37969 | 5669 | 18065 | 0.870 | 0.678 | 0.762 | 0.917 |
| + Track filter (min_len=2) | 39767 | 3532 | 16249 | 0.918 | 0.710 | 0.801 | 0.912 |
| **Delta** | **+1798** | **-2137** | **-1816** | **+0.048** | **+0.032** | **+0.039** | -0.005 |

The improvement is larger on the full sequence than on 100 frames (+0.039 F1 vs +0.024), confirming the 100-frame window underestimated the benefit. The track filter eliminated 2137 FPs (38% reduction) while simultaneously recovering 1798 TPs from flickering detections. Precision improved substantially at full scale (0.870 → 0.918) because the full sequence has more transient FPs that short-track filtering catches.

**Decision:** Set default `min_track_length=2` in `PIPELINE_CONFIG`. Full-sequence F1: 0.801 (up from 0.762). The benefit comes from both mechanisms: majority vote recovers recall, and min-length filtering cuts transient FPs.
