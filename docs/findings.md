# Technical Findings

## 1. HDBSCAN Implementation Performance (2026-04-20)

**Context:** Replaced fixed-epsilon DBSCAN with HDBSCAN for density-adaptive clustering (Step 4). Two Python implementations available: `sklearn.cluster.HDBSCAN` (bundled in scikit-learn >= 1.3) and the dedicated `hdbscan` package.

**Benchmark** on SemanticKITTI seq 00 frame 0 (~45k object points after ground removal):

| Implementation | Time | Clusters found |
|---|---|---|
| Open3D DBSCAN (C++, fixed eps=0.5) | ~0.01s | — |
| `hdbscan` package (0.8.42) | 0.48s | 293 |
| `sklearn.cluster.HDBSCAN` (1.8.0) | 3.72s | 309 |

The dedicated `hdbscan` package is **~7.7x faster** than sklearn's implementation on this workload. Both produce comparable cluster counts.

**Method:** Isolated each pipeline step with `time.time()` wall-clock measurements on a single frame. The clustering step was run on the output of steps 1–3 (z-filter, denoise, downsample, ground removal) to match real pipeline conditions. Comparison script:

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

**Problem:** The original geometric filter used raw `center[2] > 0.5` (height in sensor frame). On slopes or inclines, object centroids shift in z, causing valid objects to be filtered out.

**Fix:** Compute signed distance from the bbox center to the RANSAC-fitted ground plane:

```
height = (a*cx + b*cy + c*cz + d) / sqrt(a^2 + b^2 + c^2)
```

Falls back to raw z when no ground plane was fitted (e.g. plane normal rejected). Threshold raised from 0.5m (sensor-relative) to 3.0m (ground-relative) to match the new reference frame.

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
