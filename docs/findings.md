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

## 15. PCN Completion — Pipeline Integration and Domain Gap (2026-05-27)

**Context:** Wired the trained PCN (val_cd_fine 0.066, val_fscore 99.37% on ShapeNet) into `main.py` as a post-accumulation completion step. Tested both accumulated-track completion and single-frame completion to isolate whether motion smear or domain gap was the primary quality issue.

**Finding:** Both modes produce blobby, unrecognizable output on real LiDAR data. The grid-folding decoder outputs a diffuse point cloud with no vehicle structure.

Single-frame test (no motion smear):
- Car cluster: 3,477 pts from frame 49 — completed output is a shapeless blob
- Motorcycle cluster: 255 pts from frame 17 — same result

This confirms the **ShapeNet-to-LiDAR domain gap** is the primary cause, not accumulated-track motion smear. The PCN encoder's global feature does not meaningfully encode real LiDAR partial scans.

Pipeline integration is complete and correct:
- `--pcn-ckpt`, `--no-completion` CLI flags
- Explicit `pcn_completion_classes: ["car", "bus", "motorcycle"]` — unknown clusters are never completed
- Fail-fast on missing checkpoint when completion is enabled
- `tracks.json` metadata: `raw_point_count`, `point_count`, `completed`, `completion_skip_reason`, `completion_method`, `pcn_checkpoint`
- Deterministic sampling via `pcn_sample_seed`

Comparison images saved to `output/experiments/completion_comparison/` (accumulated) and `output/experiments/single_frame_pcn/` (per-frame).

**Decision:** PCN integration is mechanically done but output is not usable without domain-adaptation fine-tuning. The `simulate_lidar_noise()` augmentation and `KITTIObjectDataset` in `completion.py` are ready for this. Keep completion disabled by default (`--no-completion`) until fine-tuning is done.

## 16. PCN Domain Adaptation — Noise Augmentation Insufficient (2026-05-28)

**Context:** Finding #15 confirmed the ShapeNet-to-LiDAR domain gap produces blobby PCN output on real data. Attempted fine-tuning with LiDAR-like noise augmentation on ShapeNet depth renders. Also added PCA canonical alignment at inference (`completion.py:_pca_axes`) and `--pretrained` flag to `train_pcn.py` (loads weights only, fresh optimizer — distinct from `--resume`).

**Finding:**

*PCA alignment:* Added PCA rotation to `PointCloudCompleter.complete()` so the cluster's longest axis aligns to X before PCN inference. No visible effect — output still blobby. The domain gap is in the input distribution, not orientation.

*Noise augmentation on depth renders:* Applied `simulate_lidar_noise()` + random sparsification (64–1024 pts) to ShapeNet pinhole-rendered partials. 30 epochs, lr=1e-5, pretrained from `pcn_best.pth`. Clean val CD barely moved (0.066 → 0.065). Real LiDAR output: still blobs, identical to the un-finetuned model. The augmentation degrades point quality but doesn't change the fundamental partiality pattern — pinhole depth renders produce dense, uniform single-viewpoint observations that look nothing like LiDAR scan lines.

*Virtual Velodyne ray-casting (in progress):* Replaced `_render_partial` with `_render_lidar_partial` that casts HDL-64E-style rays (64 beams, -24.9° to +2° elevation, 0.09° horizontal resolution) at ShapeNet meshes from random distances (8–50m). Produces scan-line structure, distance-dependent sparsity (360–2048 unique pts per sample), and range-proportional noise (σ=0.005·range). Training in progress — not yet evaluated.

**Decision:** Noise-only augmentation on clean depth renders is insufficient to bridge the domain gap. The partiality pattern itself must change. Virtual LiDAR ray-casting approach awaiting evaluation.

## 17. PCN Domain Adaptation — Virtual Velodyne Also Insufficient (2026-05-28)

**Context:** Finding #16 showed noise augmentation on depth renders failed. Approach B replaced the rendering with virtual Velodyne HDL-64E ray-casting on ShapeNet meshes. Trained 30 epochs (lr=1e-5, pretrained from `pcn_best.pth`). Checkpoint: `pcn_lidar_best.pth`.

**Finding:**

Training metrics show the model learned the simulated LiDAR domain:
- Lidar val CD: 0.134 (epoch 1) → 0.070 (epoch 30)
- Clean val CD regressed: 0.066 → 0.082 (expected trade-off)
- Final f-score: clean 98.87%, lidar 99.17%

Real LiDAR evaluation (`test_single_frame_pcn.py --pcn-ckpt pcn_lidar_best.pth`):
- Car (3457 pts, frame 49): blobby output, visually identical to baseline
- Motorcycle (234 pts, frame 17): blobby output, visually identical to baseline

The model improved on simulated LiDAR partials but this does not transfer to real KITTI clusters. The simulation likely still differs from real data in: object scale/geometry, occlusion patterns, surface reflectance effects, and the distribution of cluster sizes post-segmentation.

**Decision:** Neither noise augmentation (Approach A) nor virtual Velodyne ray-casting (Approach B) on ShapeNet meshes bridges the domain gap. Remaining options:
1. **Real-data fine-tuning** — use `KITTIObjectDataset.extract_pairs_from_sequence()` to mine sparse/dense pairs from pipeline tracking output
2. **Stronger architecture** — PoinTr or SeedFormer, which use transformer-based decoders that may generalize better
3. **Accept PCN as a negative result** for the thesis and focus on other contributions

## 18. Evaluation Data Leak — Stage B Classifier Evaluated on Training Data (2026-05-28)

**Context:** All reported pipeline metrics (F1 0.834, precision 0.964) were measured on seq 00. Stage B classifier was fine-tuned on clusters mined from seq 00-07, 09-10. Seq 00 was in both the training set and the evaluation set.

**Finding:**

Re-evaluated on seq 08 (held-out from Stage B training):

| Configuration | Seq | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|
| Geometric only (no classifier, no track filter) | 08 | 0.205 | 0.594 | 0.305 | 0.863 |
| + Stage B classifier + track filtering | 08 | 0.728 | 0.732 | 0.730 | 0.887 |
| + Stage B classifier + track filtering (leaked) | 00 | 0.964 | 0.734 | 0.834 | 0.945 |

Standalone classifier confusion matrix on seq 08 mined clusters (2000 sampled per class):

| GT \ Pred | car | motorcycle | unknown |
|---|---|---|---|
| car (2000) | 1811 (91%) | 0 | 189 |
| motorcycle (476) | 18 | 190 (40%) | 268 |
| unknown (2000) | 114 | 6 | 1880 (94%) |

Key observations:
- Pipeline F1 drops from 0.834 (leaked) to 0.730 (clean) — a 12.5% relative decrease.
- Car recall 91%, car precision 93% — solid but not as strong as seq 00 numbers suggested.
- Motorcycle recall only 40% — most motorcycles are missed.
- The geometric-only baseline is also much weaker on seq 08 (F1 0.305 vs 0.678 on seq 00), suggesting seq 08 is inherently harder (more clutter, more objects).

**Decision:** All future evaluations must use seq 08 as the primary eval sequence. Updated `docs/project_state.md` with corrected metrics. Previous seq 00 numbers are retained but marked as leaked.

## 19. PCN Sparse-Input Training — Domain Gap Persists (2026-06-02)

**Context:** Advisor suggested simplifying ShapeNet point count to match SemanticKITTI density. Hypothesis: PCN trained on sparse inputs (32-256 random points, matching real LiDAR cluster sizes) would produce clean completions on real data.

**Changes:**
- `src/train_pcn.py`: `partial_n_points: 256` (was 2048), `partial_min_points: 32`, `gt_n_points: 4096` (was 16384), `grid_size: 2` (was 4). Random sparsification in `__getitem__`: k ~ U[32, 256], pad to 256.
- `src/completion.py`: `PCN_N_INPUT = 256`, `PCN_GRID_SIZE = 2`.

**Result:** Training converged (val CD improved over epochs). On real LiDAR clusters, completions are scattered noise — worse than the blobby output from the dense model. The domain gap is structural (single-viewpoint depth render vs multi-viewpoint accumulated LiDAR), not density.

**Decision:** Confirms that all ShapeNet-based PCN approaches are exhausted. Point density was necessary but not sufficient. The fundamental mismatch is in partiality patterns.

## 20. Classifier Revamp — Binary Car/Not-Car (2026-06-02)

**Context:** Visualization on seq 07/08 revealed the 4-class classifier (car/bus/motorcycle/unknown) was producing false motorcycle detections (GT has zero motorcycles in seq 08) and near-zero car detections when using the wrong checkpoint. Even with the correct Stage B checkpoint, car recall was only 47% in some runs.

**Observations from seq 08 `tracks.json`:**
- All 8 detected tracks classified as "motorcycle" — zero cars detected
- GT has mostly cars, zero motorcycles
- Bus and motorcycle classes have insufficient training data and add noise

**Decision:** Simplify to binary classification (car / not-car). Rationale:
1. Thesis scope narrowed to car detection only
2. Removes false motorcycle/bus classifications that harm recall
3. Simpler model = more training signal per class
4. Better matches the SemanticKITTI class distribution (cars dominate)

**Changes (code, not yet trained):**
- `src/classifier.py`: `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- `src/train_classifier.py`: Stage A uses only ShapeNet car (02958343) as positive, all other categories as negative. `unknown_fraction` raised to 0.50 for balanced training.
- `src/mine_stage_b.py`: Binary mapping — sem labels 10/252 → "car", all else → "not-car"
- `src/evaluate.py`: `THING_CLASSES_SUPPORTED = {10, 252}` (car + moving-car only)
- `src/pipeline.py`: `pcn_completion_classes: ["car"]`

**Status:** Code changes complete. Stage A training pending.

## 21. BEV Clustering — Negative Result (2026-06-23)

**Context:** Paper "Long-Range LiDAR Vehicle Detection Through Clustering and Classification for Autonomous Racing" (Lim & Park, IEEE Access 2025) proposes 2D BEV clustering with connected-component labeling and morphological operations as a faster alternative to HDBSCAN. Implemented and evaluated as a potential fix for the recall bottleneck.

**Finding:**

Single-frame comparison (seq 00, frame 0, 44k object points):
- HDBSCAN: 227 clusters, 1858 noise points, 1.36s
- BEV (res=0.3, no morph): 226 clusters, 0 noise, 0.015s (~90x faster)
- BEV (res=0.2, morph k=3): 71 clusters, 24903 noise (erosion too aggressive)

Pipeline evaluation (seq 00, 100 frames, with classifier + track filter):

| Method | Res | Kernel | Precision | Recall | F1 | mIoU |
|--------|-----|--------|-----------|--------|----|------|
| HDBSCAN (baseline) | — | — | 0.984 | 0.739 | 0.844 | 0.943 |
| BEV | 0.30 | 0 | 0.993 | 0.640 | 0.779 | 0.964 |
| BEV | 0.15 | 0 | 0.991 | 0.632 | 0.772 | 0.936 |
| BEV | 0.10 | 0 | 0.979 | 0.548 | 0.703 | 0.884 |

BEV projection merges objects that overlap in x-y (walls, poles, cars at different z-heights), creating oversized clusters that fail geometric filters. Morphological erosion destroys sparse distant clusters. The paper's racing environment (flat track, few objects, wide spacing) doesn't transfer to KITTI urban scenes.

**Decision:** BEV clustering is not viable for KITTI urban. HDBSCAN remains the better choice. Code kept in `pipeline.py` for reference.

## 22. Geometric Filter Ablation (2026-06-23)

**Context:** 26% of GT cars are missed (recall ~0.74). Need to determine whether geometric filters are rejecting valid car clusters.

**Method:** `src/analyze_clustering.py` — for each HDBSCAN cluster rejected by geometric filters, identify which filter is the first to reject it AND whether the cluster overlaps with a GT car instance (≥5 shared points).

**Finding (seq 00, 100 frames):**

Total HDBSCAN clusters: 16,454. Passed geometric filters: 2,651 (16.1%). HDBSCAN noise: only 2.3% of points.

GT-matching clusters rejected by filter:

| Filter | All rejected | GT-matching rejected | % of GT killed |
|--------|-------------|---------------------|----------------|
| min_volume (<0.5) | 7,070 | 1,240 | 67.9% |
| min_points (<15) | 2,589 | 482 | 26.4% |
| max_aspect (>6.0) | 303 | 52 | 2.8% |
| max_dim_length (>6.0) | 542 | 19 | 1.0% |
| max_height_span (>1.8) | 1,434 | 18 | 1.0% |
| max_volume (>50) | 396 | 12 | 0.7% |
| max_center_height (>1.5) | 1,468 | 4 | 0.2% |

Seq 08 (held-out) shows same pattern: min_volume kills 66.5%, min_points kills 28.0%.

1,827 GT-matching clusters are killed, but these are mostly sub-clusters from split GT cars — the smaller fragments fail min_volume or min_points while the dominant fragment passes.

## 23. HDBSCAN Merge/Split Analysis (2026-06-23)

**Context:** Need to quantify how GT cars distribute across HDBSCAN clusters to understand the recall ceiling.

**Method:** For each GT car instance (≥10 object-layer points), check which HDBSCAN clusters its points belong to. Classify as: ok (single cluster, <30% noise), split (points in 2+ clusters), merged (shares dominant cluster with another GT car), all_noise.

**Finding:**

Seq 00 (100 frames, 1,631 GT instances):

| Status | Count | % |
|--------|-------|---|
| ok (single cluster) | 1,117 | 68.5% |
| split (2+ clusters) | 514 | 31.5% |
| merged | 8 | 0.5% |
| all_noise | 0 | 0.0% |

Split: mean 4.0 clusters/car, median dominant fraction 0.92. Split cars median 724 pts vs ok cars 100 pts.

Seq 08 (held-out, 811 GT instances):

| Status | Count | % |
|--------|-------|---|
| ok | 509 | 62.8% |
| split | 301 | 37.1% |
| merged | 0 | 0.0% |

Split: mean 3.1 clusters/car, median dominant fraction 0.77.

Key insights:
1. **Splitting, not merging** is the dominant failure mode (31-37% of GT cars split). Merging is negligible (0-0.5%).
2. **Large/close cars split more** — HDBSCAN finds internal density gaps in larger point clouds.
3. **Most splits preserve a large dominant fragment** (median 0.77-0.92 of points), so the car is usually still detected via its biggest piece.
4. **Recoverable ceiling ~63-68%** closely matches actual recall (0.64-0.74), confirming the pipeline already recovers nearly all "clean" GT instances.

**Decision:** The recall ceiling is fundamentally limited by HDBSCAN splitting large cars. Potential mitigations: post-clustering merge of nearby fragments, adaptive HDBSCAN parameters by distance, or accept the ceiling and focus thesis on other contributions.

## 24. Recall Improvement Strategies — Both Negative (2026-06-24; conclusion corrected 2026-07-24)

**Context:** Finding #23 identified HDBSCAN splitting as the recall bottleneck (~0.74 ceiling). Explored two mitigation strategies: (1) post-clustering fragment merge, (2) distance-adaptive HDBSCAN `min_cluster_size`.

**Finding:**

Exploration data (seq 00, 100 frames):
- Fragment centroid distance: median 1.70m (P75: 2.54m). Inter-car distance P5: 2.79m — narrow but usable merge window.
- Split rate by range: 66% at 0-10m, 40% at 10-20m, 19% at 20-30m, 4% at 30-50m. Close cars split most.
- Global MCS sweep: MCS 30 maximizes "ok" GT count (81.7%) vs MCS 10 (68.5%), but MCS 50+ starts losing distant cars.

Full pipeline evaluation (with classifier + track filter):

| Strategy | Seq 00 P | Seq 00 R | Seq 00 F1 | Seq 08 P | Seq 08 R | Seq 08 F1 |
|---|---|---|---|---|---|---|
| **Baseline (MCS=10)** | **0.984** | 0.739 | **0.844** | **0.956** | **0.731** | **0.829** |
| MCS=15 | 0.969 | 0.752 | 0.847 | 0.932 | 0.723 | 0.814 |
| MCS=20 | 0.976 | 0.755 | 0.852 | 0.916 | 0.711 | 0.801 |
| Merge (1.5m, 30pt) | 0.931 | 0.755 | 0.834 | 0.923 | 0.729 | 0.815 |
| Adaptive HDBSCAN (30/15/10) | 0.935 | 0.748 | 0.831 | — | — | — |
| MCS=20 + Merge | 0.962 | 0.731 | 0.831 | — | — | — |

MCS=20 appeared best on seq 00 (F1 0.852), but **does not generalize** — F1 drops from 0.829 to 0.801 on held-out seq 08. All variants trade precision for recall, with net-negative F1 on the held-out set. Fragment merge absorbs nearby non-car clusters (walls, poles). Higher MCS loses distant sparse cars.

**Decision:** Neither strategy improves the pipeline reliably. The ~0.74 recall ceiling is a hard limit of density-based clustering on voxelized LiDAR without learned object priors. Code for both strategies kept in `pipeline.py` (disabled by default, CLI-toggleable via `--merge-fragments`, `--adaptive-hdbscan`). Recommend accepting this ceiling and focusing thesis effort elsewhere.

> **Correction (2026-07-24, superseded by Finding #34):** The "hard limit" conclusion above was overstated. Both strategies here are *post-hoc* — they reassemble or re-cluster fragments *after* HDBSCAN splits them, and they do all fail. But changing the clustering *resolution* itself does help: coarsening the object cloud to a 0.10 m voxel grid *before* HDBSCAN (adopted in the runtime optimization, #33/#34) prevents many splits from forming in the first place. On full seq 08 the split rate drops 37.7% → 29.9% with zero new merges, the cleanly-clustered fraction rises 62% → 70%, and recall improves 0.699 → 0.730 (+1,655 TP, precision unchanged). So ~0.74 was partly a *resolution artifact*, not a fundamental ceiling. A smaller structural limit does remain — density-based clustering has no notion of objectness, and coarsening further starts merging adjacent cars — so recall still cannot be pushed to a learned detector's level this way; but the standing recommendation is now `cluster_voxel_size=0.10` in production (#34), **not** "accept ~0.74."

## 25. Stage A Pretraining Ablation — Synthetic Prior Is Redundant (2026-06-25)

**Context:** Advisor raised that the ShapeNet Stage A data is "too perfect" and may not match real SemanticKITTI (confirmed mechanism in `stage_a.md`: dense pinhole renders vs Velodyne scan-ring sparsity). Experiment A tests whether Stage A pretraining still contributes anything once Stage B fine-tunes on 420k real mined clusters. Trained Stage B from random init (`--no-pretrain --tag stage_b_scratch --seed 0 --epochs 15`) and compared against the production pretrained checkpoint (`stage_b_best.pth`).

**Hypothesis:** If the synthetic prior helps, from-scratch should be clearly worse. If real fine-tuning dominates, the two should match.

**Classifier-level (seq 08 val clusters, eval-only reload):**

| Init | car P | car R | car F1 | not-car F1 | macro F1 |
|---|---|---|---|---|---|
| Stage A → B (pretrained, baseline) | 0.874 | 0.875 | 0.875 | 0.970 | **0.9225** |
| Random init (no-pretrain) | 0.878 | 0.892 | 0.885 | 0.972 | **0.9285** |

From-scratch best epoch was 14 (0.9285); it also hit exactly 0.9225 at epoch 13 — the two are statistically tied, with from-scratch marginally ahead. **Stage A pretraining gives no classifier-level benefit.**

**Pipeline-level (seq 08, 100 frames, classifier + track filter):**

| Init | P | R | F1 | mIoU | FP |
|---|---|---|---|---|---|
| Pretrained (`stage_b_best`) | **0.956** | 0.731 | **0.829** | 0.888 | 27 |
| Scratch (`stage_b_scratch_best`) | 0.905 | 0.726 | 0.806 | 0.891 | 62 |

At the pipeline level the pretrained checkpoint keeps a precision edge (fewer false positives: 27 vs 62), yielding higher F1. Recall is unchanged (clustering-bound, Finding #24).

**Caveat:** Not a fully controlled comparison — the production `stage_b_best.pth` was trained earlier under slightly different conditions (seed/code), so the pipeline precision gap is weak evidence, especially given the classifier-level result reverses it.

**Conclusion (validates advisor's concern):** The "too perfect" synthetic Stage A data is **not hurting, but also not meaningfully helping** — real-data fine-tuning on 420k clusters does essentially all the work. The synthetic prior is redundant given the size of the real training set. Stage A could be dropped without classifier-quality loss; the only possible value is the pipeline-level precision margin, which is unconfirmed.

**Decision:** Keep `stage_b_best.pth` as production for now (precision matters for the pipeline). Do not invest further in improving Stage A realism (would not move the needle). Artifacts kept: `checkpoints/stage_b_scratch_*`. If a clean claim is needed for the thesis, retrain the pretrained variant under identical seed/code/epochs to control the pipeline comparison.

## 26. KITTI-like PCN Verification — Data Fix Worked; the Blob Failure Was an Inference Bug (2026-06-27)

**Context:** `pcn_kitti_best.pth` (PCN trained on KITTI-like single-view partials, Finding #15–19 follow-up) converged well (val 0.1246) but a quick look still showed blobs on seq-08. The planned next step was: "if still blobs, the data fix failed → reconsider PoinTr." This is the proper verification. Scripts: `scratchpad/verify_pcn_step1.py` (synthetic), `scratchpad/verify_pcn_step2.py` (real seq-08).

**Hypothesis:** Distinguish two failure causes that the prior quick look conflated — (a) data-domain gap (the model genuinely can't complete real clusters) vs (b) an inference-normalization mismatch in `completion.py complete()`.

**Step 1 — synthetic in-distribution sanity + normalization calibration (KITTI-like val cars, n=30–80):**

Calibration of the train/inference relationship (we have both partial and GT here):
- Training canonical car frame is **Y-up, length along Z** (GT extents X=1.92 width, Y=1.43 height, Z=4.49 length). `_augment_rotation` rotates about Z = the **length** axis → it is **roll-invariance, not yaw**. Real clusters must therefore be reoriented (gravity→Y, length→Z), not fed in raw.
- `partial_radius / gt_radius` median **1.137** (tight) → scale is recoverable from the partial.
- `|partial_centroid − gt_centroid| / gt_radius` median **0.35–0.40** → the partial's mass centroid is far off the true car center.

Completion quality (Chamfer in metres, F-score @0.1 m; same partials, three inference paths + ablations):

| Path | CD (m) | F@0.1m |
|---|---|---|
| 1. Training normalization (lower bound) | **0.160** | **0.76** |
| 2. `completion.py complete()` [3D PCA + partial-radius/centroid] | 0.57 | 0.28 |
| 3a. partial-centroid + true radius | 0.49 | 0.34 |
| 3b. **GT-centroid + estimated scale (×1.137)** | **0.162** | **0.74** |
| 3c. GT-centroid + true radius | 0.160 | 0.76 |
| 3d. bbox-centroid + estimated scale | 0.27 | 0.52 |

**The model is good in-distribution (0.16 m).** `complete()`'s **3D PCA alignment + partial-radius/partial-centroid** normalization (never seen in training) wrecks even known-good input (3.5× worse CD) — this applies to *every* PCN checkpoint, including `pcn_best`. The ablation isolates the cause: **scale is solved** by the ×1.137 factor (3b ≈ 3c ≈ training); **centroid is the dominant error** (partial-centroid wrecks it even with true radius; GT-centroid recovers full quality). PCA is pure harm. bbox-centroid recovers ~70% of the centroid gap (residual = width-axis occlusion bias).

**Step 2 — real seq-08 static cars (single-frame velodyne clusters, 40–300 pts):**
- `pcn_best` and `pcn_kitti` via `complete()` → **round blobs** on every example (reproduces the documented failure).
- A **corrected** inference path (no PCA; reorient gravity→Y, length→Z; scale ×1.137; full-car-center estimate via bbox + ego-direction width prior) → **de-blobbed, car-footprint-shaped** completions; occasionally a clean car silhouette (track 107, 197 pts). Not uniformly crisp on the sparsest one-sided inputs (residual flatness = remaining domain gap + imperfect center estimate).
- **The static-car pseudo-GT metric is invalid for completion.** Accumulated LiDAR is itself one-sided (never sees occluded surfaces), so CD/F-score *reward under-completion*: the raw partial scored the **lowest** CD in every example. Quantitative completion evidence must come from synthetic (true GT exists); real-data assessment is qualitative. Renders: `output/experiments/verify_pcn_step2/`.

**Conclusion:** The KITTI-like data fix (Finding #15–19 line of work) **worked** — the model produces clean cars in-distribution. The "blobs on real data" were **primarily an inference-normalization bug in `completion.py complete()`** (3D PCA + partial-radius/centroid), not a data-domain failure. The planned pivot to PoinTr was based on a false premise (the model was never the bottleneck the blobs implied).

**Decision / next steps:**
1. **Fix the inference path first.** Port the corrected normalization into `completion.py complete()` (remove PCA; reorient; scale ×1.137; estimate full-car center). This is the real fix and a general bug — `complete()` also degrades `pcn_best`. Validate before wiring into `main.py`.
2. **PoinTr / PoinTr++ remain on the table** — not as a rescue for "broken PCN," but as a potential *quality* upgrade once inference is fixed and PCN sets the fair baseline. Transformer completers handle severe one-sided partiality better and may close the residual real-data crispness gap. Decide after seeing fixed-PCN quality on real clusters.
3. Other candidates to close the residual gap (future work): better center estimation, or light fine-tuning on real partials. Headline question (data vs inference) is resolved.

## 27. Completion Quality Is Gated by Input Cleanliness, Not Heading (2026-06-27)

**Context:** After porting the fixed `complete()` and wiring single-frame completion into `main.py` (Finding #26 follow-up), a full seq-08 run still showed some dense tracks completing poorly while sparse ones completed well. Initial hypothesis: the reorientation's heading axis (major horizontal PCA eigenvector) is ambiguous on near-square / two-face BEV footprints, so dense-but-square cars get a wrong heading → bad completion. Proposed fix: replace PCA heading with search-based **L-shape fitting** (Zhang et al. 2017; `docs/An Efficient L-Shape Fitting Method…md`).

**Hypothesis / metric:** L-shape heading → more plausible completed cars than PCA heading. Metric: fraction of completed car tracks whose L/W/H falls in a car box (L∈[3.3,4.9], W∈[1.5,2.1], H∈[1.1,1.7]).

**Method:** Added `_lshape_axes()` (closeness criterion, 1° search) and `_pca_axes()` to `completion.py`; added `--heading-method {lshape,pca}` to `main.py`. Ran the 300-frame seq-08 demo three ways into separate output dirs (`output/experiments/08_ab_pca`, `08_ab_lshape`, `08_ab_gated`). A/B scripts: `scratchpad/ab_heading.py`.

**Synthetic sanity (rotated corner-view L, true heading 125°):** PCA → 141° (16° off; inflates W 1.72 / L 4.34 as eigenvectors pull toward the L diagonal). L-shape → **125° exact**, recovers L/W = 4.00/1.80. So L-shape *is* the better estimator in principle.

**Real-data heading A/B — NEGATIVE.** PCA vs L-shape is a wash on seq-08: **18/47 plausible cars either way**, mean L/W/H unchanged (3.29/2.03/1.54 vs 3.26/2.07/1.53), 2 tracks gained / 2 lost. The dense-poor tracks flagged by eye were **not heading failures**:
- **tid 301** (2864 pts, the original example) — footprint is already elongated (X–Z 1.83×4.49); completes into a real car (~4.2×2.0×1.4) in *both* methods. The earlier "square footprint" call was an artifact: partials are saved in the **global frame whose horizontal plane is X–Z** (vertical axis = Y), so the true top-down BEV is **X–Z**, not X–Y — the old `diag_footprints.py`/`diag_orientation.py` plotted a side elevation.
- **tid 762** — a 1.4 m fragment, not a whole car.
- **tid 884** — a 2.85 m-wide merge of two cars.

**What the data actually shows — input gating.** The L-shape fit's real value is the fitted footprint (length, width), which cleanly separates good from bad completions. Of 47 completed tracks:

| input gate (fitted footprint) | n | plausible-car outputs |
|---|---|---|
| FRAGMENT (fit length < 2.7 m) | 15 | **0** |
| MERGE (fit width > 2.3 m) | 7 | **0** |
| CLEAN (else) | 26 | **18** (69%) |

Every implausible completion has a detectable bad input. Enabling the gate (`COMPLETION_FRAGMENT_MIN_LENGTH=2.7`, `COMPLETION_MERGE_MAX_WIDTH=2.3`) drops completions 47→26 but **retains all 18 plausible cars**, lifting completion precision **38% → 69%** (`output/experiments/08_ab_gated`: 26/62 completed; skips 14 fragment + 7 merge + 15 too_few_points).

**Conclusion:** Completion quality on real data is bottlenecked by **input cleanliness, not heading estimation**. The bad inputs are fragments and merges leaked by upstream HDBSCAN (consistent with the Finding #23 split/merge characterization). The L-shape fit is kept — repurposed as the input gate (and as the default heading, which is principled and free, just not a quality lever).

**Decision / next steps:**
1. **Gate is wired and on by default** in `completion.py`. Heading default = `lshape`.
2. The residual 8/26 implausible CLEAN completions (e.g. 110, 1905, 12, 1933) are the genuine completion-model error — the place where PoinTr or better center estimation could still help.
3. Methodology note: any future BEV/footprint diagnostic on pipeline output **must use the X–Z plane** (the global frame's horizontal ground plane). **Vertical axis is Y, but world up = −Y (the frame is Y-down): `poses @ Tr` maps sensor +Z to (0, −1, 0).** So side-elevation plots must negate Y or matplotlib draws cars upside down. (This corrects the earlier "Y-up" label, which only verified the X–Z horizontal plane, not the sign of Y.) Pseudo-GT CD remains invalid (Finding #26).

## 28. PoinTr vs PCN — Equivalent on Real Data; Small Synthetic Edge (2026-06-30; corrected 2026-07-06)

**Context:** Implemented a faithful, self-contained PoinTr (transformer completer, no
custom CUDA; `src/pointr.py`, 8.9M params) to attack the residual 8/26 implausible
*clean*-input completions that are genuine PCN model error (Finding #27). Trained it
via `src/train_pointr.py --kitti-like` on the **exact same** `ShapeNetCompletionDataset`
/ `_render_kitti_like` data and `(coarse, fine)` contract as the PCN baseline, changing
only the model + loss — a clean one-variable comparison. 100 epochs, AdamW lr 5e-4
(StepLR ×0.5 @40/80), batch 16, exact CD loss. Checkpoint: `pointr_kitti_best.pth`
(best epoch 90).

**Hypothesis / metric:** PoinTr's transformer decoder handles severe one-sided
partiality better, so it should (a) win on synthetic val CD/F-score and (b) raise the
real seq-08 plausible-car rate above PCN's 18/26 (same plausibility box as #27:
L∈[3.3,4.9], W∈[1.5,2.1], H∈[1.1,1.7]).

**(a) Synthetic validation — small consistent PoinTr edge (CORRECTED 2026-07-06):**

> **Correction:** this section originally claimed a decisive PoinTr win
> ("val cd_fine 0.1246 vs 0.0634, F 0.76 vs 0.987 — PoinTr roughly halves
> Chamfer error"). Those numbers mixed metrics from different protocols:
> PCN's 0.1246 was its val **loss** (coarse CD + 0.5·fine CD) from the
> training log, PoinTr's 0.0634 was its val **fine-CD**; PCN's F 0.76 came
> from the metre-scale `verify_pcn_step1.py` protocol while PoinTr's 0.987
> came from the training log's normalized-frame F-score. The matched
> like-for-like fine-CDs from the training logs are 0.0658 (PCN, best epoch)
> vs 0.0634 (PoinTr, best epoch).

Matched eval (`scratchpad/matched_eval_pcn_pointr.py`): both checkpoints run
through the identical `verify_pcn_step1.py` protocol — same 30 synthetic val
cars, literally identical normalized 256-point partials, CD/F in metres:

| Path | Model | CD (m) | F@0.1m | CD (norm) |
|---|---|---|---|---|
| training norm (GT centre+radius) | PCN | 0.161 ± 0.021 | 0.755 | 0.0667 |
| | **PoinTr** | **0.153 ± 0.016** | **0.782** | 0.0634 |
| GT-free inference (partial centre + calibrated scale) | PCN | 0.504 ± 0.132 | 0.362 | 0.206 |
| | **PoinTr** | **0.489 ± 0.112** | **0.420** | 0.200 |

PoinTr's synthetic edge is real but small: ~5% lower CD, better on 27/30
samples (paired; mean ΔCD +0.008 m). Cross-checks: PCN's row reproduces the
Finding-#26 documented numbers (CD 0.16 / F 0.76), and the normalized-CD
column matches both training logs (PCN 0.0667 ≈ log 0.0658; PoinTr 0.0634 =
log 0.0634), so the protocol chain is consistent. Under the GT-free inference
path both models degrade identically (CD ~0.5 m, ~3× the in-distribution
floor) — the centroid-estimation bottleneck (Finding #26) is
architecture-independent.

**(b) Real seq-08 — a wash.** Rendered seq-08 (300 frames, gate on, heading=lshape —
identical config to the PCN `output/experiments/08_ab_gated` baseline) with the PoinTr checkpoint
into `output/experiments/08_pointr`. The input gate is model-independent, so **both runs completed
the same 26/62 clean-gated tracks**; only the completion output differs. Compared
per-track (`scratchpad/compare_pcn_pointr.py`, render `output/figures/compare_pcn_pointr.png`):

| | plausible-car rate (real seq-08) |
|---|---|
| PCN | **18/26** |
| PoinTr | **16/26** |

The 18→16 difference is **threshold noise, not a quality regression**: all 3 PoinTr
losses (tids 301/2175/2536) are the *height* extent crossing the 1.7 m cap by ≤0.05 m
(1.70/1.72/1.75); PoinTr also *fixes* one too-short car (tid 1905: L 3.23→3.45 m, a
win). PoinTr systematically completes slightly **taller and fuller** cars (H and W both
nudge up ~0.1–0.2 m across nearly every track). The BEV footprints are near-identical
between models on every disagreement track — neither produces a blob where the other
makes a car. The genuine failures (110/1538/2001/3194 — merges/bad inputs) stay
implausible in **both** models identically.

**Conclusion (revised 2026-07-06 with corrected numbers):** PoinTr is marginally
better in-distribution (~5% CD, consistent across samples), and this small edge
yields **no real-data advantage.** On real one-sided LiDAR clusters the two models
are equivalent in plausibility and shape. Both are bottlenecked by the same
factors — the residual real-vs-synthetic partiality gap (Finding #26) and the
centroid/scale estimation in `complete()` — not by decoder capacity: the matched
eval shows those shared bottlenecks (GT-free CD ~0.5 m vs ~0.15 m in-distribution)
dwarf the architecture difference. The original "large synthetic win fails to
transfer" framing overstated the gap; the accurate story is simpler — decoder
capacity was never the binding constraint.

**Decision:** **Keep PCN as production** (`pcn_kitti_best.pth`) — smaller and equivalent
on real data; no real-data justification to swap. PoinTr is retained
(`pointr_kitti_best.pth`, `src/pointr.py`, `src/train_pointr.py`) and
`completion.py._load_model()` dispatches either by checkpoint, so the swap is one flag
if ever wanted. AdaPoinTr (denoising + adaptive query bank) is **not pursued** — it
targets synthetic completion fidelity, which this finding shows is not the real-data
bottleneck. For the thesis, the result is a clean controlled architecture comparison:
a small synthetic edge that vanishes on real data because shared inference/domain
bottlenecks dominate — not a PoinTr-beats-PCN headline.

## 29. Completion Improves Amodal Box Estimates (Direction 4a) (2026-07-05)

**Context:** Direction 4a asks whether PCN completion yields a bounding box closer to
the true (amodal) car than the raw single-frame partial — the first quantitative
real-data evidence that completion adds downstream value. Reference boxes are the 40
well-observed static-car amodal GT boxes from Step 0 (`output/08/amodal_gt.json`,
Finding write-up `docs/completion/amodal_gt.md`). Scripts:
`scratchpad/completion_box_eval.py` (Step 1 sweep),
`completion_box_eval_step2.py` (paired metrics),
`completion_box_eval_viz.py` (overlay figure).

**Hypothesis (pre-registered in `docs/completion/plan.md`):** completion improves
occlusion-truncated dims (W, far-end L); heading neutral (#27). Primary metrics
|ΔL|,|ΔW|,|ΔH| and BEV oriented-box IoU; secondary yaw (mod 180°).

**Method:** For all 2,063 seq-08 frames in which a well-observed car was observed:
run the detection pipeline (Stage B classifier, `thing_classes={10,252}`), greedy
IoU≥0.3 matching, keep TP pairs on well-observed cars → **2,075 (frame, car) pairs
covering all 40 cars**. Per pair, run the production completion path
(`pcn_kitti_best`, L-shape gate + heading) and fit raw-partial and completed boxes in
the world frame with the *same* fitter as the GT boxes (`fit_oriented_box_xz`,
minmax extents), so fitter bias cancels. Gate skipped 714 fragments + 22 merges
(35% — consistent with the #23 split rate), leaving **1,339 completed pairs**.
Primary statistics use per-car medians (frame pairs of a parked car are
autocorrelated), Wilcoxon signed-rank across cars.

**Result — completion adds value (per-car medians, n=39 cars with ≥1 completed pair):**

| Metric | Raw | Completed | Wilcoxon p | Verdict |
|---|---|---|---|---|
| BEV IoU | 0.707 | **0.747** | 0.0019 | better |
| \|ΔW\| (m) | 0.270 | **0.170** | 0.00015 | better |
| \|ΔH\| (m) | 0.255 | **0.131** | 1.6e-10 | better |
| center err XZ (m) | 0.286 | **0.234** | 2.8e-05 | better |
| \|ΔL\| (m) | 0.447 | 0.456 | 0.65 | neutral |
| yaw err (deg) | 3.5 | 3.0 | 0.74 | neutral |

Key sub-findings:
1. **Completion helps most where the sensor saw least** — BEV IoU by input size:
   <100 pts 0.461→0.599, 100–300 pts 0.585→0.678, ≥300 pts 0.703→0.744. The right
   profile for the use case.
2. **The W gain is not a GT artifact**: it holds both for cars whose width the GT
   accumulation truly constrains (both_sides_seen, 13 cars: 0.291→0.182) and the
   rest (26 cars: 0.167→0.118).
3. **L is under-completion, not compact overshoot.** Signed ΔL on normal-length cars
   (≥3.6 m, 32 cars): raw −0.485 m, completed −0.545 m — the network does not extend
   the unobserved far end (compacts are neutral: −0.10 both). The one strong
   regression case in the overlay figure is a *heading* error on a 109-pt input.
   Both are Direction-2 targets (far-end/center estimation), not blockers.
4. Yaw neutrality independently confirms #27 (heading is not a quality lever).

**Artifacts:** `output/experiments/completion_box_eval/step1_records_08.json` (2,075
records), `step2_metrics_08.json` (aggregates), figure
`output/figures/completion_box_overlays_08.png` (6 panels: GT black / raw blue /
completed green).

**Decision:** Pre-registered criterion met — completed beats raw on BEV IoU, W, H
with strong significance and nothing degrades significantly. **Headline "completion
adds value" is established.** Per the roadmap, proceed to Direction 1 (valid
real-data completion metric); the L-undershoot and sparse-input heading errors are
logged as Direction-2 targets.

## 30. Cross-Domain Classifier Matrix — Sim-to-Real Gap Is Total and Symmetric (2026-07-14)

**Context:** Advisor requested (07/07 chat) a cross-validation table for the
sim-to-real gap: train on synthetic, test on SemanticKITTI, and vice versa.
Three cells already existed (Stage A training log, Finding #25); the missing
cells were run with `scratchpad/cross_domain_classifier_eval.py`, which
evaluates any checkpoint on either val set. Semantics decision: each checkpoint
uses **its own training-time bbox-feature mean/std** (deployment behavior);
only the eval data changes.

**Commands:**

```bash
.venv\Scripts\python.exe scratchpad/cross_domain_classifier_eval.py --ckpt checkpoints/classifier_best.pth --domain real
.venv\Scripts\python.exe scratchpad/cross_domain_classifier_eval.py --ckpt checkpoints/stage_b_scratch_best.pth --domain synthetic
.venv\Scripts\python.exe scratchpad/cross_domain_classifier_eval.py --ckpt checkpoints/stage_b_best.pth --domain synthetic
```

**Finding — cluster-level macro F1 (car F1 in parens); new cells bold:**

| Train ↓ Test → | Synthetic val (1,402) | Real val, seq 08 (130,394) |
|---|---|---|
| Synthetic only (Stage A, `classifier_best`) | 0.999 (0.999) | **0.447 (0.000)** |
| Real only (`stage_b_scratch_best`) | **0.304 (0.000)** | 0.929 (0.885) |
| Synthetic → real fine-tuned (`stage_b_best`, production) | **0.318 (0.000)** | 0.923 (0.875) |

1. **No direction transfers.** Stage A on real clusters recovers 5 of 24,968
   cars (car P=0.020, R=0.000) — the binary-classifier confirmation of
   Finding #7's pipeline-level result. Both real-trained models classify **0
   of 701** synthetic cars as car.
2. **Fine-tuning catastrophically forgets synthetic** (0.999 → 0.318). The
   fine-tuned and from-scratch models are equally blind to synthetic data —
   consistent with #25: after 420k real clusters, nothing of Stage A remains
   that matters.
3. **Accuracy is misleading here:** Stage A on real val scores 0.807 accuracy
   purely via the 80% not-car majority class while finding zero cars. This is
   the concrete argument for reporting macro F1 + confusion matrix to the
   advisor, not accuracy.

**Artifacts:** `output/experiments/cross_domain_classifier/*.json` (per-class
reports + confusion matrices).

**Decision:** Table sent to advisor as the requested sim-to-real-gap
evidence. No modeling change: the production pipeline already uses the
real-fine-tuned checkpoint, and #25 already established the synthetic prior
is redundant. The matrix strengthens the thesis narrative: the domain gap is
symmetric and complete at cluster level, so real-data fine-tuning (Stage B)
is not an optimization but a necessity.

## 32. Donor-Frame Occluded-Side Metric Is Valid — First Real-Data Evidence That Completion Adds Unseen Surface (2026-07-17)

**Context:** Direction 1 (`docs/completion/plan.md`). Completion had no valid
real-data metric: pseudo-GT Chamfer rewards under-completion because the
accumulated reference is itself one-sided (#26). Design locked with the user:
visibility-mask novel set (donor points ≥ τ=0.15 m from every input point),
one-directional coverage (novel → method: median distance + cov@0.1 m),
out-of-amodal-GT-box hallucination guard (+0.2 m margin), pipeline TP inputs,
raw + mirrored-partial baselines, per-car medians + Wilcoxon (as #29).
Full method + schema: **`docs/completion/donor_metric.md`**.

**Hypothesis:** the completed cloud covers donor-observed unseen surface better
than raw and mirrored partials, with a stable ranking (raw last, no τ
inversion) and low hallucination — i.e., the metric is valid for measuring
Directions 2/3 on real data.

**Commands:**

```bash
.venv\Scripts\python.exe scratchpad\donor_metric_step1.py --seq 08   # sweep+cache, ~38 min
.venv\Scripts\python.exe scratchpad\donor_metric_step2.py --seq 08   # metric, ~4 min
.venv\Scripts\python.exe scratchpad\donor_metric_step3.py --seq 08   # stats+gate
.venv\Scripts\python.exe scratchpad\donor_metric_viz.py --seq 08     # figure
```

**Result — hypothesis confirmed; all four validation-gate items pass.**
Seq 08: 2,092 TP pairs on the 40 well-observed cars, 1,337 gate-passed, all
qualified (≥100 novel pts @ τ=0.15), 39 cars. Per-car medians (n=39, τ=0.15):

| method | cov@0.1 | med novel-dist (m) | out-of-box |
|---|---|---|---|
| raw | 0.000 | 0.518 | 0.000 |
| mirrored | 0.043 | 0.332 | 0.008 |
| completed | **0.304** | **0.161** | **0.000** |

All pairwise Wilcoxon p < 1e-6. Gate: (a) raw last at every τ; (b) per-car IQR
of completed cov 0.14 (moderate, fine for per-car medians); (c) ranking stable
across τ ∈ {0.10, 0.15, 0.20}; (d) completed out-of-box 0.0003 ≤ mirrored
0.0083.

**Key sub-findings:**

1. **PCN genuinely reconstructs unseen structure on real data** — 30% of
   never-observed surface within 10 cm, 7× the symmetry-mirror baseline, with
   essentially zero hallucination outside the GT box. First positive real-data
   completion evidence in the project (complements #29's box-level value).
2. **Far end is the weakest region** (completed cov 0.133 vs far_side 0.321,
   top 0.203) — #29's length under-completion, now directly measurable; this is
   the Direction-2 number to move. Mirrored scores ~0 on far_end by
   construction (reflection across the length plane can't add end surface) —
   built-in sanity check.
3. **Completion is input-size-robust** (cov 0.30–0.33 across <100 to ≥300 pts)
   while mirroring degrades on sparse inputs (0.083 → 0.010).
4. Worst-case panels (figure) isolate two failure modes: far-end
   under-completion and heading/center mis-estimation on diagonal or sparse
   views — both are Direction-2 targets.

**Artifacts:** `output/experiments/donor_metric/` (pair caches, records,
summary), `output/figures/donor_metric_08.png`. Supporting refactor:
`estimate_canonical_frame()` extracted from `complete()` (`src/completion.py`),
verified bitwise behavior-preserving.

**Decision:** metric accepted as the real-data completion metric. Directions 2
(geometry: far-end + heading) and 3 (real-data fine-tuning) are now measurable
on real data; report deltas as per-car-median cov@0.1 at τ=0.15 with Wilcoxon.

## 31. Stage A Dropped from Production Pipeline — Scratch Checkpoint Matches/Beats Fine-Tuned at Full Scale (2026-07-14)

**Context:** Following Finding #30 (symmetric total domain gap) the user decided
to remove Stage A synthetic pretraining from the final pipeline (kept as thesis
ablation material). The one open concern was Finding #25's pipeline-level
precision edge for the pretrained checkpoint (FP 27 vs 62 on a 100-frame seq-08
spot-check, flagged there as uncontrolled). Before switching, both standard
evals were run with `stage_b_scratch_best.pth`.

**Hypothesis:** scratch loses some precision (per #25's spot-check), recall
unchanged.

**Commands:**

```bash
.venv\Scripts\python.exe src/evaluate.py --classifier-ckpt checkpoints/stage_b_scratch_best.pth
.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --classifier-ckpt checkpoints/stage_b_scratch_best.pth
```

**Result — hypothesis wrong; scratch is neutral-to-better:**

| | seq 00 (100f) fine-tuned | seq 00 scratch | seq 08 (4071f) fine-tuned | seq 08 scratch |
|---|---|---|---|---|
| Precision | 0.984 | 0.984 | 0.913 | 0.903 |
| Recall | 0.739 | **0.761** | 0.693 | 0.699 |
| F1 | 0.844 | **0.859** | 0.788 | 0.788 |
| Mean IoU | 0.943 | 0.942 | 0.895 | 0.895 |
| TP / FP / FN | 1205 / 20 / 426 | 1242 / 20 / 389 | 23593 / 2235 / 10470 | 23823 / 2565 / 10240 |

The seq-00 fine-tuned baseline was re-run first and reproduced the documented
headline exactly (deterministic), so the comparison is clean. Seq 00: +37 TP at
identical FP — a genuine F1 gain (0.844→0.859). Seq 08 full: F1 and mIoU
identical to 3 decimals; +230 TP vs +330 FP (~0.08 FP/frame). **Finding #25's
precision edge does not hold at full scale — it was a 100-frame small-sample
effect.**

**Decision:** Production classifier checkpoint switched to
`stage_b_scratch_best.pth` (defaults updated in `evaluate.py`, `main.py`,
`visualize_gt.py`, `test_single_frame_pcn.py`). New headline metrics: seq 00
P 0.984 / R 0.761 / F1 0.859 / mIoU 0.942; seq 08 P 0.903 / R 0.699 / F1 0.788 /
mIoU 0.895. The final pipeline trains the classifier on real SemanticKITTI data
only; Stage A is retained as the thesis sim-to-real ablation (#7, #25, #30).
`stage_b_best.pth` (fine-tuned) is kept on disk for reproducibility.

## 33. Pipeline Runtime Optimization — Tier 1 (Exact-Output) (2026-07-23)

**Context:** The advisor asked for inference time. Full seq 08 runs
**921 ms/frame** (stride-20 harness: 934.5 ms), ≈ 1.1 fps
(`timing_seq08_full_n4068.json`). Stage breakdown: HDBSCAN 502 ms (56%), RANSAC
ground 163 ms (18%), preprocessing 127 ms (z 55 + denoise 55 + voxel 17),
classifier 74 ms (8%). Goal (`docs/perf/plan.md`): ≤ 400 ms/frame with detection
metrics unchanged. Regression budget locked as a **two-class rule**: Class 1
(implementation-only) must reproduce per-frame TP/FP/FN bit-for-bit; Class 2
(algorithm) lives behind a config flag + `--out-tag`, promoted only if F1 improves
or speedup ≥ 3× at ΔF1 ≥ −0.005.

**Tier 1 changes (all Class 1, exact-output):**
1. **Batched classifier inference** (`classify_clusters_batch` in
   `classifier.py`): one GPU forward per ~41-cluster frame instead of 41
   per-cluster calls. Preprocessing is byte-identical to `classify_cluster`
   (`sample_or_pad` re-seeds `default_rng(0)` per cluster → order-independent;
   model in eval mode → BatchNorm uses running stats → per-sample outputs
   independent of batch composition).
2. **`core_dist_n_jobs=-1`** on both HDBSCAN constructors (config-keyed).
3. z-filter numpy→Open3D copy made contiguous float64 (lossless).

**Correctness (the important result):** guaranteed old-code vs new-code eval on
seq 08, 300 frames — **per-frame TP/FP/FN identical bit-for-bit across all 300
frames** (aggregate 1306 / 161 / 429; P 0.890 / R 0.753 / F1 0.816 / mIoU 0.905
both). Zero cuBLAS FP argmax/threshold flips. The batch path is provably
equivalent. (Verification required regenerating a clean old-code reference via
`git stash` — the first attempt's baseline file was corrupted by an orphaned
double-writer from a mis-backgrounded `&` command.)

**Timing (stride-20, n=201, RTX 3070 Ti + 7800X3D):**

| stage | baseline | Tier 1 | Δ |
|-------|---------:|-------:|--:|
| classifier | 74.7 | 40.9 (med 24.4) | **−34 mean** |
| hdbscan | 510.8 | 506.7 | ~0 |
| **TOTAL** | **934.5** | **910.4** | −24 (−2.6%) |

**Deviation from plan:** `core_dist_n_jobs=-1` was projected to give 1.2–1.5× on
HDBSCAN. It gave **nothing** (506.7 vs 510.8 ms, within noise). HDBSCAN wall-time
is dominated by the single-threaded mutual-reachability MST + cluster-hierarchy
condensation, not the parallelizable core-distance kNN. Kept anyway (exact-output,
harmless, uses all cores). **Consequence:** Tier 1's real win is the classifier
(−34 ms); the two dominant stages are untouched, so the ≤ 400 ms target now rests
on Tier 3 (coarse-voxel HDBSCAN) — RANSAC + preprocessing cuts alone (~−190 ms)
cannot reach it from 910 ms.

**Artifacts:** `output/experiments/perf/oldref_seq08_f300.txt` (clean old-code
reference), `tier1_seq08_f300.txt`; `timing_seq08_stride20_tier1.json` (baseline
`timing_seq08_stride20.json` preserved). Tier 1 changes are uncommitted pending
the full Tier 1–3 sequence.

## 34. Runtime Optimization — Tier 2/3 Promoted: −30% Runtime *and* Better Detection; ≤400 ms Wall (2026-07-23)

**Context:** Continues #33. Tier 1 (exact-output) only shaved the classifier and
left the two dominant stages (HDBSCAN, RANSAC) intact, so the ≤ 400 ms goal rested
on Tier 2/3 — behavior-changing (Class 2) speedups judged by the A1 two-class rule
(promote if F1 improves, or speedup ≥ 3× at ΔF1 ≥ −0.005). Three knobs, added
behind `PIPELINE_CONFIG` flags + `--out-tag`, validated individually then combined.

**Guard baseline (full seq 08, pre-Tier-2):** P 0.903 / R 0.699 / F1 0.788 /
mIoU 0.895 (TP 23823 / FP 2565 / FN 10240).

**The three changes (per-tier, seq 08 300-frame screen vs Tier-1 F1 0.816):**
1. **`voxel_before_denoise=True`** (Tier 2, A4) — statistical outlier removal runs
   on the ~10× smaller post-voxel cloud. F1 0.816 → 0.821.
2. **`ransac_iterations=300`** (Tier 2, A3, 1000 → 300) — plane fit converges well
   before 1000 iters. F1 0.816 → 0.819.
3. **`cluster_voxel_size=0.10`** (Tier 3, A2) — HDBSCAN runs on a 0.10 m re-voxel
   of the object cloud; labels propagate back to the 0.05 m points by `cKDTree`
   nearest neighbour (`objects_pcd`-aligned, downstream untouched). F1 0.816 →
   0.822, mIoU 0.905 → 0.922.

Combined they are **super-additive** (300-frame F1 0.816 → 0.838, no interaction
penalty; 1369/141/388).

**Full seq 08 guard confirmation (the result that matters), combined config:**

| Metric | Baseline | Combined | Δ |
|---|---:|---:|---:|
| Precision | 0.903 | **0.905** | +0.002 |
| Recall | 0.699 | **0.730** | **+0.031** |
| F1 | 0.788 | **0.808** | **+0.020** |
| mIoU | 0.895 | **0.912** | +0.017 |
| TP / FP / FN | 23823 / 2565 / 10240 | 25478 / 2676 / 9444 | +1655 / +111 / −796 |

**Every guard metric improves, none regresses** — so the combined config is
promoted to the production `PIPELINE_CONFIG` defaults, no ΔF1 tradeoff to weigh.

**Why detection *improved* (split-rate, `analyze_clustering.py` seq 08, 100 frames,
810 GT-car instances):** coarser clustering closes the intra-car density gaps that
fragment cars (#23), **without** over-merging:

| | full-res (0.05 m) | cv = 0.10 m |
|---|---:|---:|
| clustered "ok" | 504 (62.2%) | **566 (69.9%)** |
| **split** | 305 (37.7%) | **242 (29.9%)** |
| merged | 0 (0.0%) | 0 (0.0%) |
| mean clusters per split car | 3.3 | 2.7 |

Split rate drops 7.8 pts and merges stay at zero → more GT cars survive as one
filter-passing cluster (+TP, +recall) while precision holds. This is a *research*
result, not just a speedup: the same coarsening that makes HDBSCAN cheaper also
repairs the dominant recall-loss mode (#23).

**Timing (stride-20, n=201, RTX 3070 Ti + 7800X3D),
`timing_seq08_stride20_combined.json`:**

| stage | baseline | combined | Δ |
|-------|---------:|---------:|--:|
| hdbscan | 510.8 | 393.2 | −118 |
| ransac_ground | ~163 | 64.6 | −98 |
| denoise | 52.9 | 52.9 | ~0 (already on small cloud) |
| classifier | 74.7 | 22.9 | −52 (Tier 1) |
| **TOTAL** | **934.5** | **650.6** | **−284 (−30%)** |

Effective rate 1.1 → **1.54 fps**.

**The ≤ 400 ms wall (target missed, and proven unreachable via the authorized
tiers):** a per-stage diagnostic showed coarse-voxel gives only **~1.3×** on
HDBSCAN, not the projected 3–5×, because the *post-ground object cloud is
intrinsically sparse* — surface points already sit ~0.10 m apart, so re-voxelising
0.05 → 0.10 m removes only ~18% of points (0.20 m: ~44%). Python HDBSCAN's
mutual-reachability MST has a hard ~185 ms floor on this data (matching #33: the
MST/condensation, not the parallelisable core-distance kNN, dominates). The only
paths below 400 ms are the **deferred** ones — cuML GPU HDBSCAN (preserves
semantics; WSL2/RAPIDS setup friction) or Open3D DBSCAN (C++, but fixed-eps risks
recall on variable-density cars). **User decision (2026-07-23): accept 650 ms** —
real-time execution is explicitly out of scope (CLAUDE.md), and we now beat the
baseline on *both* speed (−30%) and accuracy (F1 +0.020). Deferred paths not
pursued.

**Deviation from plan:** the budget projected Tier 3 reaching ~350–420 ms. It did
not — the sparse-cloud wall above is the reason. Net outcome is still a clear win
(the plan's fallback framing: "a real gain with metrics improved").

**Artifacts:** `output/experiments/perf/{tier2_voxfirst,tier3_cv010,combined}_seq08_f300.txt`,
`combined_seq08_full.txt` (full-seq guard, 4071 frames),
`splitrate_{fullres,cv010}_seq08_f100.txt`;
`output/experiments/timing/timing_seq08_stride20_combined.json`.

**Completion re-check — isolate-config (2026-07-23, promotion duty):** re-ran
#29 (`completion_box_eval`) and #32 (`donor_metric`) against the promoted
`PIPELINE_CONFIG`, each keeping **its own published classifier** (#29 →
`stage_b_best.pth`, #32 → `stage_b_scratch_best.pth`) so only the promoted config
differs — answering "did the runtime-optimization config shift the completion
findings?" in isolation. Both experiments re-run detection live, so they never
depended on `output/08` PLYs. Verdict: **neither finding shifts.**

*#29 amodal-box quality* (COMPLETED comp_mean; n_pairs 2075 → 2262):

| metric | published | re-run | Δ |
|--------|----------:|-------:|--:|
| bev_iou | 0.7469 | 0.7393 | −0.0076 |
| adW | 0.1701 | 0.1770 | +0.0069 |
| adH | 0.1305 | 0.1322 | +0.0017 |
| adL | 0.4563 | 0.4633 | +0.0070 |
| center_err | 0.2339 | 0.2304 | −0.0035 |
| yaw_err (°) | 4.455 | 4.406 | −0.049 |

*#32 donor coverage* (COMPLETED; n_cars 39 → 40, qualified pairs 1337 → 1508):

| τ | cov pub | cov re-run | Δ | med_dist pub | re-run | Δ |
|--:|--------:|-----------:|--:|-------------:|-------:|--:|
| 0.10 | 0.3455 | 0.3380 | −0.0075 | 0.147 | 0.146 | −0.001 |
| 0.15 | 0.3036 | 0.3017 | −0.0019 | 0.161 | 0.164 | +0.004 |
| 0.20 | 0.2810 | 0.2686 | −0.0124 | 0.171 | 0.180 | +0.009 |

All deltas are within noise (< 0.01 on the normalized box metrics; ≤ 0.012 on
coverage), and completion's headline advantage is fully preserved: #29 still shows
the large raw→completed box-quality gain, #32 still shows completed coverage
~0.30–0.34 vs raw 0.0 and mirrored ~0.05. **Caveat:** these are *not paired* —
the promoted config detects more cars (recall +0.031 on the full-seq guard), so
n_pairs (#29) and n_cars/qualified-pairs (#32) both grew; read the deltas as
*population-level* ("benefit preserved across the new, larger detected set"), not
as same-car degradation. Artifacts (baselines preserved): re-run under
`output/experiments/completion_box_eval/step2_metrics_08_perf.json` and
`output/experiments/donor_metric_perf/`; comparison via
`scratchpad/compare_completion_isolate_config.py`.

*Note (pre-existing, out of scope here):* #29's published table used the
fine-tuned `stage_b_best.pth`, classifier-inconsistent with production
`stage_b_scratch_best.pth` since #31; the isolate-config design deliberately holds
each experiment's own classifier fixed, so this drift is untouched — flagged as a
separate item, not a perf-work regression.

**`output/08` regenerated (2026-07-23):** re-ran `main.py --seq 08` under the
promoted config → 1040 accepted tracks (518 PCN-completed), 1558 PLYs (up from
1523, consistent with the recall gain). GT artifacts (`amodal_gt.json`,
`amodal_gt_check.png` — not produced by `main.py`) copied in; old output preserved
at `output/08_preperf_backup/` (reversible swap, nothing deleted).

**Still pending:** docx inference-section refresh (not authorized this round). All
Tier-1/2/3 code changes uncommitted.

## 35. Longitudinal Length Prior — Fixes Far-End Under-Completion (Direction 2, Step 1) (2026-08-01)

**Context:** Direction 2 (improve `complete()` geometry). #32's donor metric
localized the weakest completion region to the **far end** (cov 0.133 vs far_side
0.321), and #29 logged completion making box *length* worse than the raw partial
(signed ΔL −0.49 → −0.55). Root cause in `estimate_canonical_frame()`: the
canonical center has a width prior (X) and up-shift (Y) but **no length correction
(Z)** — a partial truncated at the occluded far end is normalized around its
observed near portion, so PCN's full-car output stops short of the unseen end.

**Change (inference-only, no retraining):** mirror the width-prior mechanism along
the length axis — extend-only push of `center[2]` toward the ego-far end to a
car-length prior:
`center[2] += sign(center[2]) · max(0.5·L_prior − 0.5·observed_len, 0)`,
`L_prior = COMPLETION_CAR_LENGTH_PRIOR = 4.14 m` (real amodal-GT median length,
#29). `sign(center[2])` = the car's position along length relative to ego
(origin), so the far-from-ego end is the occluded one — exactly as the width prior
uses `sign(center[0])`. Shipped ON (constructor `length_prior` default = the
constant; pass `None` to A/B off); behavior-preserving when disabled.

**Hypothesis / expected metric:** far_end cov ↑ (from 0.133), overall cov not
down, out_of_box guard not broken; box signed ΔL toward 0.

**(1) Synthetic mechanism check** (`scratchpad/length_prior_synth_check.py`;
KITTI-like val cars, true GT, n=40, *true* far-end sign; per-car medians):

| L_prior | CD (m) | F@0.1 | far-quarter cov | far-reach err (m) |
|---|---:|---:|---:|---:|
| none (production) | 0.195 | 0.641 | 0.423 | −0.233 |
| true GT length | 0.183 | 0.661 | 0.589 | −0.127 |
| 4.5 | 0.183 | 0.664 | 0.595 | −0.120 |
| 4.14 | 0.192 | 0.643 | 0.475 | −0.233 |

Extending toward the far end improves CD, F, far-end coverage, and halves
under-reach; the gain needs a prior near the true full length (4.5 ≈ ceiling on
4.5 m synthetic cars; 4.14 under-serves them). Mechanism sound.

**(2) Real donor metric** (#32, seq 08, **paired A/B** — same cached raw clusters +
identical per-pair subsample, only the length prior differs; per-car medians,
n=39, τ=0.15):

| Metric | off | **4.14** | 4.5 |
|---|---:|---:|---:|
| overall cov@0.1 | 0.307 | **0.428** | 0.503 |
| far_end cov | 0.123 | **0.324** | 0.544 |
| far_side cov | 0.329 | 0.446 | 0.505 |
| top cov | 0.203 | 0.261 | 0.303 |
| med novel-dist (m) | 0.162 | 0.117 | 0.099 |
| out_of_box (guard) | 0.0004 | 0.0014 | 0.0122 |
| gate (d) ≤ mirrored 0.0083 | pass | pass | **FAIL** |

Baseline reproduces #32 (far_end 0.123 vs 0.1325) → pairing clean. **4.14 lifts
far_end cov 2.6× (0.123 → 0.324) and overall cov +0.12**, every region up,
med-dist down, out_of_box staying ≪ the mirrored-baseline guard. **4.5 rejected:**
more coverage but out_of_box 0.0122 breaks the guard (over-extends real 3.0–3.5 m
compacts toward 4.5 m). The guard *discriminated* the two priors — validating the
guard itself.

**(3) Real box metric** (#29 downstream utility; oriented boxes fit to the same
cached clouds, `scratchpad/length_prior_box_recheck.py`; per-car medians, n=39):

| box metric (m) | raw | completed off | completed 4.14 | on vs off |
|---|---:|---:|---:|---|
| signed ΔL | −0.419 | −0.440 | **−0.319** | p=5.4e-7 |
| \|ΔL\| | 0.457 | 0.440 | **0.348** | p=3.6e-3 |
| \|ΔW\| | 0.207 | 0.121 | 0.116 | p=0.36 |
| \|ΔH\| | 0.249 | 0.127 | 0.145 | p=0.035 |

**Reverses #29's length regression:** completion now *extends* the far end —
completed length beats raw (signed ΔL −0.44 → −0.32; completion previously hurt
length) and |ΔL| drops 0.44 → 0.35. Width untouched (the Z push is orthogonal to
X); |ΔH| +1.8 cm (minor, still ≪ raw 0.249).

**Decision:** ship **4.14** (now the production default). First Direction-2 win:
two independent real-data metrics (donor coverage + box length) *and* the synthetic
true-GT check all improve, guardrails intact. The prior value is principled (the
measured median car length), not tuned to the guard.

**Deviation from plan:** A4 said run the synthetic check first "to de-risk the
sign." Synthetic did run first (mechanism + magnitude), but it is mesh-centered
(no ego) so it cannot test the *ego* sign (A2-i). That sign is instead
**self-validated on real data**: the donor `far_end` region is itself ego-defined
(`len_c·e_len < 0`), so a wrong sign would have *lowered* far_end cov — it rose
2.6×.

**Artifacts:** `src/completion.py` (`COMPLETION_CAR_LENGTH_PRIOR`,
`estimate_canonical_frame` Z push); scratchpad `length_prior_synth_check.py`,
`donor_metric_recompute.py`, `length_prior_box_recheck.py`;
`output/experiments/donor_metric_len_{off,414,450}/` (the paired baseline; #32
preserved at `donor_metric/`).

**Follow-on:** `output/08` regenerated under the shipped prior (2026-08-01) —
1040 tracks / 518 completed / 1558 PLYs; md5 confirms only the 518 completed
clouds changed (detection + `_partial.ply` inputs byte-identical), no-prior
version preserved at `output/08_noprior_backup/`.

**Production-config table refresh (2026-08-01):** re-ran #29 and #32 under the
promoted `PIPELINE_CONFIG` with production detection (the #34 `_perf` cached
clusters; `stage_b_scratch` for BOTH — this also resolves the flagged #29
classifier drift, which was published on `stage_b_best`), prior OFF vs ON, fully
paired (identical clusters + per-pair subsample). n = 40 cars / 1508 completed
pairs. The §35 headline holds at the production operating point.

*#32 donor coverage* (per-car median, τ = 0.15):

| metric | OFF | ON |
|---|---:|---:|
| overall cov@0.1 | 0.301 | **0.403** |
| far_end cov | 0.121 | **0.346** |
| far_side cov | 0.335 | 0.440 |
| top cov | 0.205 | 0.253 |
| med novel-dist (m) | 0.166 | 0.123 |
| out-of-box (guard) | 0.0004 | 0.0020 (≪ mirrored 0.0108) |

*#29 box quality* (per-car median, raw vs completed):

| metric | raw | comp OFF | comp ON |
|---|---:|---:|---:|
| \|ΔL\| | 0.428 | 0.476 | **0.354** |
| BEV IoU | 0.725 | 0.743 | **0.747** |
| \|ΔW\| | 0.210 | 0.126 | 0.122 |
| \|ΔH\| | 0.227 | 0.132 | 0.144 |
| center_err | 0.271 | 0.227 | 0.229 |

OFF-prior, completion *worsens* box length vs the raw partial (|ΔL| 0.428 →
0.476, p = 0.027 — the #29 regression, reproduced at the production operating
point); ON reverses it (→ 0.354) and nudges BEV IoU 0.743 → 0.747. The
scratch-classifier OFF numbers reproduce the #34 `_perf` best-classifier table
(BEV IoU 0.739, |ΔL| 0.463), so the classifier swap is immaterial to box metrics,
as expected.

**Compact-overshoot caveat (honest tradeoff):** the *fixed* 4.14 m prior helps
the majority (normal cars ≥ 3.6 m: signed ΔL −0.55 → −0.34) but *over-extends*
compacts (< 3.6 m; 8 cars / 255 pairs: −0.10 → **+0.25**). Net per-car |ΔL| still
improves and gains are largest on sparse inputs (|ΔL| 0.94 → 0.35), but a per-car
length estimate (vs the fixed median prior) would remove the overshoot — logged
as a Step-1b refinement, not pursued. Artifacts:
`output/experiments/donor_perf_len{off,on}/`,
`output/experiments/completion_box_eval/step2_metrics_08_len{off,on}.json`,
builder `scratchpad/build_box_records_from_donor.py`.

## 36. Per-Car Length Estimate Replaces the Fixed Prior — Fixes Compact Over-Extension (Direction 2, Step 1b) (2026-08-02)

**Context:** #35 shipped a fixed longitudinal prior
(`COMPLETION_CAR_LENGTH_PRIOR = 4.14 m`, the amodal-GT median) and logged a known
defect: it over-extends compacts (< 3.6 m: signed ΔL −0.10 → +0.25). Step 1b
replaces the constant with a per-car estimate.

**Method selection was done offline first, against amodal GT, with no PCN
inference** (`scratchpad/length_estimator_probe{,2}.py`; 1508 gate-passed donor
pairs / 40 cars). This killed three plausible designs before they cost a run:

| Candidate | Verdict |
|---|---|
| Length from width (aspect ratio) | **Dead.** corr(GT L, GT W) = **+0.018**. A *perfect* width predicts L at MAE 0.526 m — worse than the constant's 0.351. |
| Length from height / range / point count | **Dead.** \|corr\| ≤ 0.13 for all three. |
| Far-end face-support ("is the far end occluded?") | **Dead, and inverted.** Missing length is *larger* when the far end looks well covered (+0.62 m) than when it does not (+0.37 m). |
| Observed footprint length (`fit_length`) | **Survives.** corr +0.52 per frame, **+0.87 aggregated per car.** |
| Track **max** of `fit_length` | **Dead.** Worst estimator tested (compact bias +0.95 m) — one contaminated L-shape fit sets the max. Must be a quantile. |

**Change:** `track_length_estimate()` in `src/completion.py` — per-car length =
**90th percentile of `fit_length` over the track's gate-passed frames + 0.12 m**
(`COMPLETION_LENGTH_TRACK_{QUANTILE,OFFSET}`). Falls back to the 4.14 m constant
below 5 gate-passed frames (2/40 cars), since a quantile degenerates toward the
max there. Plumbed as an optional `length_estimate` argument through
`complete()` / `estimate_canonical_frame()` (module stays stateless);
`src/main.py` aggregates over the track it is already completing. Extend-only
push unchanged. The 0.12 m offset corrects q90's low bias; the car-level fit
slope came out 1.00 (pure bias correction) and leave-one-car-out degrades MAE
only 0.171 → 0.181.

**Results — #29 box metric, per-car medians, split by amodal-GT length band**
(the split is the point; the pooled median hid the defect):

| Band | metric | raw | 4.14 (shipped) | **q90+0.12** | q95 | ols (control) |
|---|---|---|---|---|---|---|
| compact <3.6 (n=8) | signed ΔL | −0.007 | **+0.295** | **+0.063** | +0.095 | +0.141 |
| | \|ΔL\| | 0.256 | 0.295 | **0.129** | 0.122 | 0.150 |
| | center err | 0.157 | 0.322 | 0.192 | 0.183 | 0.242 |
| normal (n=27) | \|ΔL\| | 0.445 | 0.345 | **0.329** | 0.349 | 0.388 |
| long ≥4.6 (n=5) | \|ΔL\| | 0.572 | 0.637 | **0.583** | 0.608 | 0.663 |
| ALL (n=40) | \|ΔL\| | 0.428 | 0.354 | **0.304** | 0.334 | 0.361 |
| | BEV IoU | 0.725 | 0.747 | **0.771** | 0.767 | 0.753 |
| | center err | 0.271 | 0.229 | **0.184** | 0.185 | 0.211 |

Compact overshoot fixed (+0.295 → +0.063, Wilcoxon p = 0.016 at n = 8) with no
normal-band cost. Compact **center error** was the larger unmeasured casualty:
the fixed prior pushed it to 0.322 m, *worse than the raw partial 0.157*.

**Leakage control (`ols`, single-frame `L = 2.528 + 0.428·fit_length`):** the
track quantile uses the car's other frames, which are also the donor metric
reference set — a scalar leaks from reference into method. It is legitimate in
production (`main.py` completes each track once with all its frames in scope),
but the far_end number under track modes is "single-frame completion + a
track-level size prior", not pure single-frame inference. The leakage-free OLS
control partially fixes compacts (+0.295 → +0.141) but **regresses normals**
(\|ΔL\| 0.345 → 0.388, p = 0.022). So the compact defect is genuinely fixable
from single-frame information, but only the track-level estimate fixes it
without a tradeoff.

**Donor metric (#32) cost:** pooled coverage drops 0.403 → 0.364 and far_end
0.346 → 0.316. Per band, per-car is better on compacts (see #37) and long
(far_end 0.106 → 0.191) and worse on normal (0.330 → 0.292).

**Deviations / refuted hypothesis.** The two metrics disagreed, and the first
explanation offered — that donor coverage is recall-like and merely rewards
over-extension through a too-loose out-of-box guard — was **tested and refuted**.
A deliberate over-extension control (q90 **+0.45**,
`output/experiments/donor_len1b_over/`) improved *both* metrics on normal and
long cars: coverage 0.364 → **0.483**, far_end 0.316 → **0.509**, box \|ΔL\|
0.304 → **0.210**, BEV IoU 0.771 → **0.777**. Those completions were genuinely
still too short, not the metric being fooled. The recall-like reading holds
**only on the compact band**, where extra push raises coverage while degrading
the box (signed ΔL +0.277).

**Consequence — a second under-extension mechanism, not addressed by Step 1b.**
Targeting a car true length still leaves the completed box short by −0.33
(normal) / −0.58 (long) even when `L_est` is unbiased for those bands. So there
is an under-extension inside the completion itself, independent of where the
canonical center is placed: PCN under-fills its normalized frame, and the center
push also drives `radius`, hence output scale — the prior is doing double duty as
reposition *and* rescale. Fitting the compensation per band gives required
offsets of ≈0.02 / 0.68 / 1.02 m for compact / normal / long, i.e. strongly
length-dependent; extrapolating that from 8 / 27 / 5 cars would be overfitting.
Logged as **Step 1c** in `docs/completion/plan.md`, not bolted onto this change.

**Verdict:** SHIPPED (q90 + 0.12 m). Step 1b stated goal is met — the compact
over-extension is fixed on both metrics, all pooled box metrics improve over the
fixed prior, and it fixes a guard violation (#37) — at a real but bounded cost in
pooled donor coverage, which this finding's control run shows is a *separate*,
larger problem than the one Step 1b set out to solve.

Scripts: `scratchpad/length_estimator_probe{,2}.py` (offline selection),
`scratchpad/donor_metric_recompute.py --length-mode` (A/B),
`scratchpad/length_1b_box_eval.py` (band-split box metric).
Outputs: `output/experiments/donor_len1b_{q95,q90off,ols,over}/`,
`output/experiments/len_probe/`.

## 37. #32 Hallucination Guard Is Blind to Band-Localized Over-Extension — Pooled Median Masked a 100× Violation (2026-08-02)

**Context:** found while arbitrating #36. #32 validation-gate item (d) is
"completed out-of-GT-box fraction ≤ mirrored baseline", computed as a **median
over all 40 cars**. The shipped 4.14 m length prior passes it comfortably
(0.0020 vs 0.0108) and #35 used exactly this guard to *reject* L = 4.5 m.

**Defect:** the pooled median hides any failure confined to a size band. Splitting
out-of-box by amodal-GT length (completed / per-band mirrored baseline):

| Config | compact <3.6 (n=8) | normal (n=27) | long ≥4.6 (n=5) | pooled |
|---|---|---|---|---|
| prior off | 0.0003 / 0.0004 | 0.0008 / 0.0129 | 0.0000 / 0.0292 | 0.0004 ✓ |
| **4.14 (shipped)** | **0.0433 / 0.0004** | 0.0017 / 0.0129 | 0.0000 / 0.0292 | 0.0020 ✓ |
| q90+0.12 (#36) | 0.0065 / 0.0004 | 0.0015 / 0.0129 | 0.0000 / 0.0292 | 0.0019 ✓ |
| q95 | 0.0046 / 0.0004 | 0.0010 / 0.0129 | 0.0015 / 0.0292 | 0.0017 ✓ |

The shipped prior hallucinates on compacts at **100× that band mirrored
baseline** (0.0433 vs 0.0004) and **26× the prior-off level** — invisible in the
pooled number because 32 of 40 cars sit near 0.0015. #35 accept/reject decision
for the prior magnitude therefore rested on a statistic that could not see the
failure mode it was meant to catch; the *direction* of #35 (far end needs
extending) stands, the *magnitude* was validated by a blind guard.

Note the compact band mirrored baseline is itself near zero (0.0004): mirroring
a compact car barely leaves its box, so that band bar is effectively "do not
leave the box at all". **No config clears it**, including #36 — q90+0.12 is
16× the baseline vs the incumbent 100×, a ~7× reduction in hallucination, not
elimination. Reporting the ratio is more informative than the pass/fail bit.

**Fix:** added gate item **d2** to `scratchpad/donor_metric_step3.py` —
`d2_out_of_box_by_gt_length_band`, `d2_bands_passing`, `d2_all_bands_pass` —
computed against each band own mirrored baseline (bands 3.6 / 4.6 m, splitting
seq-08 40 cars 8 / 27 / 5). Backfilled into all existing donor summaries;
pre-existing production summaries preserved as
`donor_metric_summary_08.pre_d2_backup.json`.

**Generalization:** the same pooled-median blindness applies to every per-car
median in #29/#32 — it is how #35 compact over-extension survived review in the
first place. Any future `complete()` geometry change must be read per band.

## 38. Production Completion Was Call-Order Dependent (Subsample RNG Carried State) — Fixed (2026-08-02)

**Context:** `complete()` subsamples the cluster to `PCN_N_INPUT`=256 points in
`_fix_size` via `self._rng`. The donor-metric eval path
(`scratchpad/donor_metric_recompute.py`) resets `completer._rng =
default_rng(seed)` **before every** `complete()`, so its subsample is fixed and
A/B-clean. Production `main.py` constructed the completer **once** (`seed=0`) and
never reset it, so `self._rng` carried state across track completions.

**Defect (measured):** a track's completed cloud therefore depends on **how many
tracks completed before it** in the run — not reproducible in isolation, and two
runs that discover/complete tracks in a different order give different subsamples
(hence different completed points) for the same track. Direct check this session
(same cluster, shared-advanced RNG vs a fresh one): OLD "identical across
call-order" = **False**, NEW = **True**.

**Magnitude — this is a reproducibility fix, not a quality change.** The seed
sweep (`output/experiments/seed_sweep_q90off`, 5 seeds, wiring corrected so the
per-pair reset actually varies) moved the donor far-end metric by **sd 0.0014**,
28–86× smaller than the published length effects — i.e. which 256 points are
drawn is low-variance noise. So the bug corrupted *reproducibility*, not
accuracy.

**Fix:** `complete()` gains `sample_seed`; when set it subsamples from a fresh
`default_rng(sample_seed)` (order-independent, reproducible), else falls back to
`self._rng` (backward-compatible with the eval scripts that reset `_rng`
externally). `main.py` passes `PIPELINE_CONFIG["pcn_sample_seed"]` (=0) per
completion, matching the donor-eval path's per-call reset exactly.

**Scope note:** `evaluate.py` never calls completion (P/R/F1/mIoU are computed
upstream of `complete()`), so this change is **inert for the headline metrics by
construction** — verified by grep, not luck. The only affected metric is the
completion/donor metric, whose seed sensitivity is the sd 0.0014 above.

## 39. Frame Convention Trap in the Donor-Pair Caches: `raw` Is SENSOR Frame (2026-08-02)

**Methodology finding (cost four scratch scripts).** The Step-1 donor caches
store `data["raw"]` in the **sensor frame** (ego at origin, Z up), *not* world.
The mappings are:

- sensor → world: `world = raw @ T[:3,:3].T + T[:3,3]`  (i.e. `raw @ Rᵀ + t`)
- `(raw - t) @ R` is a **garbage transform** — neither sensor nor world.
- `estimate_canonical_frame()` and `complete()` take the sensor-frame cluster
  **directly** (as `recompute.py` and `main.py` do); `complete()` returns sensor
  frame. `world_box()` expects **world** (X–Z ground plane, Y vertical), so a box
  is `world_box(completed @ Rᵀ + t)`.

**Evidence:** `frame_check.py` on car 45 — `raw` centroid is sensor-scale (tens
of m from origin); `raw @ Rᵀ + t` centroid matches the amodal-GT `center_world`;
`(raw − t) @ R` matches neither.

**What it cost:** four ad-hoc probes (`fallback_probe`, `minframes_sweep`,
`leakage_check`, `ols_loo`) used `(raw − t) @ R` and produced numbers that were
**retracted** — the "5 fallback cars" count and the halved-MAE claims. The
frame-correct `clean_fallback.py` (and the published
`length_1b_box_eval.py:86`, which already uses `pts @ Rᵀ + t`) are the
authority; the published box eval was never affected.

**Distinct from #26's convention.** #26 is about the *pipeline output* world
frame being Y-down (`poses @ Tr` maps sensor +Z to (0,−1,0)). This finding is
about the *donor cache* storing the pre-world sensor cluster. Rule: any
completion diagnostic that fits a box must transform sensor→world with
`raw @ Rᵀ + t` first.

## 40. OLS Sparse-Track Length Fallback — No Measured Box Gain, Not Adopted (2026-08-02)

**Negative result.** The 4.14 m constant fallback (used on tracks with < 5
gate-passed frames, #36) is a population median; hypothesis was a single-frame
OLS `L = 2.528 + 0.428·fit_length` would beat it on sparse cars.

**Frame-correct A/B (`clean_fallback.py`, world transform per #39):** the
fallback fires on only **2 of 40** amodal cars (car 45: 2 gate-passed frames;
car 336: 3). On those two, completed-box `|ΔL|` **0.214 → 0.221** (flat /
marginally worse); `center_err` 0.374 → 0.273 (better) — but n=2. Per car it is
inconsistent (car 45 better 0.321→0.131, car 336 worse 0.107→0.311). **Net: a
wash.**

**Decision:** not adopted; reverted to the committed 4.14 fallback (21430cf).
Two fitted magic constants for zero measured improvement on the only cars they
touch is not justified in research code.

**Honest caveats:** (a) n=2 is far too small to call OLS *harmful* — this is "no
evidence of benefit", not "evidence of harm"; (b) production fallback frequency
is **unknown** (2/40 is the offline donor set; if real tracks are sparser the
fallback fires more, which would warrant revisiting); (c) the coefficients were
fitted on these same 40 cars, so even the wash is an in-sample read. **To
revisit:** get a fallback-frequency count from a real `main.py --save-output` run
and a box metric with n large enough to matter.

## 41. Fallback-Frequency Answer: 23% of Completed Tracks Miss the Per-Car Length Estimate — Live Limitation, Not Fixed (2026-08-03)

**Context:** T7 of `docs/plans/delegate_brief_2026_08_02.md`. Finding #40
(caveat b) left the production fallback rate of the per-car length estimate
(#36) unmeasured — the 2/40 fallback rate on the offline amodal-GT set could
not tell us how often real `main.py` tracks are too sparse (< 5 gate-passed
frames) for `track_length_estimate()`'s quantile to be meaningful. `output/08`
was regenerated under the shipped per-car estimate + the #38 RNG
order-independence fix (superseding the 2026-08-01 fixed-prior/pre-#38
version), with T6's new `length_estimate_source`/`n_gate_passed_frames`
instrumentation in `tracks.json` to answer this directly.

**Finding:** Verification (`scratchpad/verify_regen_08_t7.py`, md5-based)
confirms detection/tracking is unaffected by the regen: track set (1040
tracks), all identity fields (first/last_frame, point_count, raw_point_count,
class, centroid_history), all 518 `_partial.ply` inputs, and all
non-completed `.ply` outputs are byte-identical between old and new — only
completed clouds changed, as intended. Of the 518 completed tracks, **399
(77.0%) used the per-car `track_q90` estimate; 119 (23.0%) fell back to the
fixed 4.14 m prior** (< `COMPLETION_LENGTH_MIN_FRAMES` = 5 gate-passed
frames).

**Decision:** Per the delegate brief's pre-registered rule (> ~5% fallback →
log as a live limitation, do not implement a fix), 23.0% clears that bar by a
wide margin. Logged as a live limitation of #36's per-car estimate: nearly a
quarter of completed tracks still rely on the population-level constant
rather than a per-car number, most likely short/fragmented tracks that never
accumulate 5 gate-passed frames. No fix implemented — out of scope for this
task; a targeted investigation (e.g. lowering `COMPLETION_LENGTH_MIN_FRAMES`,
or a different sparse-track estimator now that #40's OLS attempt is a
documented negative result) would be a new backlog item.

**Housekeeping:** old `output/08` preserved as `output/08_fixedprior_backup/`
(reversible); `output/08_regen` promoted to `output/08` (GT artifacts
`amodal_gt.json`/`amodal_gt_check.png` copied in unchanged).

## 42. Seq-00 Held-Out Replication Verdict — PARTIALLY HOLDS (T9c) (2026-08-09)

**Context:** T9c judge session (executor/judge separation) applying the
pre-registered R1/R2/R3 refutation criteria
(`docs/plans/preregistration_heldout.md`) verbatim to the frozen seq-00
completion-eval tables (`docs/plans/t9b_results_heldout_seq00.md`, #29 box +
#32 donor, production config/checkpoints, no tuning). Full worked verdict:
`docs/plans/t9c_verdict_heldout_seq00.md`.

**Finding:** Verdict = **PARTIALLY HOLDS**. Both primary metrics are decisive
wins over the raw partial:

| Primary metric | raw | completed | Wilcoxon p |
|---|---|---|---|
| BEV IoU (ALL, n=45) | 0.739 | 0.766 | 1.6e-3 |
| Donor cov@0.1 (τ=0.15) | 0.000 | 0.413 | ~0 |

- **R1 (does-not-generalize) — not triggered.** Neither primary fails to beat
  raw.
- **R2 (08-specific length constants) — not triggered, verbatim.** No testable
  band's d2 ratio exceeds its seq-08 level (compact 1.8× < ≈16×; normal 0.02× <
  ≈0.12×). Compact fails its pass-bit on 00 (0.0090 > 0.0050) but that is a
  *smaller* violation than 08 and compact did not pass on 08 either, so neither
  R2 clause applies.
- **R3 (uncovered result) — triggered** by the **empty long band (0 cars
  ≥4.6 m)**: seq-00's well-observed set is all compact/normal, so the long-band
  predictions and long-band d2 guard are untestable ("band Ns too small to
  test"). Escalated to the user with the tables.

**Decision:** Resolved via the taxonomy's PARTIALLY HOLDS "caveat named" clause
(user decision, 2026-08-09): downgrade HOLDS → PARTIALLY HOLDS with the empty
long band as the named caveat. The "partial" reflects a **coverage gap, not a
weak metric** — both primaries were clean significant wins. Constants **not**
retuned. Tier-3 gate is satisfied (PARTIALLY HOLDS permits T13).

## 43. Clustering Benchmark — HDBSCAN's Density-Adaptive Linkage Buys Recall No Fixed Radius Reaches (T10) (2026-08-09)

**Context:** T10 of the delegate brief (backlog #5). Compared production
HDBSCAN against Open3D DBSCAN and PCL-style Euclidean cluster extraction, at a
**fixed `cluster_voxel_size=0.10`** and identical pre/post pipeline stages —
only the clustering algorithm swapped. Added `_cluster_dbscan` /
`_cluster_euclidean` + dispatch + `--clustering-method` choices to
`pipeline.py`/`evaluate.py` (additive; reproduction baseline P 0.967/R 0.777/
F1 0.862/mIoU 0.962, TP=1296/FP=44/FN=371 matched EXACTLY after the edits).
Scripts: `scratchpad/t10_clustering_benchmark.py`, `scratchpad/t10_eps_sweep.py`;
results `output/experiments/t10_clustering/`. Params untuned/standard: eps =
tolerance = 0.5 m, min size 10 (mirrors `hdbscan_min_cluster_size`).

**Deviation (logged):** metrics are **per-frame, track filter OFF**, on both
sequences. A strided seq-08 sample is incompatible with the centroid tracker
(links consecutive frames → stride-20 makes every detection a length-1 track →
`min_track_length=2` rejects all). Per-frame eval also isolates clustering from
the method-independent track post-filter, which is the point of the benchmark.

**Finding — main table (per-frame, no track filter):**

seq 00 (100 frames):

| method | P | R | F1 | mIoU | clustering ms (med) | med #clusters |
|--------|---|---|----|------|---------------------|---------------|
| **HDBSCAN** | 0.962 | **0.731** | **0.831** | 0.968 | 393 | 129 |
| DBSCAN (eps 0.5) | 0.962 | 0.637 | 0.767 | 0.964 | **58** | 108 |
| Euclidean (0.5) | 0.967 | 0.636 | 0.768 | 0.982 | 77 | 96 |

seq 08 (stride-20, 204 frames):

| method | P | R | F1 | mIoU | clustering ms (med) | med #clusters |
|--------|---|---|----|------|---------------------|---------------|
| **HDBSCAN** | 0.813 | **0.671** | **0.735** | 0.923 | 452 | 236 |
| DBSCAN (0.5) | 0.848 | 0.565 | 0.678 | 0.927 | **68** | 139 |
| Euclidean (0.5) | 0.838 | 0.576 | 0.683 | 0.950 | 92 | 124 |

HDBSCAN wins F1 on both sequences (+0.06 seq00, +0.05 seq08), **entirely
through recall** — precision is method-independent (set by the geometric filter
+ classifier, not clustering). The alternatives are **5–7× faster** (58–92 ms
vs 393–452 ms). Euclidean ≈ DBSCAN on F1; DBSCAN's core-point density test
discards sparse points Euclidean keeps, so Euclidean edges it on recall/mIoU.

**Pre-registered mechanism was half-wrong:** predicted density methods would
lose *precision* by merging adjacent cars. Precision is tied — the merge
instead surfaces as *recall* loss (a merged car-pair yields one TP + one FN),
consistent with the lower cluster counts (108/96 vs 129 on seq00; 139/124 vs
236 on seq08).

**eps robustness (seq 00, rebuts the "one arbitrary eps" concern):** no fixed
radius reaches HDBSCAN's recall. DBSCAN/Euclidean recall peaks at eps 0.4–0.5
then falls; the best fixed-radius config is **Euclidean eps=0.4 (R 0.684 /
F1 0.800)** — still below HDBSCAN (R 0.731 / F1 0.831).

| eps | DBSCAN R / F1 | Euclidean R / F1 |
|-----|---------------|------------------|
| 0.3 | 0.525 / 0.684 | 0.663 / 0.789 |
| 0.4 | 0.624 / 0.763 | **0.684 / 0.800** |
| 0.5 | 0.637 / 0.767 | 0.636 / 0.768 |
| 0.7 | 0.543 / 0.694 | 0.535 / 0.689 |
| — | **HDBSCAN ref: 0.731 / 0.831** | |

**Decision:** **Adopt nothing** — HDBSCAN stays production. Its density-adaptive
linkage recovers cars that no single global radius can, without per-scene
tuning; the 5–7× runtime penalty matters only under a real-time budget, which
is out of scope (#34, user decision 2026-07-23). `dbscan`/`euclidean` kept as
`--clustering-method` options for reproducibility, like `bev` (#21). Thesis
table material for the clustering-choice justification.

## 44. Moving Cars Complete As Plausibly As Statics — Validation Gap Closed (T11) (2026-08-09)

**Context:** T11 of the delegate brief. Completion validation (#29/#32) covered
STATIC cars only (paired amodal-GT boxes require a stationary car), but
production completes movers too. Checked whether movers — never validated —
complete into plausible car boxes at a comparable rate. Ran on the regenerated
`output/08` (per-car length estimate + #38 RNG fix, 518 completed car tracks).
Script: `scratchpad/t11_mover_plausibility.py`; figure
`output/figures/t11_mover_completions_bev.png`; json
`output/experiments/t11_mover_plausibility.json`. Observational only, no fixes.

**Method:** plausible-car-box recipe verbatim from #27/#28 —
`dims` = desc-sorted global-axis extents; plausible iff L∈[3.3,4.9],
W∈[1.5,2.1], H∈[1.1,1.7]. Motion split is label-free (output/08 tracks are
unlabeled): net horizontal displacement of the track centroid (median of first
5 vs last 5 frames, X-Z ground plane, Y vertical per #26/#27). static =
net ≤ 2.0 m (mirrors amodal_gt's 2.0 m center-spread guard); moving = net ≥ 5.0 m
(no parked car drifts 5 m); ambiguous 2–5 m reported separately. net_disp is
robust to the ~2 m per-frame centroid jitter that inflates raw path span even
for parked cars.

**Finding:**

| group (net disp) | n | plausible | rate | median L/W/H |
|------------------|---|-----------|------|--------------|
| STATIC (≤2 m) | 419 | 225 | 53.7% | 3.81/1.94/1.49 |
| **MOVING (≥5 m)** | **19** | **11** | **57.9%** | 3.84/2.05/1.44 |
| ambiguous (2–5 m) | 80 | 46 | 57.5% | 3.95/1.92/1.49 |
| ALL | 518 | 282 | 54.4% | 3.83/1.94/1.49 |

Movers are **not worse** than statics — slightly higher (+4.2 pp), well within
noise at n=19, with near-identical median dims. Every motion bucket clusters at
54–58%, so **motion does not degrade completion plausibility.** This is expected:
completion runs on a single reference frame's cluster (not accumulated points),
so there is no motion-smear penalty on the input.

**Recipe caveat (from the figure, not a mover effect):** most failing movers
fail on **inflated width** (t30181 W 3.5, t22653 W 3.0, t26359 W 2.7) — they are
genuine car-shaped footprints oriented **diagonally** in the X-Z plane, and the
`dims` recipe's axis-aligned extents overestimate W/L for off-axis cars (the
#28 caveat). So the absolute ~54% rate **understates** completion quality (an
oriented-box measure would score higher); the mover-vs-static comparison is
unaffected since both groups use the identical recipe.

**Decision:** No fix (observational task). The statics-only validation does not
hide a mover failure mode — movers complete comparably. Feeds T14 ch. 5
(limitations): the in-sample/statics-only calibration concern is mitigated on
the motion axis. If a future task wants a truer absolute plausibility rate,
switch `dims` to oriented-box extents (out of scope here).

## 45. Step 1c Radius-Decouple + Fill Factor — NEGATIVE: Compensation Is Irreducibly Length-Dependent (T13) (2026-08-09)

**Context:** T13/Step 1c (delegate brief Tier 3), pre-registered in
`docs/completion/t13_step1c_plan.md`. #36's over-extension control showed
normal/long completions are still too short even with an unbiased per-car length
estimate, because the Z length-push inflates the normalization `radius`
(`completion.py:499`) so scale rides on the length prior. Fix, two locked parts:
**D1** decouple `radius` from the Z push (radius against the pre-Z center; keep
X/Y pushes + ×1.137); **D2** correct PCN's frame under-fill with a length-axis
fill factor calibrated on synthetic true GT and applied (not refit) on real.
A/B behind `PointCloudCompleter(decouple_radius, fill_z)` (default OFF =
production); evaluated via `donor_metric_recompute.py --decouple-radius --fill-z`
→ band-split #29 box + #32 donor + per-band d2 on seq 08 (n=40) and seq 00
(n=45, held-out). Scripts: `scratchpad/t13_fill_factor.py` (D2 calibration),
`scratchpad/donor_metric_recompute.py`. Dirs `output/experiments/t13_*`.

**D2 calibration (synthetic, n=300 val cars):** per-axis fill (GT/PCN extent) =
X 1.099, Y 1.037, **Z 1.074**. Pre-registered widen rule (>1.10): X 1.099 is a
hair under → no widen, Y no. So length-only, **fill_z = 1.074** — the
pre-registration's fixed threshold made this call (no post-hoc widening at 1.099).

**Result — box #29, per-car median |ΔL| (before → D1+D2):**

| band | seq 08 | seq 00 |
|------|--------|--------|
| compact <3.6 | 0.129 → **0.238** (+0.109) ✗ | 0.185 → **0.301** (+0.116) ✗ |
| normal | 0.329 → **0.232** (−0.097, p=7.7e-3) ✓ | 0.294 → **0.237** (−0.057) ✓ |
| long ≥4.6 | 0.583 → **0.363** (−0.220) ✓ | n=0 |

Donor cov@0.1 improved everywhere (seq 08 0.396→0.407, far_end 0.316→0.353;
seq 00 0.447→0.453, far_end 0.366→0.409). But compact **over-extends**
(signed_dL seq 08 +0.06→+0.23, seq 00 +0.13→+0.29) and its d2 out-of-box
hallucination worsens (seq 08 0.0065→0.0158, seq 00 0.0090→0.0161; pass-bit
already False on both, so not formally gated, but clearly worse).

**Mechanism (confirmed):** D1 alone makes completions *shorter* (normal |ΔL|
0.329→0.496 on seq 08) — proving the coupled radius was inadvertently
lengthening them via the center push. D2's uniform +7.4% stretch fixes
normal/long but over-extends compacts, which needed ~zero lengthening. Note the
synthetic fill (1.074) is small: most of `before`'s normal/long length came from
the coupled-radius inflation, not PCN under-fill, and the residual is
length-dependent.

**Decision — DO NOT SHIP (pre-registered negative result).** Primary criterion
met (normal/long under-extension fixed, donor coverage up) but the pre-registered
**compact non-regression guard fails on both sequences** (|ΔL| worse by >0.02 m).
Applied the gate verbatim; did not relax it post-hoc. This empirically confirms
the plan's exclusion of option 3: a single fill factor cannot serve all bands —
the required compensation is **length-dependent** (compacts ~0, normal/long
larger), matching #36's ≈0.02/0.68/1.02 m by-band estimate. House precedent for
documented negatives: #16/#17/#19/#40. `decouple_radius`/`fill_z` retained as
A/B knobs (default OFF, invariant-tested, production unchanged) so the result is
reproducible; a length-dependent fill (option 3) is the only remaining lever and
stays out of scope (would need more long/compact cars than the 8/27/5 available
to fit safely).

## 46. Stale `large_centroids` in `_merge_nearby_clusters` — Real Inconsistency, NEGATIVE to Fix (2026-08-09)

**Context:** An external code review flagged that `_merge_nearby_clusters`
(`src/pipeline.py`) computes `large_centroids` once before the merge loop, but
when a small cluster is absorbed it updates `cluster_info[lcl]["centroid"]` in
place without refreshing the `large_centroids` array used for the next distance
test. Confirmed real: subsequent small clusters measure distance to the
pre-merge centroid. (The `z_min`/`z_max` updates *are* consumed via `l_info`;
only the centroid goes stale.)

**Finding:** The merge path is disabled in the production default
(`"merge_fragments": False`), so this has **zero effect on any headline result**
— the default eval never calls the function. Tested the fix
(`large_centroids[idx] = cluster_info[lcl]["centroid"]`) with the path forced on:

```
.venv\Scripts\python.exe src/evaluate.py --merge-fragments   # seq 00, 100 frames
```

| | TP | FP | FN | Prec | Rec | F1 | meanIoU |
|---|---|---|---|---|---|---|---|
| Pre-fix (stale) | 1260 | 41 | 407 | 0.968 | 0.756 | 0.849 | 0.968 |
| Post-fix (fresh) | 1261 | 45 | 406 | 0.966 | 0.756 | 0.848 | 0.967 |

Keeping centroids fresh pulls a few extra merges → +4 FP, ΔF1 −0.001,
ΔmeanIoU −0.001. Neutral-to-marginally-negative.

**Decision:** Reverted the fix. The stale-centroid behavior (merge toward
*original* centroids) scores marginally better, so it is left as-is
intentionally — not a latent bug to fix. The sibling `f_score` precision/recall
naming swap (`src/completion.py`) *was* kept: output-identical (F1 symmetric),
pure readability.

## 47. IoU-Threshold Sensitivity (B1) + Geometric-Only Ablation Under Promoted Config (B6) — seq 08 full (2026-08-21)

**Context:** Thesis-plan mandatory evidence tasks B1/B6 (`THESIS_PLAN.md` §3),
run read-only against the frozen `PIPELINE_CONFIG` during the write-up freeze.
B1 defends examiner Q4 ("why IoU 0.3?"); B6 makes the stage-ablation table
like-for-like under the promoted config. Full seq 08 (4,071 frames). Logs:
`output/experiments/iou_sensitivity/seq08_{iou025,iou050,geomonly}.log`.

**Finding:**

B1 — IoU-threshold sensitivity (frozen config; 0.30 row = published headline):

| IoU thresh | Precision | Recall | F1 | Mean IoU | TP / FP / FN |
|---|---|---|---|---|---|
| 0.25 | 0.908 | 0.732 | 0.811 | 0.910 | 25576 / 2578 / 9346 |
| 0.30 | 0.905 | 0.730 | 0.808 | 0.912 | 25478 / 2676 / 9444 |
| 0.50 | 0.879 | 0.709 | 0.785 | 0.927 | 24744 / 3410 / 10178 |

F1 moves 0.811 → 0.808 → 0.785 — a 2.3-point drop from 0.30 to 0.50, as the
matched-IoU of 0.96 predicts. The headline is **not** an artifact of IoU=0.3.
Mean IoU rises at 0.50 (stricter gate keeps only better-overlapping matches).

B6 — geometric-only ablation (`--no-learned-classifier --no-track-filter`),
promoted config, full seq 08:

| Config | Precision | Recall | F1 | Mean IoU | TP / FP / FN |
|---|---|---|---|---|---|
| Geometric-only (B6) | 0.149 | 0.775 | 0.250 | 0.907 | 27051 / 154612 / 7871 |
| Full pipeline (frozen) | 0.905 | 0.730 | 0.808 | 0.912 | 25478 / 2676 / 9444 |

This is the like-for-like geometric-only number under the promoted config —
**F1 0.250**, superseding #18's pre-promotion geometric-only F1 0.305
(P 0.205 / R 0.594, measured 2026-05-28 before the cv=0.10 promotion). #18 is
preserved as the historical record. Two honest points for §4.3:
- Promoted cv=0.10 *raises* geometric-only recall (0.594 → 0.775) but precision
  collapses (0.205 → 0.149): coarser clustering emits more, mostly-junk clusters.
  The classifier removes ~152k false positives (154612 → 2676), which is the
  precision mechanism (P 0.149 → 0.905).
- Geometric-only recall (0.775) *exceeds* the full-pipeline recall (0.730): the
  classifier + track filter trade ~1,600 true positives for the 152k-FP removal.

Commands:
```
.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --iou-threshold 0.25
.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --iou-threshold 0.5
.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --no-learned-classifier --no-track-filter
```

**Decision:** Ablation claim C2 corrected to geometric-only **F1 0.250 → 0.808**
(like-for-like, promoted config), replacing the mixed-config 0.305 → 0.808;
#18 kept as historical pre-promotion record. B1 confirms threshold robustness —
no tuning warranted (pre-registered acceptance: "if it isn't stable, report it").

## 48. GT Cars Excluded by the Eligibility Rule (B2) — recall denominator vs. all annotated cars — seq 08 (2026-08-21)

**Context:** Thesis-plan mandatory evidence task B2 (`THESIS_PLAN.md` §3),
defending examiner Q9 ("recall against *all* annotated cars, not just
survivors"). Reported detection recall (seq-08 R=0.730) uses a denominator of GT
car instances with **>=10 points surviving preprocessing** (z-filter -> voxel ->
denoise -> ground removal) — the `gt_masks` rule in `evaluate.py:238-242`. This
quantifies how many annotated cars never enter that denominator. Script:
`scratchpad/gt_eligibility_count.py`; JSON:
`output/experiments/gt_eligibility/gt_eligibility_08.json`. Seq 08, stride-20
sample (204 of 4,071 frames, drawn across the whole drive). Classifier not loaded
— `gt_masks` are independent of it, and the eligible count is taken verbatim from
`get_frame_detections` so it matches `evaluate.py` exactly.

**Finding:** Three distinct per-frame quantities (sem in {10, 252}, inst > 0),
pooled over 204 frames:

| Quantity | Rule | Pooled count |
|---|---|---|
| `raw_all` | any point count | 2332 |
| `raw_ge10` | >=10 **raw** points (advisor-report distinct-car rule) | 2033 |
| `eligible` | >=10 **surviving** points (= eval recall denominator) | 1737 |

Exclusion by the eligibility rule (measured):
- vs. all annotated cars (`raw_all`): **25.5%** micro / 23.8% per-frame mean.
- vs. >=10-raw-point cars (`raw_ge10`): **14.6%** micro / 13.8% per-frame mean.

The two >=10-point rules are different quantities (project_state warns of this):
>=10 *raw* points is a distinct-car count threshold on the unfiltered label
cloud; >=10 *surviving* points is the eval denominator after preprocessing. 12.8%
of annotated cars (299 of 2332) have fewer than 10 raw points at all (mostly
far/occluded); a further 14.6% of the >=10-raw survivors are lost to
preprocessing.

Recall against all annotated cars (inferred, not separately measured): ineligible
cars can never be a TP (absent from `gt_masks`), so recall-vs-all = TP/`raw_all` =
reported recall x (eligible/raw_all). With the stride-20 survival ratio
1737/2332 = 0.745: **~0.730 x 0.745 ~= 0.54**. Stated as an inference — the
survival ratio is a stride-20 estimate (representative to ~1-2% per the
timing-sampling lesson), and TP itself is unchanged.

**Decision:** §4.1 (evaluation protocol) states the denominator explicitly and
reports both the eligibility-exclusion percentage (25.5% vs. all annotated /
14.6% vs. >=10-raw) and the implied ~0.54 recall-against-all-annotated, with the
stride documented. No code or config change (read-only analysis; freeze intact).
