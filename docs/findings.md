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

## 24. Recall Improvement Strategies — Both Negative (2026-06-24)

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

## 28. PoinTr vs PCN — Synthetic Win Does Not Transfer to Real Data (2026-06-30)

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

**(a) Synthetic validation — decisive PoinTr win:**

| Model | val cd_fine | val F@0.1m |
|---|---|---|
| PCN (`pcn_kitti_best`) | 0.1246 | 0.76 |
| **PoinTr (`pointr_kitti_best`)** | **0.0634** | **0.987** |

PoinTr roughly halves Chamfer error and pushes F-score 0.76→0.99 on the identical
synthetic val set.

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

**Conclusion:** PoinTr's large synthetic-CD advantage **does not transfer to real
data.** On real one-sided LiDAR clusters the two models are equivalent in plausibility
and shape. Both are bottlenecked by the same factors — the residual real-vs-synthetic
partiality gap (Finding #26) and the centroid/scale estimation in `complete()` — not by
decoder capacity. This is the same synthetic→real transfer gap that defined the entire
PCN saga (#15–19, #26). The transformer's extra fidelity is spent on synthetic detail
that real scans neither contain nor benefit from.

**Decision:** **Keep PCN as production** (`pcn_kitti_best.pth`) — smaller and equivalent
on real data; no real-data justification to swap. PoinTr is retained
(`pointr_kitti_best.pth`, `src/pointr.py`, `src/train_pointr.py`) and
`completion.py._load_model()` dispatches either by checkpoint, so the swap is one flag
if ever wanted. AdaPoinTr (denoising + adaptive query bank) is **not pursued** — it
targets synthetic completion fidelity, which this finding shows is not the real-data
bottleneck. For the thesis, the result is a clean negative/transfer-gap contribution,
not a PoinTr-beats-PCN headline.

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
