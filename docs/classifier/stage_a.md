# Classifier — Stage A (Synthetic ShapeNet Pretraining)

Stage A pretrains the dual-branch PointNet classifier on **synthetic ShapeNet
partial renders**. It produces `checkpoints/classifier_best.pth`, which is the
initialization for Stage B fine-tuning on real LiDAR (see `stage_b.md`).

> **Scope note.** The classifier is **binary** (`car` / `not-car`) as of
> Finding #20. Earlier 4-class (car/bus/motorcycle/unknown) checkpoints are
> obsolete.

## Purpose

Real LiDAR car clusters are scarce and class-imbalanced (~20% car, near-zero
bus/motorcycle — Finding #8). Training on real data alone would under-fit the
positive class and provide no balanced shape prior. Stage A supplies a balanced,
shape-rich prior from synthetic CAD models so that Stage B only has to adapt to
the LiDAR domain rather than learn "what a car looks like" from scratch.

## Data source

| Item | Value |
|------|-------|
| Root | `dataset/shapenet_data/<synset_id>/<model>/models/model_normalized.obj` |
| Positive (`car`) | ShapeNet synset `02958343`, scaled to 4.5 m max extent |
| Negative (`not-car`) | All **non-car** ShapeNet synsets, random scale 0.5–4.0 m |
| Class balance | `unknown_fraction = 0.50` (negatives subsampled to match positives) |
| Split | 80/20 train/val per category, `split_seed = 42` |

Negatives are deliberately drawn from *every* other ShapeNet category and given a
wide random scale so the classifier learns to reject arbitrary non-car shapes at
arbitrary sizes.

## Partial rendering (`_render_partial`)

Each sample is a single-viewpoint partial point cloud produced by **pinhole
depth-camera ray casting** against the mesh (Open3D `RaycastingScene`):

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| FOV | 60° |
| Viewpoint elevation | −20° … +30° |
| Viewpoint azimuth | 0° … 360° |
| Camera radius | 1.5 … 2.5 × object scale |
| Min hits to accept | 50 |

The mesh surface is area-weighted triangle-sampled; the visible subset from the
random camera pose becomes the partial cloud. Viewpoint is randomized per sample
(deterministic per index via `split_seed + idx`).

> ⚠️ **Known domain gap ("too perfect").** A pinhole depth render produces a
> *dense, uniform, single-viewpoint* surface sample. Real Velodyne data has
> anisotropic scan-ring structure, range-dependent sparsity, foreground
> occlusion from *other* objects, and range-proportional noise. This mismatch is
> why Stage A alone is unusable on real data (see "Domain gap" below). An
> ablation (Finding #25) shows that once Stage B fine-tunes on 420k real
> clusters, this synthetic prior is **redundant** — training Stage B from
> random init matches it at the classifier level (macro-F1 0.9285 vs 0.9225).
> The "too perfect" data is not hurting, but not meaningfully helping either.

## Preprocessing (shared with inference)

Defined in `src/classifier.py`:

1. **Centroid-center** the metric-scale points.
2. **Bbox features** — 8-d metric vector via `extract_bbox_features`:
   sorted extents (min, med, max), volume, aspect max/min, aspect med/min,
   `log1p(point_count)`, vertical height span. Z-score normalized using
   training-set stats, clipped to ±5.
3. **Point branch** — `sample_or_pad` to `NUM_POINTS = 512`, then
   `normalize_unit_sphere` (scale-invariant; absolute size lives in bbox feats).
4. **Augmentation (train only)** — random Z-axis rotation + Gaussian jitter
   (σ = 0.005 after normalization).

## Model architecture (`PointNetClassifier`)

Dual-branch, ~80 k parameters (a deliberately small PointNet — see Finding #12):

```
point branch:  (B,512,3) → Conv1d 3→64→128→256 (BN+ReLU) → max-pool → (B,256)
bbox branch:   (B,8)     → Linear 8→32 (ReLU)            → (B,32)
head:          concat (B,288) → Linear 288→128 → ReLU → Dropout(0.3) → 128→2
```

The bbox branch injects real-world metric scale that unit-sphere normalization
strips from the point branch. No T-Nets (clusters are already in a consistent
sensor frame).

## Training hyperparameters (`TRAIN_CONFIG`)

| Param | Value |
|-------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | Adam, lr 1e-3, weight decay 1e-5 |
| LR schedule | StepLR, step 30, γ 0.5 |
| Loss | Class-weighted cross-entropy (weight ∝ 1/√count) |
| Unknown threshold | 0.50 (max-softmax below → `not-car`) |
| Checkpoint selection | Best thresholded macro-F1 |

## Results

- Best val macro-F1: **0.9986** (synthetic val set).
- This number reflects discrimination on *synthetic* data only and does **not**
  transfer to real LiDAR — see below.

## Domain gap (Finding #7)

When the Stage A checkpoint is run directly on real pipeline clusters, it
predicts **every** cluster as `not-car` with ~100% confidence: 0 TP, pipeline
F1 0.000. The synthetic feature distribution does not overlap real LiDAR at all.
This is the expected synthetic-to-real gap and is the entire motivation for
Stage B fine-tuning.

## Reproduce

```bash
# Activate env first (.venv\Scripts\activate)
python src/train_classifier.py --epochs 50            # full Stage A
python src/train_classifier.py --eval-only --resume checkpoints/classifier_best.pth
```

## Key files

- `src/train_classifier.py` — `ShapeNetClassificationDataset`, `TRAIN_CONFIG`,
  `_render_partial`, training loop.
- `src/classifier.py` — `PointNetClassifier`, shared preprocessing,
  `extract_bbox_features`, inference (`classify_cluster`, `load_classifier`).

## Outputs

- `checkpoints/classifier_best.pth` — best Stage A checkpoint (Stage B init).
- `checkpoints/classifier_last.pth`, `classifier_epoch{N}.pth` — periodic.
- `checkpoints/classifier_training_log.csv` — per-epoch metrics.
