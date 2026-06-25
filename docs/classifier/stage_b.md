# Classifier — Stage B (Real LiDAR Fine-Tuning)

Stage B fine-tunes the Stage A PointNet (see `stage_a.md`) on **real LiDAR
clusters mined from SemanticKITTI**, bridging the synthetic-to-real domain gap.
It produces `checkpoints/stage_b_best.pth`, the classifier used in the pipeline.

## Purpose

Stage A learns balanced car/not-car *shape* priors from synthetic CAD models but
collapses on real data (Finding #7: 0 TP, predicts everything `not-car`). Stage B
adapts those features to the real LiDAR distribution — scan-ring sparsity,
occlusion, noise — using a large set of automatically labeled real clusters.

## Mining real clusters (`src/mine_stage_b.py`)

For each frame, the pipeline's stages 1–5 are replayed and GT semantic labels are
attached to surviving clusters, then each cluster is labeled and saved.

**Procedure per frame:**
1. Stage 1–3: z-filter, statistical denoise, voxel downsample.
2. Propagate SemanticKITTI labels onto downsampled points via KD-tree
   (nearest denoised point).
3. Stage 4–5: RANSAC ground removal, HDBSCAN clustering, geometric filtering.
4. For each surviving cluster, vote its points' semantic labels into the binary
   mapping and accept if the majority class **purity ≥ 0.75**:
   - `10` (car), `252` (moving-car) → **car**
   - everything else → **not-car**
5. Centroid-center the cluster and save as `.npy` under `<split>/<class>/`.

Clusters below the purity threshold are discarded (too mixed to train on; discard
rate ~0.7%, Finding #8).

> **Why mine through the real pipeline?** The training clusters are produced by
> the *same* clustering + filtering the classifier sees at inference — including
> HDBSCAN splits and geometric-filter survivors — so Stage B learns on the exact
> cluster distribution it must classify.

## Dataset

| Split | Sequences | Frames/seq | Samples | car / not-car |
|-------|-----------|-----------|---------|---------------|
| train | 00–07, 09, 10 | 5000 | 420,333 | 88,600 / 331,733 |
| val   | 08 | 5000 | 130,394 | 24,968 / 105,426 |

Class balance is intrinsically ~21% car / ~79% not-car. Handled at training time
by class-weighted loss, not resampling.

> **Eval discipline (Finding #18).** Seq 08 is held out from training and is the
> **mandated** evaluation sequence. Earlier seq-00 numbers were a data leak
> (seq 00 was in the Stage B training set) and are retained only as flagged
> historical values.

Stored at `dataset/stage_b/{train,val}/{car,not-car}/*.npy` with
`metadata_{split}.json` (per-cluster purity, point count, centroid, semantic
histogram, and the full `PIPELINE_CONFIG` snapshot for reproducibility).

## Preprocessing & model

Identical to Stage A (`stage_a.md`): centroid-center → 8-d bbox features (z-score
normalized) + `sample_or_pad` to 512 points → unit-sphere normalization; Z-rotation
and σ=0.005 jitter for train augmentation. Same `PointNetClassifier`.

Bbox-feature mean/std are **recomputed** from the real Stage B training set
(50 k-sample estimate) — the synthetic stats from Stage A do not apply to the
real domain.

## Training hyperparameters (`STAGE_B_CONFIG`)

| Param | Value |
|-------|-------|
| Init | Stage A weights (`classifier_best.pth`), **fresh optimizer/scheduler** |
| Epochs | 15 |
| Batch size | 64 |
| Optimizer | Adam, lr 1e-4, weight decay 1e-5 |
| LR schedule | StepLR, step 10, γ 0.5 |
| Loss | Class-weighted cross-entropy (weight ∝ 1/√count) |
| Unknown threshold | 0.50 |
| Checkpoint selection | Best thresholded macro-F1 |

Stage A is loaded as **weights only** (fine-tune from scratch optimizer state),
not a resume. The `--stage-b` flag auto-loads `classifier_best.pth` unless
`--no-pretrain` is passed.

### Ablation flags

Added for Experiment A (Stage A pretraining ablation):

- `--no-pretrain` — train Stage B from random init (skip Stage A checkpoint).
- `--tag <prefix>` — override checkpoint/log filename prefix (avoids overwriting
  `stage_b_best.pth`).
- `--seed <int>` — set RNG seed for reproducibility.

> **Ablation result (Finding #25).** From-scratch Stage B (`--no-pretrain`,
> seed 0, 15 epochs) reaches val macro-F1 **0.9285**, matching/slightly beating
> the pretrained baseline (0.9225). Stage A pretraining is redundant given the
> 420k-cluster real training set. The pretrained checkpoint retains a pipeline
> precision edge (P 0.956 vs 0.905) but under uncontrolled conditions — see the
> finding for caveats.

## Results

Best checkpoint: **epoch 13/15**, val macro-F1 **0.9225** (seq 08 val clusters).

Classifier-level (seq 08 val clusters, `stage_b_best.pth`):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| car | 0.874 | 0.875 | 0.875 |
| not-car | 0.970 | 0.970 | 0.970 |

Pipeline-level (seq 08, with classifier + track filter — Finding #24 baseline):

| Metric | Value |
|--------|-------|
| Precision | 0.956 |
| Recall | 0.731 |
| F1 | 0.829 |

> Pipeline recall is capped at ~0.74 by **HDBSCAN cluster splitting**, not the
> classifier (Findings #22–24). The classifier's job is precision: it lifts
> pipeline precision from ~0.21 (geometric-only) to ~0.96.

## Reproduce

```bash
# Mine the dataset (already built; re-run only to regenerate)
python src/mine_stage_b.py --seq 00 01 02 03 04 05 06 07 09 10 --frames 5000 --split train
python src/mine_stage_b.py --seq 08 --frames 5000 --split val

# Fine-tune (auto-loads Stage A checkpoint)
python src/train_classifier.py --stage-b --epochs 15

# Evaluate
python src/train_classifier.py --stage-b --eval-only --resume checkpoints/stage_b_best.pth
python src/evaluate.py --seq 08 --frames 100 --classifier-ckpt checkpoints/stage_b_best.pth
```

## Key files

- `src/mine_stage_b.py` — cluster mining + purity labeling.
- `src/train_classifier.py` — `StageBDataset`, `STAGE_B_CONFIG`, training loop.
- `src/classifier.py` — model + shared preprocessing + inference.

## Outputs

- `checkpoints/stage_b_best.pth` — production classifier checkpoint.
- `checkpoints/stage_b_last.pth`, `stage_b_epoch{N}.pth` — periodic.
- `checkpoints/stage_b_training_log.csv` — per-epoch metrics.
- `dataset/stage_b/` — mined clusters + metadata.
