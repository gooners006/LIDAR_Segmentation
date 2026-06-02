# Session Summary — 2026-04-20/21

## What was done

### 1. Pipeline review against external feedback

Reviewed all pipeline stages against `docs/pipeline_feedback.md` (critical review from advisor/reviewer). Prioritized three improvements and implemented them.

### 2. HDBSCAN clustering (Step 4)

- **Replaced** Open3D's fixed-epsilon DBSCAN with density-adaptive HDBSCAN.
- **sklearn 1.8.0 bug:** `cluster_selection_epsilon` crashes in Cython tree code. Dropped that parameter.
- **Performance:** sklearn's HDBSCAN is 3.7s/frame — unacceptably slow. Switched to the dedicated `hdbscan` package (0.48s/frame, 7.7x faster). See `docs/findings.md` for full benchmark.
- **Detection quality** (20-frame eval, IoU >= 0.3):

  | Method           | Precision | Recall | F1    |
  | ---------------- | --------- | ------ | ----- |
  | DBSCAN (eps=0.5) | 0.188     | 0.804  | 0.305 |
  | HDBSCAN          | 0.154     | 0.868  | 0.261 |

  HDBSCAN improves recall (+6.4%) but increases false positives. Geometric filters need tightening to compensate.

### 3. Ground-plane-relative height filtering (Step 5)

- `filter_clusters` now computes bbox center height as signed distance from the RANSAC ground plane, instead of raw `center[2]`.
- Falls back to raw z when no plane was fitted.
- Threshold changed from `max_center_z=0.5` (sensor-relative) to `max_center_height_above_ground=3.0` (ground-relative).
- `remove_ground` now returns 4 values: `(ground_pcd, objects_pcd, plane_model, inlier_indices)`.

### 4. Completion infrastructure (Step 7 — dormant)

- Added `simulate_lidar_noise()` — sim-to-real augmentation for ShapeNet pretraining.
- Added `KITTIObjectDataset` — loads sparse/dense `.npy` pairs with `extract_pairs_from_sequence()` to build training data from pipeline tracking output.
- Added `PointCloudCompleter.fine_tune()` interface stub.
- **Step 7 is not wired into the active pipeline.** `main.py` runs steps 1–6 only. `completion.py` exists for future integration.

### 5. evaluate.py fixes

- **Bug fixed:** detection vs GT masks could have different lengths (KD-tree proximity matching). Now uses exact RANSAC inlier indices from `remove_ground`.
- **Refactored:** removed module-level side effects. Added argparse (`--seq`, `--frames`, `--iou-threshold`). `evaluate_frame` takes explicit path parameters.

### 6. Documentation sync

- Updated CLAUDE.md and README.md: 7 stages → 6, DBSCAN → HDBSCAN, removed step 7 references, updated parameter table, removed dead references to `tools/run_sequence.py` and `docs/plan.md`.
- `completion.py` marked as `(dormant)` in architecture listing.

## What's next

### Immediate (open issues)

1. **Tune geometric filters for HDBSCAN.** The higher recall comes with more FPs. Likely candidates: increase `min_volume`, increase `min_points_in_cluster`, or add a compactness/aspect-ratio filter. Run the 20-frame eval after each change to track precision/recall tradeoff.
2. **Full 100-frame evaluation.** Current comparison is 20 frames only. Run `python src/evaluate.py --frames 100` for both methods and record final numbers.

### Medium-term

3. **Completion model integration (Step 7).** Infrastructure is ready in `completion.py`. Needs: choose an architecture (e.g. PCN, SnowflakeNet), train on ShapeNet with `simulate_lidar_noise` augmentation, evaluate with `chamfer_distance` and `f_score`, then wire `completer.complete()` output back into the pipeline loop.
4. **Self-supervised training data.** Run pipeline with `--save-output` on long sequences, then call `KITTIObjectDataset.extract_pairs_from_sequence()` to build real sparse/dense pairs for fine-tuning.

### Low priority

5. **Tracker upgrade.** CentroidTracker works but is prone to ID-switching. IOU-based matching (SORT-style) would improve track consistency if completion uses multi-frame accumulation.
6. **In-window stats overlay.** Attempted Open3D `gui.Application` approach but reverted — requires full visualization rewrite. Consider simpler alternatives (window title, separate stats window, or post-render overlay with PIL).

## Files changed (uncommitted)

```
Modified:  CLAUDE.md, README.md, src/main.py, src/evaluate.py
New:       src/pipeline.py, src/classifier.py, src/completion.py
New:       docs/findings.md, docs/pipeline_feedback.md
Deleted:   docs/plan.md
```

---

# Session Summary — 2026-04-25

## What was done

### 1. ShapeNetCore exploration in notebook

Added section 8 to `notebooks/data_exploratory.ipynb` — explores ShapeNetCore v2 from HuggingFace, focused on three KITTI-relevant categories:

- **car** (`02958343.zip`, ~5.7 GB), **bus** (`02924116.zip`, ~727 MB), **motorcycle** (`03790512.zip`, ~572 MB)
- Subsections 8.1–8.7: repo listing via `HfApi`, WordNet synset resolution, KITTI mapping, zip structure inspection, sample model visualization with `trimesh`, mesh statistics, complete vs simulated partial LiDAR comparison.
- All data fetched dynamically from HuggingFace API — no hardcoded values.

### 2. New dependencies installed

- `huggingface_hub` — download from gated HF repos (`pip install huggingface_hub`)
- `trimesh` — load OBJ meshes, sample point clouds from surfaces (`pip install trimesh`)

### 3. HuggingFace authentication

- Logged in via `from huggingface_hub import login; login(token='...')`. Token stored locally.
- `huggingface-cli login` is **deprecated and broken** — always use the Python API.
- Listing repo files works without auth; downloading from gated repos requires auth.

### 4. Session summary skill created

- Created `.claude/skills/session-summary/SKILL.md` — triggers on "update session summary", "wrap up", "end of session", etc.
- Reads `docs/session_summary.md`, reviews conversation, appends new dated entry, shows draft before writing.

### 5. CLAUDE.md updated

- Added directive at top: "Read `docs/session_summary.md` for a recap of previous sessions."

## Environment notes

- ShapeNet zip naming: WordNet 3.0 noun synset offsets (8-digit). Resolve with `nltk.corpus.wordnet.synset_from_pos_and_offset('n', int(sid))`.
- OBJ models pre-aligned: +Y up, -Z front. Path inside zip: `<synsetId>/<modelId>/models/model_normalized.obj`.
- `DATA.md` in the HF repo documents format and alignment conventions.

## What's next

### Immediate

1. **Run notebook end-to-end** to verify all ShapeNet cells execute (Ran, verified).
2. **Mesh stats for car** (~7k models) may be slow — consider sampling a subset.

### Medium-term

3. **Connect ShapeNet to completion pipeline.** Use `trimesh.sample()` for dense point clouds, apply `simulate_lidar_noise()` from `completion.py` for partial inputs, train completion network.
4. **Scale analysis.** ShapeNet models are normalized — need real-world scaling factors to match KITTI object dimensions.

## Files changed

```
Modified:  CLAUDE.md, .gitignore, docs/session_summary.md, notebooks/data_exploratory.ipynb
New:       .claude/skills/session-summary/SKILL.md
```

---

# Session Summary — 2026-05-05

## What was done

### 1. PCN model implemented (`src/pcn.py`)

Built Point Completion Network (Yuan et al., 2018) from scratch:

- **PCNEncoder**: Stacked PointNet — Conv1d 3→128→256 (stage1), concat global feature back, Conv1d 512→512→1024 (stage2), max pool → (B, 1024). 6.87M params total.
- **PCNDecoder**: FC 1024→1024→1024→3072 → coarse (B, 1024, 3); 4×4 grid folding with fold_mlp Conv1d 1029→512→512→3 → fine (B, 16384, 3).
- **`chamfer_distance_chunked`**: Per-sample loop with `torch.cdist`, chunk_size=2048. Avoids 16384×16384 distance matrix OOM.
- **`pcn_loss`**: Coarse CD (1024 vs 1024) + 0.5 × fine CD (subsampled to 4096 vs 4096 via `torch.randint` + `torch.gather`).

### 2. Training script (`src/train_pcn.py`)

- **`ShapeNetCompletionDataset`**: Loads OBJ meshes with trimesh, scales to real-world dims (car=4.5m, bus=10m, motorcycle=2.2m), samples 16384 GT points, generates partials via Open3D `RaycastingScene` depth rendering (256×256) from random viewpoints (elev [-20°,30°], azim [0°,360°], radius [2.5,4.5]m), back-projects to 3D, subsamples to 2048.
- Training: Adam lr=1e-4, StepLR (×0.5 every 40 epochs), batch=8, 100 epochs. Logs to CSV, checkpoints every 10 epochs + best.
- 3,834 train samples, ~0.69s per sample load time.
- **Training completed** (100 epochs, ~18h total on RTX 3070 Ti).

### 3. PCN training results

Training ran 100 epochs on 3,834 ShapeNet samples (car/bus/motorcycle). Key metrics:

| Epoch | Train Loss | Val Loss | Val CD Coarse | Val CD Fine | Val F-Score | LR     |
| ----- | ---------- | -------- | ------------- | ----------- | ----------- | ------ |
| 1     | 0.672      | 0.586    | 0.396         | 0.380       | —           | 1e-4   |
| 10    | 0.463      | 0.448    | 0.317         | 0.261       | 0.748       | 1e-4   |
| 40    | 0.403      | 0.400    | 0.287         | 0.225       | 0.809       | 5e-5   |
| 58    | 0.384      | 0.376    | 0.272         | 0.210       | —           | 5e-5   |
| 80    | 0.380      | 0.374    | 0.270         | 0.209       | 0.839       | 2.5e-5 |
| 100   | 0.371      | 0.375    | 0.270         | 0.210       | 0.841       | 2.5e-5 |

- **Best checkpoint** (`pcn_best.pth`): epoch 80, val_loss=0.374, F-Score=0.839.
- Loss converged around epoch 60–70; final 30 epochs show minimal improvement (val loss plateau ~0.374).
- LR schedule: 1e-4 → 5e-5 (epoch 40) → 2.5e-5 (epoch 80). Each LR drop gave a small improvement.
- F-Score peaked at 0.841 (epoch 100), indicating good shape reconstruction quality on ShapeNet.
- Checkpoints saved: every 10 epochs + best + last (13 files in `checkpoints/`).

### 4. Dataset documentation (`docs/datasets.md`)

Documents both datasets: SemanticKITTI (22 seqs, 41,624 frames, file formats, thing classes, coordinate system) and ShapeNetCore v2 (4,809 models: 3,533 car, 939 bus, 337 motorcycle; OBJ format, alignment conventions).

### 5. PCN architecture documentation (`docs/pcn/`)

Three detailed breakdown docs:

- `pcn_encoder.md` — step-by-step encoder with tensor shapes, stacked vs vanilla PointNet comparison
- `pcn_decoder.md` — coarse FC, 2D grid construction, tiling mechanics, folding MLP, residual offsets
- `pcn_loss.md` — Chamfer Distance explanation, chunking algorithm, memory budget table, subsampling via `torch.gather`
- `pcn.md` — main overview linking to the three sub-docs

### 6. PyTorch with CUDA installed

`pip install torch --index-url https://download.pytorch.org/whl/cu124` — verified CUDA available on RTX 3070 Ti.

## Environment notes

- User's markdown viewer doesn't support LaTeX `$...$` — use backtick code formatting for math expressions.
- Open3D `RaycastingScene` tensor API is the correct way to generate synthetic partial point clouds (viewpoint-dependent occlusion).

## What's next

_See 2026-05-06 for updated next steps._

## Files changed

```
Modified:  .gitignore, docs/references.bib
New:       src/pcn.py, src/train_pcn.py, docs/datasets.md, docs/pcn/pcn.md, docs/pcn/pcn_encoder.md, docs/pcn/pcn_decoder.md, docs/pcn/pcn_loss.md
```

---

# Session Summary — 2026-05-06

## What was done

### 1. Progress report written

- Created LaTeX progress report (`docs/report/progress_report_2026_05_06.tex`) covering full project history (Feb–May 2026).
- Follows FPT university template format (header/footer, title page, IEEEtran bibliography).
- Recent work (April 25 onward) highlighted with a blue `tcolorbox` banner.
- Compiled successfully to 9-page PDF. Advisor name corrected to Dr. Doan Nhat Quang.

### 2. LaTeX report skill created

- Created `.claude/skills/latex-report/SKILL.md` — triggers on "report", "thesis", "writeup", "latex", etc.
- Auto-gathers context from session_summary.md, findings.md, git log, training logs, references.bib.
- Presents summary and asks for input before writing. Outputs to `docs/report/`.

### 3. Report template files organized

- Renamed `projectthesis_presentation-MSME-Report Writing Guidelines-20221117.pdf` → `docs/report-guidelines.pdf`
- Renamed `report-guideline-resources/` → `docs/report-template/`

### 4. BEV representation idea noted

- Added finding #3 to `docs/findings.md`: Bird's-Eye View (BEV) representation as future research direction — projects 3D LiDAR to 2D top-down view for efficient conv-based architectures.

## What's next

### Immediate

1. **Qualitative evaluation.** Visualize PCN completions on ShapeNet test samples (input → coarse → fine).
2. **Evaluate on KITTI objects.** Run trained PCN on real partial point clouds from the segmentation pipeline — key domain-transfer test.
3. **Add EMD loss.** Sinkhorn approximation for coarse output.

### Medium-term

4. **Domain adaptation.** Fine-tune with `simulate_lidar_noise()` augmentation on real KITTI data.
5. **Try stronger architectures.** PoinTr or SeedFormer if PCN quality is insufficient.
6. **Explore BEV representation.** Investigate BEV-based models as an alternative/complement to 3D point-based pipeline.

## Files changed

```
Modified:  docs/session_summary.md, docs/findings.md
New:       .claude/skills/latex-report/SKILL.md, docs/report/progress_report_2026_05_06.tex, docs/report/Images/fpt.png
Renamed:   docs/report-guidelines.pdf (from long filename), docs/report-template/ (from report-guideline-resources/)
```

---

# Session Summary — 2026-05-11

## What was done

### 1. Answered advisor's review questions

Created `docs/advisor_questions_2026_05_10.md` responding to 4 questions from Dr. Doan Nhat Quang about the progress report:

1. Stage 6 (classification) — confirmed implemented in `src/classifier.py`
2. PCN trained on ShapeNet — yes; tested on KITTI LiDAR — not yet
3. Phase 2 continues from full Phase 1 output (segmented + tracked objects)
4. Stage 6 is in Phase 1 (before completion), not after Phase 2

Translated to Vietnamese with natural advisor-student tone.

### 2. Pipeline diagram for report (`docs/report/figures/pipeline_diagram.tex`)

Created TikZ flowchart showing full pipeline with per-stage In/Out/Method annotations:

- Phase 1 (blue, Stages 1–6 + Tracking) and Phase 2 (green/orange, Stage 7) in separate dashed boxes
- Fixed two layout issues: phase box outlines clipping IO text, Phase 2 label overlapping arrow
- Compiled PDF: `docs/report/figures/pipeline_diagram.pdf`

## What's next

### Immediate

1. **Qualitative evaluation.** Visualize PCN completions on ShapeNet test samples (input → coarse → fine).
2. **Evaluate on KITTI objects.** Run trained PCN on real partial point clouds from the segmentation pipeline — key domain-transfer test.
3. **Add EMD loss.** Sinkhorn approximation for coarse output.
4. **Include pipeline diagram in report.** Use `\resizebox{\textwidth}{!}{\input{...}}` in the main report .tex.

### Medium-term

5. **Domain adaptation.** Fine-tune with `simulate_lidar_noise()` augmentation on real KITTI data.
6. **Try stronger architectures.** PoinTr or SeedFormer if PCN quality is insufficient.
7. **Explore BEV representation.** Investigate BEV-based models as alternative/complement to 3D point-based pipeline.
8. **Tune geometric filters.** Tighten filters to reduce HDBSCAN false positives.

## Files changed

```
New:       docs/advisor_questions_2026_05_10.md, docs/report/figures/pipeline_diagram.tex, docs/report/figures/pipeline_diagram.pdf
```

---

# Session Summary — 2026-05-14

## What was done

### 1. PCN training bug fixes (`src/pcn.py`, `src/train_pcn.py`)

Four correctness issues fixed:

- **Camera radius inside buses.** Fixed radius `(2.5, 4.5)` placed camera inside ~10m buses. Changed to scale-proportional factor: `r = uniform(1.5×scale_m, 2.5×scale_m)`.
- **Scale-biased Chamfer loss.** Buses contributed ~4.5× more to CD loss. Added unit-sphere normalization of both GT and partial before loss.
- **Non-random val split.** Sorted ShapeNet hex-hash IDs shuffled with `default_rng(split_seed=42)` before split.
- **Recursive fallback in `__getitem__`.** Replaced with bounded flat retry loop (max 10), raises `RuntimeError` on exhaustion.
- Memory estimate in `chamfer_distance_chunked` docstring corrected (128MB, not 384MB).

Old PCN checkpoint is invalidated — must retrain.

### 2. Learned classifier — Stage A (`src/classifier.py`, `src/train_classifier.py`)

Replaced heuristic bbox classifier with dual-branch PointNet. 4 classes: car (0), bus (1), motorcycle (2), unknown (3).

- **Architecture:** `PointNetClassifier` (80K params). Point branch: Conv1d 3→64→128→256 + max pool. Bbox branch: Linear 8→32. Head: 288→128→Dropout(0.3)→4.
- **Bbox features:** 8-dim metric-scale vector (sorted extents, volume, aspect ratios, log count, height). OBB if ≥20 pts, AABB fallback.
- **Unknowns:** 50% geometric primitives, 25% axis-cropped ShapeNet partials, 25% noisy vehicle subsets. Separate train/val seeds. Synthetic-only — not evidence of real-world rejection.
- **Training:** CrossEntropyLoss with `1/sqrt(counts)` class weights, Adam lr=1e-3, StepLR(30, 0.5), 50 epochs, batch 32. Checkpoint on best unthresholded val macro F1.

### 3. Pipeline integration (`src/main.py`)

- New CLI flags: `--classifier-ckpt`, `--classifier-unknown-threshold 0.65`, `--no-learned-classifier`.
- Safe loading: missing checkpoint → warn, fall back to heuristic.
- Assert label/point alignment before classification.

### 4. Code review fixes (two rounds)

Round 1:

- Training subsampling was accidentally deterministic (`rng=None` → `default_rng(0)`). Fixed to `default_rng()` (unseeded).
- `get_raw_sample()` used global RNG — not reproducible. Fixed with per-index seeded RNG threaded through rendering path.
- Resume/eval-only recomputed bbox stats instead of loading from checkpoint. Fixed to load from checkpoint when available.
- Richer synthetic unknowns (cropped partials + noisy subsets added).
- `classification_report()` now passes explicit `labels=list(range(NUM_CLASSES))`.
- `load_classifier()` validates `class_labels` on checkpoint load.
- `compute_cluster_extent()` docstring fixed ("sorted" → "unsorted").

Round 2:

- Full determinism: seeded RNG threaded through `_get_vehicle_raw` → `_render_partial` → `_random_extrinsic` and retry fallback.
- `load_classifier()` now validates `bbox_feat_dim` and `num_points`, reads from top-level checkpoint keys (authoritative) before falling back to config.
- Synthetic unknown caveat printed at end of training.

## What's next

### Immediate

1. **Train classifier.** `python src/train_classifier.py` — no checkpoint exists yet; pipeline falls back to heuristic.
2. **Retrain PCN.** `python src/train_pcn.py` — old checkpoint invalidated by camera radius, unit-sphere loss, and seeded split fixes.
3. **Pipeline smoke test.** `python src/main.py --no-gui --save-output --seq 00 --frames 20` — verify classifier integration end-to-end.
4. **Tighten geometric filters.** Precision is 0.154 with HDBSCAN — too many false positives. Increase `min_volume`, `min_points_in_cluster`, or add aspect-ratio filter.

### Medium-term

5. **Wire PCN into pipeline.** Connect completion to tracked objects in `main.py`; currently dormant in `completion.py`.
6. **Stage B classifier evaluation.** Real LiDAR cluster evaluation with SemanticKITTI labels for actual rejection quality.
7. **Compare learned vs heuristic classifier.** Run with/without `--no-learned-classifier`, compare tracks.json output.
8. **Qualitative PCN evaluation.** Visualize completions on ShapeNet test samples (input → coarse → fine).
9. **Evaluate PCN on KITTI objects.** Run trained PCN on real partial point clouds — key domain-transfer test.
10. **Include pipeline diagram in report.**

### Low priority

11. **Add EMD loss.** Sinkhorn approximation for PCN coarse output.
12. **Domain adaptation.** Fine-tune PCN with `simulate_lidar_noise()` augmentation on real KITTI data.
13. **Try stronger architectures.** PoinTr or SeedFormer if PCN quality is insufficient.
14. **Explore BEV representation.** Investigate BEV-based models as alternative/complement.
15. **Tracker upgrade.** IOU-based matching (SORT-style) if completion uses multi-frame accumulation.

---

# Session Summary — 2026-05-14 (cont.)

## What was done

### 1. FP analysis script (`src/analyze_fp.py`)

Created false-positive analysis tool that runs pipeline stages 1-5 on SemanticKITTI, matches detections to GT via greedy IoU, and writes per-cluster features to CSV (18 columns). Prints TP and FP distributions side-by-side (semantic breakdown + percentiles for 6 features).

- Usage: `python src/analyze_fp.py --seq 00 --frames 100`
- Output: `output/eval_fp_analysis_{seq}.csv`
- Baseline result (100 frames, seq 00): 5882 detections, 1393 TP, 4489 FP (precision 0.237)
- Top FP sources: vegetation (38.3%), building (32.9%), trunk (12.5%)

### 2. Geometric filter tuning (`src/pipeline.py`)

Data-driven filter tuning based on FP analysis distributions:

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `max_center_height_above_ground` | 3.0 | 1.5 | TP median 0.81m, building/trunk FP medians >1.7m |
| `max_height_span` | *(none)* | 1.8 | TP p95 = 1.66m; vegetation/building FP medians >1.5m |
| `max_aspect_max_min` | *(none)* | 6.0 | TP p95 = 5.79; building FP median 5.34 |

Post-tuning results (100 frames, seq 00):

| Metric | Before | After | Change |
|---|---|---|---|
| Precision | 0.237 | 0.654 | +176% |
| Recall | 0.868 | 0.704 | -19% |
| F1 | — | 0.678 | — |
| FP count | 4489 | 676 | -85% |
| Mean IoU | — | 0.944 | — |

Residual FPs: 64% vegetation (short, compact clusters geometrically overlapping with vehicles).

### 3. Distance-dependent evaluation

Precision by distance: 0-20m = 0.692, 20-40m = 0.635, 40m+ = 0.367. 95% of TPs within 40m. Point density drops ~5x from near to far. Far-range FPs mostly "unlabeled" (likely GT annotation gaps).

### 4. Findings recorded (`docs/findings.md`)

Added findings #4 (FP analysis + filter tuning) and #5 (distance-dependent evaluation).

### 5. Stage B mining script designed (not written)

Designed `src/mine_stage_b.py`: run pipeline on SemanticKITTI, match clusters to GT, save centroid-centered .npy files organized by class (car/bus/motorcycle/unknown). User deferred writing to next session.

## Files changed

```
Modified:  docs/findings.md (added findings #4, #5)
Modified:  docs/project_state.md (precision-first plan update)
Modified:  src/pipeline.py (3 filter changes in PIPELINE_CONFIG + filter_clusters)
New:       src/analyze_fp.py
```

---

# Session Summary — 2026-05-15

## What was done

### 1. Reviewed completed training results
- Classifier Stage A: 50 epochs, val_acc 99.93%, val_macro_f1 99.95% (on synthetic data).
- PCN: 2×100 epochs, val_cd_fine 0.066, val_fscore 99.37%.

### 2. Replaced synthetic unknowns with real ShapeNet categories
- Created `src/download_shapenet.py` to download all 55 ShapeNetCore categories from HuggingFace (~51K models).
- Rewrote unknown generation in `src/train_classifier.py`: removed 8 geometric primitives, replaced with 52 non-vehicle ShapeNet categories (38,170 models). Rendered via same partial-view pipeline as vehicles.
- Fixed checkpoint selection: `val_macro_f1` → `val_macro_f1_thresh` to match thresholded pipeline behavior.
- New dataset: 3,835 vehicle + 1,643 unknown = 5,478 samples. Estimated ~10-11 hours for 50 epochs.
- Classifier retraining started by user (running in background).

### 3. Built Stage B mining script (`src/mine_stage_b.py`)
- Mines real LiDAR clusters from SemanticKITTI sequences using pipeline stages 1-5.
- Propagates GT semantic labels via KD-tree nearest-neighbor after voxel downsampling.
- Classifies clusters by purity threshold (default 0.75), saves centroid-centered `.npy` files organized by class.
- Outputs `metadata_{split}.json` with config, class counts, per-cluster stats (purity, raw_sem_purity, histogram).
- Applied robustness fixes: bin/label assertion, cluster alignment assertion, empty-frame guards, discard tracking.
- Sanity check (seq 00, 100 frames): 1,979 clusters, 17 discarded. car 1,246 / unknown 684 / motorcycle 32 / bus 0.

### 4. Full mining run started
- Train: `--seq 00 01 02 03 04 05 06 07 09 10 --frames 5000 --split train` (~6-7 hours)
- Val: `--seq 08 --frames 5000 --split val` (~70 min)

### 5. CLAUDE.md improvements
- Fixed activation command for Windows.
- Added training/mining commands, key files section, Stage A/B strategy docs.

### 6. Finding #6 recorded
- Stage B mining class distribution from seq 00 sanity check. Confirms Stage A+B strategy necessity.

### 7. Val mining completed
- Val split (seq 08): 130,182 clusters — car 25,047 / unknown 104,659 / motorcycle 476 / bus 0.
- Train split (seq 00-07,09-10): 420,130 clusters — car 88,491 / unknown 329,961 / motorcycle 1,650 / bus 28.

### 8. Stage B fine-tuning code
- Added `StageBDataset` + `STAGE_B_CONFIG` to `src/train_classifier.py`.
- `--stage-b` flag: auto-loads Stage A checkpoint, recomputes bbox stats from 50K real samples, fresh optimizer, LR 1e-4, batch 64, 15 epochs.
- Separate checkpoints: `stage_b_best.pth`, `stage_b_last.pth`, `stage_b_training_log.csv`.
- Handles missing classes (bus=0 in val) with zero weight.

### 9. Evaluation target class fix
- `evaluate.py` was penalizing classifier for correctly rejecting unsupported classes (pedestrians, trucks, etc.).
- Added `--target` flag: `all-things` (default) vs `supported-vehicles` (car/bus/motorcycle only).
- Added bin/label file count validation.

### 10. Code review fixes
- `--eval-only` requires `--resume`. `compute_bbox_stats()` reports failures. Fixed duplicate comment.

### 11. Findings #7-#9 recorded
- #7: Stage A domain gap — 0/1279 TP on real data.
- #8: Full mining results with train/val split tables.
- #9: Evaluation target class mismatch and fix.

### 12. Thesis proposal fix
- `\clearpage` before `\bibliographystyle` to prevent Timeline table floating into references.

## Files changed

```
Modified: CLAUDE.md, docs/findings.md, src/train_classifier.py, src/evaluate.py, docs/report/thesis_proposal.tex
New:      src/download_shapenet.py, src/mine_stage_b.py
```

---

# Session Summary — 2026-05-16

## What was done

### 1. Stage B evaluation and threshold optimization
- Stage B training completed by user (val_macro_f1_thresh 0.7056, 15 epochs).
- Evaluated on pipeline: Precision 0.969, Recall 0.680, F1 0.799 (at default threshold 0.65).
- Full threshold sweep (0.30–0.95, 12 values): optimal at 0.50 → F1 0.816, Precision 0.957, Recall 0.711.
- Updated `unknown_threshold` default from 0.65 to 0.50 across `src/classifier.py`, `src/evaluate.py`, `src/main.py`, `src/train_classifier.py`.
- Findings #10 (Stage B eval) and #11 (threshold sweep) recorded.

### 2. Thesis proposal rewrite (`docs/report/thesis_proposal.tex`)
- Complete rewrite per advisor feedback: too long, no results, 2-3 pages.
- Removed: results/metrics, parameter tables, timeline, ablations, standalone Related Work, references.
- New structure: Problem Statement → Proposed Approach → Issues and Challenges → Expected Contributions → Evaluation Plan.
- Three verifiable contributions: modular pipeline, PCN completion with CD+EMD, two-stage training strategy.

### 3. Full codebase code review (11/11 files)
- Reviewed all `src/*.py` files line-by-line: `pipeline.py`, `tracker.py`, `classifier.py`, `pcn.py`, `completion.py`, `evaluate.py`, `main.py`, `mine_stage_b.py`, `analyze_fp.py`, `train_classifier.py`, `train_pcn.py`.
- Refactored 2 functions for cognitive complexity: `filter_clusters` (25→~4), `tracker.update` (18→~8). Both via helper extraction.
- Added docstrings to all public functions/classes across 10 of 11 files (skipped `completion.py`).
- Fixed broken import in `analyze_fp.py`: `THING_CLASSES` → `THING_CLASSES_ALL` (would crash at runtime).
- Documented classifier architecture decisions vs original PointNet paper (Finding #12).

### 4. Trimesh → Open3D migration (`src/train_pcn.py`)
- Replaced all `trimesh` usage with Open3D equivalents: mesh loading, surface sampling, attribute names.
- Both training scripts (`train_classifier.py`, `train_pcn.py`) now use Open3D consistently.
- No retraining needed — existing PCN checkpoint unaffected.

## Files changed

```
Modified: src/pipeline.py, src/tracker.py, src/classifier.py, src/pcn.py,
          src/evaluate.py, src/main.py, src/mine_stage_b.py, src/analyze_fp.py,
          src/train_classifier.py, src/train_pcn.py,
          docs/findings.md, docs/project_state.md, docs/report/thesis_proposal.tex
```

---

# Session Summary — 2026-05-27

## What was done

### 1. Track-level filtering implementation
- Implemented two-mechanism temporal consistency filtering: (1) minimum track length, (2) majority class vote with evidence thresholds (`min_known_votes=2`, `min_known_ratio=0.5`, tie rejection).
- Added `resolve_track_class()` to `src/main.py` and `src/evaluate.py`.
- Converted `src/main.py` from frozen first-frame class to per-frame class vote accumulation (`track_class_votes: dict[int, list[str]]`).
- Rewrote `src/evaluate.py` with two-pass offline tracked evaluation: extracted `get_frame_detections()`, added `_CentroidProxy` for global-frame centroid tracking, pass 1 accumulates tracks (all detections including "unknown"), pass 2 evaluates only surviving tracks.
- Added `--no-track-filter` and `--min-track-length` CLI flags to `src/evaluate.py`.
- Added 4 config params to `PIPELINE_CONFIG`: `min_track_length`, `track_class_vote`, `min_track_known_votes`, `min_track_known_ratio`.

### 2. Track-level filtering sweep and validation
- Swept `min_track_length` in {1, 2, 3, 5} on seq 00, 100 frames.
- Best F1 at `min_track_length=2`: 0.834 (up from 0.810 per-frame baseline).
- Key insight: track filtering primarily **boosted recall** (not precision as hypothesized) by recovering detections flickering between "car" and "unknown" via majority vote.
- Full-sequence validation (4541 frames): F1 0.801 (up from 0.762), precision 0.918, recall 0.710. Improvement larger than 100-frame estimate (+0.039 vs +0.024).

### 3. Qualitative PCN evaluation
- Created `src/visualize_pcn.py`: loads PCN from checkpoint, runs inference on ShapeNet val samples (3 per category), computes CD and F-score, renders side-by-side images or interactive Open3D windows.
- Results: mean CD_fine=0.047, mean F-score=99.23% across 9 samples (bus/car/motorcycle).
- Visual observation: fine output shows 2D grid-folding artifacts (vertical striping). Quantitatively solid but visually distinguishable from GT.
- Saved images to `output/pcn_qualitative/`.

## Files changed

```
Modified: src/pipeline.py, src/main.py, src/evaluate.py,
          docs/findings.md, docs/project_state.md
New:      src/visualize_pcn.py
```

## Results / findings

- Finding #13: Track-level filtering hypothesis recorded.
- Finding #14: Full implementation results — `min_track_length=2` is optimal, F1 0.801 on full sequence (was 0.762).
- PCN completions: quantitatively good (CD 0.047, F-score 99.2%) but grid-folding artifacts visible.

## Next

1. Wire PCN completion into `main.py`
2. Classifier quality reporting (confusion matrix, FP/FN breakdown)
3. Consider PoinTr/SeedFormer if PCN visual quality is insufficient for thesis figures

---

# Session Summary — 2026-05-27 (cont.)

## What was done

### 1. PCN completion wired into pipeline
- Implemented `PointCloudCompleter` in `src/completion.py`: `_load_model()` (lazy torch import, architecture validation), `_fix_size()` (deterministic via `np.random.default_rng`), `complete()` returning `(points, skip_reason)` tuple, `is_loaded` property.
- Normalization bridge: centroid-subtract → unit-sphere normalize → fix to 2048 pts → PCN forward → denormalize back to world coords.
- Added `--pcn-ckpt`, `--no-completion` CLI flags to `src/main.py`.
- Fail-fast on missing checkpoint when completion is enabled.
- Post-accumulation completion in output loop with explicit class filtering (`pcn_completion_classes: ["car", "bus", "motorcycle"]`), min-point guard (64), skip reason tracking.
- Rich `tracks.json` metadata: `raw_point_count`, `point_count`, `completed`, `completion_enabled`, `completion_method`, `pcn_checkpoint`, `completion_skip_reason`.
- Added 3 config keys to `PIPELINE_CONFIG`: `pcn_min_points`, `pcn_completion_classes`, `pcn_sample_seed`.

### 2. Domain gap evaluation
- Ran pipeline with completion on seq 00, 100 frames, Stage B classifier: 47/47 tracks completed.
- Completed tracks show blobby, unrecognizable output — no vehicle structure.
- Created `src/test_single_frame_pcn.py` to test single-frame clusters (no motion smear). Results identical: car (3477 pts, frame 49) and motorcycle (255 pts, frame 17) both produce shapeless blobs.
- **Conclusion**: ShapeNet-to-LiDAR domain gap is the primary cause, not motion smear from accumulated tracks.

### 3. Visualization tooling
- Created `src/show_completion.py`: matplotlib-based side-by-side raw vs completed rendering. Supports `--all` (batch save) and `--track-id` (single track interactive).
- Comparison images saved to `output/completion_comparison/` and `output/single_frame_pcn/`.

## Files changed

```
Modified: src/completion.py, src/main.py, src/pipeline.py, docs/findings.md
New:      src/show_completion.py, src/test_single_frame_pcn.py
```

## Results / findings

- Finding #15: PCN integration complete, domain gap confirmed as primary quality issue. Fine-tuning required before completion is usable.

## Next

1. Classifier quality reporting (confusion matrix, FP/FN semantic breakdown)
2. PCN domain-adaptation fine-tuning (Stage B for PCN, using `simulate_lidar_noise()`)
3. Pipeline diagram for report

---

# Session — 2026-05-28

## What was done

### 1. PCN paper analysis — domain gap root cause

Read and analyzed the original PCN paper (`docs/PCN Point Completion Network.pdf`). Key findings:
- Paper uses pinhole depth renders from 8 viewpoints for ShapeNet partial inputs — fundamentally different from LiDAR scan-line patterns.
- Paper's KITTI evaluation uses GT bounding boxes for canonical alignment and reports proxy metrics only (no real completion quality).
- Missing EMD loss is **not** the bottleneck — domain gap in input distribution is.

### 2. PCA canonical alignment

Added PCA-based canonical alignment to `PointCloudCompleter.complete()` in `src/completion.py`:
- New `_pca_axes()` static method computes principal axes from covariance matrix.
- Rotates input to canonical orientation before PCN, inverse-rotates output.
- **Result**: No visible improvement — domain gap is in partiality pattern, not rotation.

### 3. Fine-tuning Approach A — noise augmentation on depth renders

Modified `src/train_pcn.py` to support `--finetune-lidar` flag with noise augmentation:
- Added `simulate_lidar_noise()` + random sparsification to `__getitem__` rendering path.
- Added `--pretrained` flag (loads weights only, fresh optimizer) vs `--resume` (full state).
- Added dual validation (clean + lidar-augmented).
- Ran 30 epochs: `--finetune-lidar --pretrained checkpoints/pcn_best.pth --epochs 30 --lr 1e-5`.
- **Result**: Failed. Clean val CD 0.066→0.065, lidar val CD ~0.091. No visible improvement on real data. Adding noise to depth renders doesn't change the fundamental pinhole partiality pattern.

### 4. Fine-tuning Approach B — virtual Velodyne HDL-64E ray-casting

Implemented `_render_lidar_partial()` in `ShapeNetCompletionDataset`:
- 64 beams at −24.9° to +2° elevation, 0.09° horizontal resolution.
- Random sensor placement: 8–50m distance, 2–15° elevation above object.
- Open3D raycasting (`o3d.t.geometry.RaycastingScene`), range-proportional noise.
- Produces realistic point counts (360–2048 unique points).
- Training started: `.venv\Scripts\python.exe src/train_pcn.py --finetune-lidar --pretrained checkpoints/pcn_best.pth --epochs 30 --lr 1e-5`
- **Status**: Training in progress (~6–7 hours estimated). Awaiting evaluation.

### 5. External review feedback

Processed detailed code review feedback on `train_pcn.py`. Accepted `--pretrained` fix (blocking). Pushed back on: batching (research code), validation split (already separate), noise timing (after centering is correct), retry logic (4 attempts is fine).

### 6. Finding #16

Documented "PCN Domain Adaptation — Noise Augmentation Insufficient" in `docs/findings.md`. Covers PCA alignment (no effect), Approach A (failed), Approach B (awaiting evaluation).

## Files changed

```
Modified: src/completion.py, src/train_pcn.py, src/test_single_frame_pcn.py, docs/findings.md
```

## Results / findings

- PCA alignment: no visible improvement on real LiDAR completion.
- Approach A (noise augmentation): failed — clean val CD 0.066→0.065, no improvement on real data.
- Approach B (virtual Velodyne): training in progress, not yet evaluated.
- Finding #16 appended to `docs/findings.md`.

## Next

1. Evaluate Approach B on real LiDAR data (`test_single_frame_pcn.py --pcn-ckpt checkpoints/pcn_lidar_best.pth`)
2. Update Finding #16 with Approach B results
3. Classifier quality reporting (confusion matrix, FP/FN semantic breakdown)

---

# Session — 2026-05-28 (continued)

## What was done

### 1. Approach B evaluation — virtual Velodyne also failed

Evaluated `checkpoints/pcn_lidar_best.pth` on real LiDAR data:
- Lidar val CD: 0.134 → 0.070 (improved on simulated domain)
- Clean val CD: 0.066 → 0.082 (regressed on original domain)
- Real LiDAR output still blobby — no transfer to real data
- All three synthetic domain adaptation approaches exhausted

### 2. Data leak discovered and fixed

Seq 00 was in both Stage B training (seq 00-07, 09-10) and evaluation — inflating reported F1.
- Re-evaluated on held-out seq 08: F1 0.834 → 0.730
- Updated `docs/project_state.md` with corrected metrics
- Classifier confusion matrix on seq 08: car P=93% R=91%, motorcycle R=40%, ~6% unknown→car leakage

### 3. GT-label-based mining script

Created `src/mine_completion_pairs.py` — mines sparse/dense completion pairs using GT SemanticKITTI labels instead of the classifier (eliminates ~6% noise):
- `classify_cluster_gt()` — majority vote on GT semantic labels with purity threshold
- `process_frame()` — pipeline stages 1-5 with KD-tree label propagation through voxel downsampling
- `mine_sequence()` — full sequence tracking + pair saving
- CLI: `--seq`, `--frames`, `--output`, `--purity` (0.75), `--min-track-length` (3)
- Smoke test: 20 frames → 17 tracks, 217 pairs (car: 16, motorcycle: 1)

### 4. Mining infrastructure updates

- `src/main.py`: added `--mine-pairs` flag (saves sparse/dense pairs using classifier-based pipeline)
- `src/completion.py`: updated `KITTIObjectDataset` to handle new `sparse_s{seq}_{id}_f{frame}.npy` naming
- `scripts/mine_all_pairs.ps1`: GT mining script for train (seq 00-07, 09-10) / val (seq 08) split

### 5. Architecture decision

PCN failed with all 3 synthetic approaches. Decision: mine real data + train stronger architecture (PoinTr, SnowflakeNet, or MSN) directly on real pairs.

### 6. Findings #17 and #18

- #17: PCN Domain Adaptation — Virtual Velodyne Also Insufficient
- #18: Evaluation Data Leak — Stage B Classifier Evaluated on Training Data

## Files changed

```
Modified: docs/findings.md, docs/project_state.md, src/completion.py, src/main.py
New: src/mine_completion_pairs.py, scripts/mine_all_pairs.ps1
```

## Results / findings

- Approach B confirmed failed — all synthetic domain adaptation exhausted
- Held-out seq 08 F1: 0.730 (corrected from leaked 0.834)
- Classifier accuracy: ~6% noise rate on mining (unknown→car leakage)
- GT mining eliminates classifier noise entirely

## Next

1. Run `.\scripts\mine_all_pairs.ps1` to mine real pairs (~5 hours)
2. Choose completion architecture (PoinTr / SnowflakeNet / MSN) and integrate
3. Train on real pairs, evaluate CD + F-Score on seq 08
4. Commit all uncommitted changes

---

# Session — 2026-06-02

## What was done

### 1. PCN sparse-input experiment (from prior session, carried over)
- Trained PCN with sparse ShapeNet inputs (32-256 random points) per advisor suggestion
- Result: completions are scattered noise on real LiDAR — domain gap is structural, not density-related
- Findings #19 recorded

### 2. GT vs Pipeline visualization tool (`src/visualize_gt.py`)
- Created toggle visualization: SPACE switches between GT semantic labels and pipeline detections
- Per-frame log: `Frame 005 [GT] | detected: 3 car, 12 unknown | GT: 5 car, 2 motorcycle`
- Fixed classifier checkpoint: was using Stage A (`classifier_best.pth`), switched to Stage B (`stage_b_best.pth`)

### 3. Classifier revamp — binary car/not-car
- Analyzed seq 07/08 detection logs: all detections classified as "motorcycle", zero cars
- Decision: simplify from 4-class to binary (car / not-car)
- Modified 7 files: `classifier.py`, `train_classifier.py`, `mine_stage_b.py`, `evaluate.py`, `pipeline.py`, `main.py`, `test_single_frame_pcn.py`
- `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- Stage A: only ShapeNet car (02958343) as positive, `unknown_fraction` raised to 0.50
- `THING_CLASSES_SUPPORTED = {10, 252}` (car + moving-car only)
- Findings #20 recorded

### 4. Occupancy Networks paper
- User reviewing "Occupancy Networks: Learning 3D Reconstruction in Function Space" — discussion deferred to next session

## Files changed

```
Modified: src/classifier.py, src/train_classifier.py, src/mine_stage_b.py,
          src/evaluate.py, src/pipeline.py, src/main.py, src/test_single_frame_pcn.py,
          src/completion.py, src/train_pcn.py,
          docs/findings.md, docs/project_state.md, docs/session_history.md
New:      src/visualize_gt.py
```

## Results / findings

- PCN sparse training: failed — domain gap is structural (finding #19)
- 4-class classifier produces false motorcycle detections on seq 08 (GT has zero motorcycles)
- Binary car/not-car revamp complete in code, training pending

## Next

1. Train binary classifier Stage A: `python src/train_classifier.py --epochs 50`
2. Re-mine Stage B data: `python src/mine_stage_b.py --seq 00 01 02 03 04 05 06 07 09 10 --frames 5000 --split train` then val on seq 08
3. Train Stage B: `python src/train_classifier.py --stage-b --epochs 15`
4. Evaluate on seq 08: `python src/evaluate.py --seq 08 --frames 100 --target supported-vehicles --classifier-ckpt checkpoints/stage_b_best.pth`
5. Discuss Occupancy Networks paper and implications for completion
6. Commit all uncommitted changes
