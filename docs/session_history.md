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

---

# Session — 2026-06-02 (continued)

## What was done

### 1. Binary classifier training pipeline — complete
- Stage A: already trained (F1 0.9986 from prior session part)
- Mined Stage B train data: seqs 00-07,09-10, 5000 frames, purity 0.75
  - Result: 420,333 clusters (88,600 car / 331,733 not-car), 2,866 discarded
- Mined Stage B val data: seq 08, 5000 frames, purity 0.75
  - Result: 130,394 clusters (24,968 car / 105,426 not-car), 958 discarded
- Trained Stage B: 15 epochs, best epoch 13, macro F1 0.9225
  - car: P=0.837 R=0.900 F1=0.868 | not-car: P=0.976 R=0.959 F1=0.967
  - Checkpoint: `checkpoints/stage_b_best.pth`

### 2. End-to-end evaluation
- Baseline (seq 08, 100 frames): P=0.963 R=0.708 F1=0.816 MeanIoU=0.888
- Compared to old 4-class: P=0.728 R=0.732 F1=0.730 → binary is substantially better

### 3. Parameter sweep for recall improvement
- Track filter sweep (min_track_length, min_known_votes, min_known_ratio): relaxing votes/ratio hurt precision without meaningful recall gain
- Geometric filter sweep (min_points_in_cluster, hdbscan_min_cluster_size): no improvement
- Classifier threshold sweep (unknown_threshold 0.30, 0.40): no improvement
- Tracker sweep (max_distance 3.0/4.0, max_disappeared 8/10): all hurt F1
- RANSAC sweep (ransac_distance_threshold 0.15, 0.25): 0.15 showed slight recall gain but within noise
- Added CLI sweep flags to `src/evaluate.py`: `--min-known-votes`, `--min-known-ratio`, `--min-points`, `--hdbscan-min-cluster-size`, `--tracker-max-distance`, `--tracker-max-disappeared`, `--ransac-distance-threshold`

### 4. RANSAC variance measurement
- 3 identical baseline runs: F1 range 0.804–0.828, recall range 0.692–0.724
- True baseline: F1 ~0.819 ± 0.024, Recall ~0.713 ± 0.032
- Conclusion: RANSAC randomness dominates — parameter tuning differences were within noise

### 5. Completion architecture review
- Read Occupancy Networks paper — ruled out (user wants point cloud output, not mesh)
- Read PoinTr and SnowflakeNet papers — detailed comparison delivered
- Recommendation: PoinTr (KITTI-proven, established codebase, transformer architecture)

### 6. Mining code audit
- Reviewed `src/mine_stage_b.py` for bugs — no critical issues found
- Minor: misleading variable name `pcd_cluster` (actually a bbox), voxel label propagation is an approximation

## Files changed

```
Modified: src/evaluate.py (added 7 CLI sweep flags for parameter tuning)
          src/pipeline.py (min_track_length temporarily changed to 1, reverted to 2)
```

Note: Stage B data mined to `dataset/stage_b/{train,val}/`, checkpoint saved to `checkpoints/stage_b_best.pth` — these are generated artifacts, not tracked in git.

## Results / findings

- Binary classifier end-to-end: F1 ~0.819 ± 0.024 (mean of 3 runs), up from 4-class F1 0.730
- Precision dramatically improved: ~0.961 vs old 0.728
- Recall ceiling ~0.71 is set by single-frame HDBSCAN clustering, not post-processing
- All parameter sweeps within RANSAC noise — no tuning gains found
- RANSAC seed should be fixed for reproducible experiments

## Next

1. Fix RANSAC seed for reproducible evaluation
2. Explore temporal point aggregation (multi-frame clustering) to break recall ceiling
3. Decide on PoinTr implementation for completion (user leaning toward it)
4. Commit current changes

---

# Session — 2026-06-04

## What was done

### 1. Pipeline feedback review (`docs/pipeline_feedback.md`)
- Assessed all 5 feedback points against current pipeline state
- Already addressed: HDBSCAN (not DBSCAN), ground-plane-relative height filter, learned PointNet classifier (not heuristics), PCN abandoned
- Still relevant: single-plane RANSAC on slopes (Patchwork++), centroid tracker ID-switching (SORT), occlusion splitting
- Confirmed current PointNet classifier is sufficient for binary car/not-car (F1 0.92); PointNet++ upgrade not justified until multi-class

### 2. Temporal point aggregation — implemented and abandoned
- **Hypothesis:** Accumulate object points from N consecutive frames before HDBSCAN clustering to break recall ceiling (~0.74) by making sparse distant cars denser
- **Implementation:** Added `temporal_window` config param, temporal buffer (deque) in `main.py` and `evaluate.py`, global-frame accumulation, ground plane transformation, current-frame-only cluster attribution
- **Result with temporal_window=3:** F1 collapsed from 0.844 to 0.073 (Recall 0.039)
  - Track filter accepted only 7/518 tracks (was 47/459 at baseline)
  - Root cause: HDBSCAN on 3x more points creates fundamentally different cluster boundaries, not just denser versions of existing clusters. With 134k accumulated points, HDBSCAN produced 645+ clusters vs ~240 single-frame, but fewer passed geometric filters (17 vs 34)
  - Fix attempt: classify using current-frame points only → no improvement (problem is upstream in clustering, not classification)
- **Diagnostic:** Single-frame HDBSCAN on ~45k points → ~230 clusters, 30-45 pass filters. Accumulated 3-frame → ~650 clusters, 17-22 pass. The clustering landscape is completely reshaped.
- **Decision:** Reverted all temporal aggregation code. Naive "cluster the union" approach is fundamentally flawed for density-based clustering.

### 3. Option 2 — lower cluster size thresholds
- **Hypothesis:** Lower `hdbscan_min_cluster_size` (10→5) and `min_points_in_cluster` (15→8 or 5) to capture sparse car clusters, trust classifier + track filter to reject noise
- **Result:** Zero change in metrics (TP=1205, FP=20, FN=426 identical to baseline)
- **Root cause analysis:** GT car point count distribution shows 96.7% of cars have >10 points, 94.3% have >15. The 426 missed cars aren't failing due to cluster size — they fail because HDBSCAN doesn't form clean clusters for them (merging with adjacent objects, splitting across clusters, or geometric filter rejection)

### 4. Recall bottleneck analysis
- Analyzed GT car point counts across 100 frames of seq 00:
  - 1680 total GT car instances (1631 eval-eligible with ≥10 points)
  - Distribution: min=1, median=189, mean=458, max=3506
  - >5 pts: 98.5%, >10 pts: 96.7%, >15 pts: 94.3%, >50 pts: 76.4%
- **Conclusion:** Recall ceiling (~0.74) is NOT from point sparsity or cluster size thresholds. It's from HDBSCAN cluster quality: merging adjacent cars, splitting single cars, and geometric filter rejection of valid but oddly-shaped clusters.
- **Next investigation needed:** Which geometric filter rejects the most real cars? What is the HDBSCAN merge/split rate on GT cars?

## Files changed

No file changes detected (all temporal aggregation code was implemented and fully reverted within the session).

## Results / findings

- **Baseline (deterministic, from prior session):** P=0.984, R=0.739, F1=0.844, mIoU=0.943 (seq 00, 100 frames, track filter ON)
- Temporal aggregation (naive union clustering) is not viable — HDBSCAN produces completely different cluster structure on accumulated points
- Lowering cluster size thresholds has no effect — the binding constraint is cluster quality, not quantity
- 96.7% of GT cars have >10 object points, ruling out point sparsity as the recall bottleneck
- Recall bottleneck is cluster quality (merge/split/filter), not cluster formation

## Next

1. Investigate which geometric filter rejects the most valid car clusters (ablation)
2. Investigate HDBSCAN merge/split rate on GT car instances
3. PoinTr implementation for point completion (improves mIoU on matched cars)

---

# Session — 2026-06-23/24

## What was done

### 1. BEV clustering implementation and evaluation (Finding #21)
- Implemented BEV clustering from Lim & Park (IEEE Access 2025) paper on long-range LiDAR vehicle detection for autonomous racing
- Added `_cluster_bev()` in `src/pipeline.py` using scipy.ndimage connected-component labeling with morphological opening
- Added `--clustering-method`, `--bev-resolution`, `--bev-morph-kernel` CLI flags to `src/evaluate.py`
- Tested across resolutions (0.10-0.30m) and morphology kernels (0-3)
- **Result: negative.** Best BEV config (res=0.30, no morph) achieved F1=0.779 vs HDBSCAN F1=0.844. 2D projection merges objects overlapping in x-y; morphological erosion destroys sparse distant clusters.

### 2. Geometric filter ablation (Finding #22)
- Created `src/analyze_clustering.py` with `diagnose_geometric_rejection()` — identifies first failing filter per cluster
- Ran on seq 00 (100 frames) and seq 08 (100 frames)
- **Result:** `min_volume` kills 68% of GT-matching rejected clusters, `min_points` kills 26%. But these are sub-fragments from split cars, not independent detections.

### 3. HDBSCAN merge/split analysis (Finding #23)
- Extended `src/analyze_clustering.py` with per-GT-instance cluster assignment analysis
- **Result:** Splitting is the dominant failure mode (31-37% of GT cars), not merging (0-0.5%). Large/close cars split more (median 724 pts vs 100 pts for clean). Recoverable ceiling 63-68% matches actual recall.

### 4. Recall improvement strategies — both negative (Finding #24)
- Created `src/explore_merge_strategies.py` to gather data on fragment distances, inter-car distances, range distributions, and MCS sweep
- Implemented `_merge_nearby_clusters()` in `src/pipeline.py` — merges small clusters into nearby larger ones within centroid distance and z-gap thresholds
- Implemented `_cluster_adaptive_hdbscan()` in `src/pipeline.py` — runs HDBSCAN with different `min_cluster_size` per distance ring
- Added `--merge-fragments`, `--merge-max-dist`, `--merge-small-threshold`, `--adaptive-hdbscan` CLI flags to `src/evaluate.py`
- Swept: MCS 10-40 globally, merge distances 1.0-2.0m, thresholds 30-50pts, adaptive rings, and combinations
- **Result: both negative.** MCS=20 appeared best on seq 00 (F1=0.852) but hurt seq 08 (F1=0.801 vs baseline 0.829). Fragment merge also hurt (precision drops outweigh recall gains). The ~0.74 recall ceiling is a hard limit.

## Files changed

- `src/pipeline.py` — added BEV clustering, adaptive HDBSCAN, fragment merge (all disabled by default); config params for all three
- `src/evaluate.py` — added CLI flags for BEV, merge, and adaptive HDBSCAN
- `src/analyze_clustering.py` — **new** — geometric filter ablation and merge/split analysis
- `src/explore_merge_strategies.py` — **new** — fragment distance and MCS sweep exploration
- `docs/findings.md` — appended findings #21-24

## Results / findings

Baseline (unchanged): P=0.984, R=0.739, F1=0.844, mIoU=0.943 (seq 00, 100 frames)

| Strategy | Seq 00 F1 | Seq 08 F1 | Verdict |
|---|---|---|---|
| BEV clustering (best) | 0.779 | — | Negative |
| MCS=20 | 0.852 | 0.801 | Overfits seq 00 |
| Merge (1.5m, 30pt) | 0.834 | 0.815 | Negative |
| Adaptive HDBSCAN | 0.831 | — | Negative |

The ~0.74 recall ceiling is confirmed as a hard limit of density-based clustering on voxelized LiDAR without learned object priors.

## Next

1. PoinTr implementation for point completion (improves mIoU on matched cars)
2. Accept recall ceiling; focus on thesis writing and remaining ablations
3. Pipeline diagram for thesis report

---
# Session — 2026-06-26

## What was done

### Completion: KITTI-like single-view PCN training
- Wired the KITTI-like single-view partial generator into `src/train_pcn.py`
  behind `--kitti-like` (+ `--tag`), gated so existing behavior is unchanged.
  `_render_kitti_like`: orient ShapeNet mesh to gravity (up = smallest-extent
  axis), single HDL-64E viewpoint raycast (64 beams -24.9°…+2.0°, 0.09° h-res,
  sensor 1.73 m, range 8-30 m), +0.015 m Gaussian noise, voxel 0.05 m, ground
  cut 0.30 m. New TRAIN_CONFIG keys documented in `docs/pcn/kitti_like_partial.md`.
- Trained PCN on KITTI-like partials: `.venv/Scripts/python.exe src/train_pcn.py
  --kitti-like --tag pcn_kitti --epochs 80`. Best val loss 0.1246 (cd_f ~0.066).
  Converged/plateaued from ~epoch 55-60; last 20 epochs added ~nothing. No
  overfitting (val tracked train). Checkpoint: `checkpoints/pcn_kitti_best.pth`
  (existing `pcn_best.pth` untouched). Per-10-epoch checkpoints + training log saved.

### Quick real-data sanity check (inconclusive)
- `src/test_single_frame_pcn.py --seq 08 --frames 100` on both `pcn_kitti_best.pth`
  and `pcn_best.pth`. Both selected the same densest car cluster (2864 pts,
  frame 43). Both produced a diffuse ~4096-pt blob (~4×6×3 m, oversized vs a real
  ~4×1.8×1.5 m car); KITTI output slightly more elongated, neither a crisp car.
- Test is weak: it targets the *densest* cluster (needs completion least, not the
  sparse regime the KITTI-like data targets), renders a single oblique view, and
  gives no quantitative metric. Not a verdict — yellow flag only.

### Housekeeping
- `CLAUDE.md`: added directive to always run Python via `.venv` (activate or call
  the venv interpreter directly; no bare system python).
- `requirements.txt`: added missing `torch`, `hdbscan`, `tqdm` and `markitdown[all]`;
  pinned `torch==2.6.0+cu124` with `--extra-index-url .../cu124` for the GPU build.
- Diagnosed broken venv launchers: `.venv` was created at `d:\LIDAR_Segmentation`
  and moved to `D:\Code\LIDAR_Segmentation`; Windows venvs aren't relocatable, so
  `pip.exe` and all Scripts launchers have a stale embedded python path
  (`python -m pip` works as a bypass). Permanent fix = recreate venv in place
  (Python 3.10.11) after no training is running. Backup freeze of the 97-package
  env saved to scratchpad (`env_freeze_backup.txt`).

## Files changed
- Modified: `CLAUDE.md`, `requirements.txt`, `src/train_pcn.py`,
  `src/train_classifier.py`, `docs/findings.md`, `docs/project_state.md`
- New: `docs/pcn/kitti_like_partial.md`, `docs/classifier/` (stage_a.md, stage_b.md)
- New checkpoints (untracked): `checkpoints/pcn_kitti_*.pth`,
  `checkpoints/pcn_kitti_training_log.csv`
- (Note: `src/train_classifier.py`, `docs/findings.md` #25, `docs/classifier/`
  were from earlier in this continued session.)

## Results / findings
- PCN-on-KITTI-like trains cleanly (val 0.1246) but the quick real-data look did
  NOT show crisp car completions — same blob failure mode as Findings #15-19, on
  one dense example. Inconclusive; needs the proper eval before any conclusion.

## Next
- Proper completion eval on real seq-08: target SPARSE clusters (40-300 pts),
  render top-down + side views, 4-6 examples, `pcn_kitti_best` vs `pcn_best`
  side by side. Record as Finding #26 once run.
- Recreate `.venv` in place (Python 3.10.11) for the permanent pip-launcher fix.
- Open follow-up (`kitti_like_partial.md`): `main.py` must run completion on a
  single representative frame per track, not accumulated `all_pts` (a smear).

---
# Session — 2026-06-27

## What was done

### Completion inference fix ported + wired (Finding #26 follow-up, earlier in session)
- Ported the corrected normalization into `src/completion.py complete()`: removed
  3D PCA; reorient gravity→Y / length→Z; scale ×1.137; full-car-center estimate
  (up-shift + ego-side width push). Constants `COMPLETION_SCALE_CORRECTION/
  CAR_WIDTH_PRIOR/UP_SHIFT`. Validated bit-for-bit vs the step-2 path.
- `src/main.py`: completion now runs on each track's **densest single frame** in
  the sensor frame (global↔sensor via `track_transforms`), mapped back to global;
  saves `<tid>_partial.ply`; added `--out-tag`; `--pcn-ckpt` default →
  `pcn_kitti_best.pth`.

### L-shape heading A/B + input gating (Finding #27)
- Read `docs/An Efficient L-Shape Fitting Method…md` (Qu et al. 2017). Implemented
  search-based L-shape fit `_lshape_axes()` (closeness criterion, 1° search) and
  `_pca_axes()` in `completion.py`; added `--heading-method {lshape,pca}` to `main.py`.
- Ran 300-frame seq-08 demo three ways: `output/08_ab_pca`, `output/08_ab_lshape`,
  `output/08_ab_gated` (`--seq 08 --frames 300 --out-tag _ab_* --heading-method *`).
- A/B script `scratchpad/ab_heading.py` (BEV uses **X–Z** ground plane: global frame
  is Y-up). Fixed a NumPy 2.0 `.ptp()` removal in `_lshape_axes`.
- Enabled the input gate by default: `COMPLETION_FRAGMENT_MIN_LENGTH=2.7`,
  `COMPLETION_MERGE_MAX_WIDTH=2.3` → `complete()` returns `fragment_input` /
  `merge_suspected`; `main.py` records `completion_skip_reason`.

## Files changed (git)
- Modified: `src/completion.py`, `src/main.py`, `docs/findings.md`,
  `docs/project_state.md`
- New (untracked): `docs/An Efficient L-Shape Fitting Method…md`,
  `scratchpad/ab_heading.py` (+ other scratch diagnostics)
- Experiment artifacts (gitignored `output/`): `output/08_ab_pca`,
  `output/08_ab_lshape`, `output/08_ab_gated`, `output/ab_heading.png`
- Unrelated untracked files present but NOT part of this session:
  `.agents/skills/thesis-reviewer/`, `DepthCamera_*.json`, `DepthCapture_*.png`

## Results / findings (Finding #27)
- Synthetic sanity: L-shape recovers exact heading (125° vs PCA's 141°, 16° off).
- **Heading A/B on real data: NEGATIVE** — PCA vs L-shape a wash (18/47 plausible
  cars either way, mean L/W/H unchanged). Dense-poor tracks were bad *inputs*:
  301 actually fine, 762 a fragment, 884 a merge. Earlier misdiagnosis came from
  plotting X–Y (side view) instead of X–Z.
- **Input gating: the win** — fragments (0/15) and merges (0/7) never complete into
  plausible cars. Gate drops completions 47→26 but retains all 18 good cars →
  completion precision **38% → 69%** (`output/08_ab_gated`: 26/62; skips 14
  fragment + 7 merge + 15 too_few_points).

## Next
- Optional: re-run full `output/08` (`--seq 08 --frames 5000`) with the gate on
  (current full run predates the gate).
- Residual lever: the 8/26 implausible *clean*-input completions (genuine PCN
  error) — PoinTr / better center estimation.
- Methodology: BEV/footprint diagnostics must use the X–Z plane.
- Carry-over: recreate `.venv` in place (Python 3.10.11) for the pip-launcher fix.

---
# Session — 2026-06-30 (seq-08 testing + completion-focus planning)

## What was done

### Full seq-08 evaluation
- Ran `python src/evaluate.py --seq 08 --frames 5000 --iou-threshold 0.3`
  (4071 frames, default two-pass track-filtered eval, stage_b_best.pth).
- First full-sequence eval on seq 08 (prior headline was seq 00, 100f).

### Visualizations (all headless; figures in output/, scripts in scratchpad/)
- `seq08_bev_detections.png` — 6-frame BEV TP/FP/FN overview.
- `seq08_failure_zooms.png` — merge/split close-ups (frames 3900, 2500, 250)
  with per-cluster "instance" panels + GT footprint outlines.
- `seq08_timeseries.png` — per-frame rolling P/R/F1 + GT-car/FN/FP density.
- `seq08_completion.png` — partial vs completed grid, axis-agnostic (sorted) dims.
- `seq08_completion_global.png` — 14 completed cars overlaid on raw frame 2543.
- `seq08_pcn_vs_pointr_shapes.png` — PCN vs PoinTr footprints, 6 shared tracks.
- `seq08_completion_3d_9882.png` — 3-angle 3D of cleanest completion.
- New scripts: viz_seq08_{bev,zoom,timeseries}.py, viz_completion.py,
  viz_completion_{global,3d}.py, viz_pcn_vs_pointr.py, view_track_3d.py
  (interactive Open3D launcher).

### Scratchpad review (discussion only)
- Reviewed all 18 scratchpad scripts. Agreed split: Group 1 = frozen lab records
  (verify_pcn_step1/2, validate_completion_port, diag_*, ab_heading,
  compare_pcn_pointr, viz_all26) backing Findings #26–28; Group 2 = reusable viz
  tools (this session's). Decision: leave everything in scratchpad, no promotion.

### Direction decision + plan
- Shift project focus to point-cloud completion, prioritizing thesis narrative
  strength; retraining acceptable.
- Wrote `docs/completion/plan.md` (4-direction roadmap + detailed Direction 4a plan).

## Files changed
- No tracked file changes from the analysis work (git clean). New untracked
  session artifacts: output/seq08_*.png, scratchpad/viz_*.py, scratchpad/view_track_3d.py.
- Docs written this wrap-up: docs/completion/plan.md (new), docs/project_state.md,
  docs/session_history.md.

## Results / findings
- Seq 08 full (4071f): P=0.913, R=0.693, F1=0.788, meanIoU=0.895
  (TP=23593 FP=2235 FN=10470). Confirms seq-00 story at 40× scale:
  precision-saturated, recall-limited. Per-frame recall anti-correlated with
  GT-car density (dense parked-car stretches drive FN); FP flat ~1/frame.
- Completion viz: dense inputs (>1.5k pts) complete into correct car shapes;
  sparse inputs (~64 pts) largely prior-driven; 90° heading ambiguity visible;
  PCN ≈ PoinTr on real (PoinTr slightly taller) — consistent with #28.

## Next
- Begin Direction 4a Step 0: amodal GT box builder (`scratchpad/amodal_gt.py`).
  See `docs/completion/plan.md` and project_state.md "Immediate Next Steps".

---

# Session — 2026-07-05 (Direction 4a complete: completion improves the box)

## What was done

### Advisor progress report
- Concise 2–3 page markdown report covering May 12 – Jul 5:
  `docs/report/progress_report_2026_07_05.md` (classifier/tracking status,
  recall-ceiling story, completion saga #26–#28, planned direction WP1–4).

### Direction 4a Steps 1–3 (Step 0 was done 2026-07-02)
- **Step 1** `scratchpad/completion_box_eval.py`: per-frame label-propagated
  detections (`evaluate.get_frame_detections`, Stage B classifier,
  thing_classes={10,252}), greedy IoU≥0.3 matching, TP pairs on the 40
  well-observed amodal-GT cars; production completion path
  (`pcn_kitti_best.pth`, L-shape gate); raw and completed boxes fitted in the
  world frame with the same fitter as GT (`fit_oriented_box_xz`, minmax
  extents). Full run: 2,063 candidate frames, ~44 min in background →
  `output/experiments/completion_box_eval/step1_records_08.json`.
- **Step 2** `scratchpad/completion_box_eval_step2.py`: |ΔL|,|ΔW|,|ΔH|,
  BEV oriented-box IoU (Sutherland–Hodgman clipping), yaw error (mod 180°,
  folded to [0,90]), XZ center error. Primary unit = car (per-car medians,
  Wilcoxon signed-rank across cars, avoids pseudo-replication); pooled
  frame-level secondary; splits by both_sides_seen, GT length, input density
  → `step2_metrics_08.json`.
- **Step 3** `scratchpad/completion_box_eval_viz.py`: 6-panel BEV overlay
  figure (GT black / raw blue / completed green)
  → `output/figures/completion_box_overlays_08.png`.
- Docs: **Finding #29** appended to `docs/findings.md`;
  `docs/completion/plan.md` Steps 1–3 marked done + DECISION recorded;
  `docs/project_state.md` Direction 4a closed, next = Direction 1.

### Methodology Q&A (thesis-defense prep)
- Walked through: L-shape fit/gate, fragment (2.7 m fit) vs GT-length (3.6 m
  analysis split) thresholds — different quantities, no inconsistency;
  Wilcoxon signed-rank + p-value interpretation; amodal-GT validity
  (constructed pseudo-GT, defensible via multi-frame-vs-single-frame
  information asymmetry, external car-dim anchor, paired-design noise
  tolerance, both_sides_seen split).
- New idea logged: cross-validate amodal GT boxes against KITTI raw 3D
  tracklets (odometry seq 08 = raw drive 2011_09_30_drive_0028), if annotated.

## Files changed
- Modified: `docs/completion/plan.md`, `docs/findings.md`, `docs/project_state.md`
- New (untracked): `docs/report/progress_report_2026_07_05.md`
- New (gitignored scratchpad/output): `scratchpad/completion_box_eval.py`,
  `scratchpad/completion_box_eval_step2.py`, `scratchpad/completion_box_eval_viz.py`;
  `output/experiments/completion_box_eval/step1_records_08.json` (2,075 records),
  `step2_metrics_08.json`; `output/figures/completion_box_overlays_08.png`

## Results / findings
- 2,075 TP pairs, 40/40 well-observed cars; 1,339 completed; gated: 714
  fragment + 22 merge (35%, matches #23 split rate).
- Per-car medians (n=39, Wilcoxon): BEV IoU 0.707→0.747 (p=.0019); |ΔW|
  0.270→0.170 (p=1.5e-4); |ΔH| 0.255→0.131 (p=1.6e-10); center err
  0.286→0.234 (p=2.8e-5); |ΔL| 0.447→0.456 (p=.65, neutral); yaw 3.5°→3.0°
  (p=.74, neutral — confirms #27).
- Gains largest on sparse inputs: <100 pts IoU 0.461→0.599; ≥300 pts
  0.703→0.744. |ΔW| gain holds in both both_sides_seen groups.
- Signed ΔL, normal cars (≥3.6 m, 32 cars): raw −0.485 m, completed −0.545 m
  → length **under-completion** (far end not extended), not compact overshoot.
- **DECISION: pre-registered criterion met — "completion adds value"
  established (Finding #29). Proceed to Direction 1.**

## Next
- Direction 1: donor-frame occluded-side Chamfer metric (+ symmetry
  self-consistency secondary); reuses Step 0 accumulation/coverage infra.
- Optional hardening: check KITTI raw tracklets for 2011_09_30_drive_0028 and
  validate a subset of the 40 amodal GT boxes against human annotations.
- Direction-2 targets on record: length under-completion, sparse-input heading
  errors, (idea) track-level fragment gate.
- Uncommitted: doc changes + progress report — consider a checkpoint commit.

---

# Session — 2026-07-06 (PoinTr matched eval: Finding #28 synthetic table corrected)

*(Entry reconstructed retroactively on 2026-07-14 from working-tree diffs and
the corrected Finding #28 text; the original session context was not
available at wrap-up.)*

## What was done
- Built `scratchpad/matched_eval_pcn_pointr.py`: runs `pcn_kitti_best.pth` and
  `pointr_kitti_best.pth` through the identical `verify_pcn_step1.py` protocol
  — same 30 synthetic val cars, literally identical normalized 256-point
  partials, CD/F in metres — to replace Finding #28's mixed-protocol synthetic
  comparison.
- Corrected **Finding #28** in `docs/findings.md`: the original "decisive
  PoinTr win" (cd_fine 0.1246 vs 0.0634, F 0.76 vs 0.987) mixed metrics from
  different protocols (PCN's val *loss* vs PoinTr's val *fine-CD*; metre-scale
  vs normalized-frame F-scores). Propagated the correction to
  `docs/project_walkthrough.md` (Storyline 4) and the milestone note in
  `docs/project_state.md`.

## Files changed
- Modified: `docs/findings.md` (#28 rewrite), `docs/project_walkthrough.md`,
  `docs/project_state.md` (milestone note)
- New (gitignored scratchpad): `scratchpad/matched_eval_pcn_pointr.py`

## Results / findings
- Matched eval, training-normalization path: PCN CD 0.161 ± 0.021 m / F@0.1m
  0.755 vs PoinTr **0.153 ± 0.016 m / 0.782** — a real but small (~5%) edge,
  better on 27/30 samples; matched like-for-like log fine-CDs are 0.0658 vs
  0.0634.
- GT-free inference path: both degrade identically (CD ~0.5 m, ~3× the
  in-distribution floor) — the centroid-estimation bottleneck (#26) is
  architecture-independent.
- Cross-checks passed: PCN row reproduces Finding #26's documented numbers;
  normalized-CD column matches both training logs.
- **Real-data equivalence and the keep-PCN decision are unchanged.** Revised
  thesis framing: a controlled architecture comparison where a small synthetic
  edge vanishes on real data because shared inference/domain bottlenecks
  dominate — decoder capacity was never the binding constraint.

## Next
- (Superseded by the 2026-07-14 session below.)

---

# Session — 2026-07-14 (Cross-domain classifier matrix; Stage A dropped from production)

*(Entry written at wrap-up from working-tree diffs and Findings #30/#31; the
original working-session context was not available, so details are taken from
those findings.)*

## What was done

### Cross-domain classifier matrix (Finding #30, advisor-requested)
- Advisor (07/07 chat) requested a sim-to-real cross-validation table:
  train-on-synthetic → test-on-real and vice versa. Three cells existed
  (Stage A log, Finding #25); the missing cells were run with a new script
  `scratchpad/cross_domain_classifier_eval.py` (evaluates any checkpoint on
  either val set). Semantics decision: each checkpoint keeps **its own
  training-time bbox-feature mean/std** (deployment behavior); only eval data
  changes.
- Commands:
  ```bash
  .venv\Scripts\python.exe scratchpad/cross_domain_classifier_eval.py --ckpt checkpoints/classifier_best.pth --domain real
  .venv\Scripts\python.exe scratchpad/cross_domain_classifier_eval.py --ckpt checkpoints/stage_b_scratch_best.pth --domain synthetic
  .venv\Scripts\python.exe scratchpad/cross_domain_classifier_eval.py --ckpt checkpoints/stage_b_best.pth --domain synthetic
  ```
- Table sent to advisor. Artifacts:
  `output/experiments/cross_domain_classifier/*.json` (per-class reports +
  confusion matrices).

### Stage A dropped from production (Finding #31)
- Following #30, the user decided to remove Stage A synthetic pretraining from
  the final pipeline (kept as thesis ablation, #7/#25/#30). Open concern was
  #25's pipeline-precision edge for the pretrained checkpoint (100-frame
  spot-check). Both standard evals were re-run with
  `checkpoints/stage_b_scratch_best.pth`:
  ```bash
  .venv\Scripts\python.exe src/evaluate.py --classifier-ckpt checkpoints/stage_b_scratch_best.pth
  .venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --classifier-ckpt checkpoints/stage_b_scratch_best.pth
  ```
- Default classifier checkpoint switched to `stage_b_scratch_best.pth` in
  `src/evaluate.py`, `src/main.py`, `src/visualize_gt.py`,
  `src/test_single_frame_pcn.py`. `stage_b_best.pth` kept on disk for
  reproducibility.
- Docs updated: Findings #30 and #31 appended to `docs/findings.md`;
  `docs/project_state.md` refreshed (production checkpoint, headline metrics,
  checkpoints list, classifier section).

## Files changed
- Modified: `docs/findings.md` (#30, #31), `docs/project_state.md`,
  `src/evaluate.py`, `src/main.py`, `src/visualize_gt.py`,
  `src/test_single_frame_pcn.py`
- New (gitignored scratchpad/output): `scratchpad/cross_domain_classifier_eval.py`,
  `output/experiments/cross_domain_classifier/*.json`
- Still uncommitted from earlier sessions: 2026-07-06 Finding #28 correction
  (`docs/findings.md`, `docs/project_walkthrough.md`, `docs/project_state.md`),
  `docs/report/progress_report_2026_07_05.md` (untracked)

## Results / findings
- **#30 — sim-to-real gap is total and symmetric.** Cluster-level macro F1
  (car F1): Stage A on real val 0.447 (0.000, recovers 5 of 24,968 cars);
  scratch on synthetic 0.304 (0.000); fine-tuned on synthetic 0.318 (0.000) —
  fine-tuning catastrophically forgets synthetic (0.999 → 0.318). Stage A's
  0.807 accuracy on real val is pure majority-class — report macro F1 +
  confusion matrix, not accuracy.
- **#31 — hypothesis (scratch loses precision) wrong; scratch is
  neutral-to-better.** Seq 00 (100f): F1 0.844 → **0.859** (+37 TP at
  identical FP=20), P 0.984, R 0.761, mIoU 0.942. Seq 08 (4071f): F1 0.788 =
  0.788, mIoU 0.895 = 0.895; P 0.913→0.903, R 0.693→0.699 (+230 TP / +330 FP,
  ~0.08 FP/frame). #25's precision edge was a 100-frame small-sample effect.
  Seq-00 fine-tuned baseline re-run first and reproduced the documented
  headline exactly (deterministic), so the comparison is clean.
- **New headline metrics (production = scratch checkpoint):** seq 00
  P 0.984 / R 0.761 / F1 0.859 / mIoU 0.942; seq 08 P 0.903 / R 0.699 /
  F1 0.788 / mIoU 0.895.

## Next
- Direction 1: donor-frame occluded-side Chamfer metric (+ symmetry
  self-consistency secondary); reuses Step 0 accumulation/coverage infra.
  Plan: `docs/completion/plan.md`.
- Optional hardening: cross-validate amodal GT boxes against KITTI raw 3D
  tracklets (odometry seq 08 = raw drive 2011_09_30_drive_0028), if annotated.
- Consider a checkpoint commit: working tree now carries the 07-06 #28
  correction, today's #30/#31 doc updates, and the checkpoint-default switch
  in four `src/` files.

---

# Session — 2026-07-17 (Direction 1 complete: donor-frame occluded-side metric validated, Finding #32)

## What was done

- **Session start + full results/storyline synthesis:** recapped all 31
  findings and mapped planned vs. possible thesis storylines on request.
- **Direction 1 planned and locked** (`/tweakable-plan`, plan approved):
  visibility-mask novel set (τ=0.15 m), one-directional coverage
  (novel→method, cov@0.1 m + median distance), out-of-amodal-GT-box
  hallucination guard (+0.2 m), pipeline TP cluster inputs, raw + mirrored
  baselines, per-car medians + Wilcoxon (as #29), symmetry self-CD secondary.
- **Refactor (`src/completion.py`):** extracted
  `PointCloudCompleter.estimate_canonical_frame()` (gate + L-shape heading +
  canonical basis + full-car center + radius) from `complete()` so the
  mirrored baseline reuses exactly the production geometry. Verified bitwise
  behavior-preserving against pre-refactor reference outputs.
- **Built the 4-script metric pipeline** (all in `scratchpad/`, gitignored by
  design; outputs under `output/experiments/donor_metric/`):
  - `donor_metric_step1.py` — detection sweep + per-pair npz cache
    (`stage_b_scratch_best.pth`, `pcn_kitti_best.pth`); seq 08 full run
    ~38 min: 2,092 TP pairs / 40 cars / 1,337 gate-passed (733 fragment,
    22 merge).
  - `donor_metric_step2.py` — donor accumulation (cached), novel sets at
    τ∈{0.10,0.15,0.20}, coverage + out-of-box + region breakdown + sym
    self-CD; ~4 min.
  - `donor_metric_step3.py` — per-car medians, Wilcoxon, validation gate,
    size/region/both-sides diagnostics.
  - `donor_metric_viz.py` — 6-panel BEV figure
    `output/figures/donor_metric_08.png` (best→worst completed coverage).
- **Reproduce check:** Step 2 rerun on cached pairs → byte-identical records.
- **Docs:** new `docs/completion/donor_metric.md` (method, results, schema,
  caveats); **Finding #32** appended to `docs/findings.md`; Direction-1
  section added to `docs/completion/plan.md`; `docs/project_state.md` updated
  (Direction 1 complete, next = Direction 2).
- Started a `/quiz-me` comprehension quiz (Q1 asked); user deferred — resume
  next session if wanted.

## Files changed

- Modified: `src/completion.py` (refactor), `docs/findings.md`,
  `docs/completion/plan.md`, `docs/project_state.md`
- New: `docs/completion/donor_metric.md`
- Untracked-by-design (gitignored): `scratchpad/donor_metric_step{1,2,3}.py`,
  `scratchpad/donor_metric_viz.py`, `output/experiments/donor_metric/`,
  `output/figures/donor_metric_08.png`
- Nothing committed this session.

## Results / findings

- **Validation gate: all four items PASS** — (a) raw ranks last at every τ;
  (b) per-car IQR of completed cov 0.14; (c) ranking stable across τ;
  (d) completed out-of-box 0.0003 ≤ mirrored 0.0083.
- **Headline (per-car medians, n=39, τ=0.15):** cov@0.1 raw 0.000 / mirrored
  0.043 / completed **0.304**; med novel-dist 0.518 / 0.332 / **0.161 m**;
  all pairwise Wilcoxon p < 1e-6. First valid real-data evidence that PCN
  reconstructs unseen surface (7× the symmetry-mirror baseline, ~zero
  hallucination).
- **Direction-2 targets quantified:** far_end cov 0.133 (vs far_side 0.321,
  top 0.203) = #29's length under-completion; heading/center errors on
  diagonal/sparse views (worst figure panels).
- Completion is input-size-robust (cov 0.30–0.33 across size bins); mirroring
  degrades on sparse inputs. Sym self-CD car median 0.122 m (untested as a
  signal).

## Next

- **Direction 2: improve `complete()` geometry** — far-end extension +
  heading/center robustness, measured with the donor metric (per-car-median
  cov@0.1 @ τ=0.15 + Wilcoxon, far_end split) alongside #29 box metrics.
  Idea backlog: `docs/completion/next_ideas.md`.
- Consider a checkpoint commit: tree carries the #32 docs +
  `estimate_canonical_frame()` refactor.
- Optional: finish the `/quiz-me` quiz (Q1 pending); test sym self-CD as a
  reference-free signal; amodal-GT cross-validation vs KITTI raw tracklets
  still on the backlog.

---

# Session — 2026-07-18

(Session ran 2026-07-17 → 18; report files carry the 07-17 date.)

## What was done

### Consolidated results report for the advisor (docx)

- Reviewed all findings (#1–#32) + project state and built a full-story,
  thematically organized results report with python-docx:
  `docs/report/results_report_2026_07_17.docx` — detection pipeline arc,
  recall ceiling, sim-to-real study, completion arc; 7 tables, 3 embedded
  figures (`seq08_bev_detections`, `completion_box_overlays_08`,
  `donor_metric_08`), appendix mapping report sections → finding numbers.
  Used corrected #28 numbers (not the overstated "halves CD" claim).
- Generator scripts live in the session scratchpad (not in repo):
  `build_results_docx.py`, `build_overview_docx.py`.

### Rev2 — response to critical review feedback

- User relayed critical review feedback (later revealed to be from another
  LLM, not the advisor). Addressed every point with existing data — no new
  experiments needed: operating-point justification (threshold sweep shows
  recall is capped upstream: τ=0.30 → R 0.695 vs 0.711 at 0.50, so
  F1-optimal = recall-maximal), filter validity across vehicle sizes (#22:
  shape thresholds ≈4% of GT-overlapping rejections; SemanticKITTI car class
  includes SUVs/vans), recall ceiling reframed as structural property of
  density clustering, gate cost stated (completion attempted on 64.5% of TP
  detections; detection recall unaffected), signed ΔL treated as known
  conservative regression (−0.485→−0.545 m, |ΔL| neutral p=0.65), and a
  far-end remediation plan (§6.1).
- Corrected a conflation in the feedback: the three domain-adaptation
  attempts (#16/#17/#19) were completion-side, not classifier-side; synthetic
  data was dropped only for classification, fixed (KITTI-like generator) for
  completion.
- Saved as `docs/report/results_report_2026_07_17_rev2.docx` (rev1 was
  file-locked in Word; kept as-is).

### Pushback after the LLM reveal + tracklet verification

- Removed the false "supervisor review" attribution from the docx and
  `project_state.md`.
- **Verified KITTI raw drive 2011_09_30_drive_0028 (= odometry seq 08) has
  NO tracklet annotations**: 404 on the official S3 bucket
  (`avg-kitti/raw_data/<drive>/<drive>_tracklets.zip`); control
  `2011_09_26_drive_0001` exists (274 KB); `2011_09_30_drive_0027` and
  `2011_10_03_drive_0027` also 404. The LLM's "mandatory tracklet
  cross-validation" is unsatisfiable — resolved as impossible. Amodal-GT
  validity now rests on: paired design, construction guards
  (viewpoint coverage + zero motion), dimension sanity check
  (median L 4.14 / W 1.75 / H 1.47 m).

### Condensed overview version (the one to send)

- `docs/report/results_overview_2026_07_17.docx` — 1,534 body words
  (~5 pages), all 6 key tables + 3 figures kept, prose compressed ~55%;
  purpose: advisor tracking/understanding. Rev2 cross-referenced as the
  extended-justification companion.

## Files changed

- Modified: `docs/project_state.md` (far-end plan recorded; tracklet
  question resolved-impossible)
- New: `docs/report/results_report_2026_07_17.docx`,
  `docs/report/results_report_2026_07_17_rev2.docx`,
  `docs/report/results_overview_2026_07_17.docx`
- Nothing committed this session.

## Results / findings

- No new experiments. One new fact established: seq-08's source drive has no
  official KITTI tracklet annotations (verified via S3 probe), so no official
  3D box GT exists for it — the amodal pseudo-GT (#29) is the only option.

## Next

- Send `results_overview_2026_07_17.docx` to the advisor.
- **Direction 2 (far-end undershoot), committed plan:** Step 1 inference
  geometry (L-shape near-corner anchor + longitudinal length prior +
  symmetry-derived center) → Step 2 visibility-weighted asymmetric Chamfer
  retrain on KITTI-like synthetic → Step 3 contingency real fine-tuning.
  Metrics: far_end cov 0.133, signed ΔL −0.545 m, full #29/#32 suites.
- Consider a checkpoint commit (tree carries #32 docs, refactor, and the
  three report files).
- Carry-over from last session: finish `/quiz-me` (Q1 pending); test sym
  self-CD as a reference-free signal.

---

# Session — 2026-07-19 (advisor Q&A, full-pipeline timing benchmark, report restyle, perf plan)

## What was done

### 1. Advisor concern: test scenarios not explicit in the results doc
- Advisor asked (chat, VN) whether reported metrics come from many frames/many
  cars or a single car. Root cause: the overview doc never stated the test
  scenarios.
- Created `docs/report/results_overview_2026_07_19.docx` (copy; the sent
  `_07_17` original preserved untouched) with a new "Test scenario and
  evaluation protocol" subsection at the top of §2: seq 08 = 4,071 frames /
  393 distinct cars / ≈34k evaluated car appearances (TP+FN=34,063 under the
  ≥10-points-after-preprocessing rule); seq 00 first 100 frames = 41 cars.
  Protocol reconstructed from `src/evaluate.py`: per-frame independent eval,
  greedy 1-to-1 point-IoU matching at 0.3, micro-averaged TP/FP/FN; completion
  experiments use per-car medians (39 static cars) so each car counts once.
- Terminology normalized throughout the docx: "scan" → "frame", with a one-time
  definition (one frame = one full 360° LiDAR scan, ~120k points).
- Ready-to-send Vietnamese reply drafted in chat.

### 2. Advisor follow-up: inference time + which completion metric matters
- Built `scratchpad/timing_benchmark.py` — per-stage wall-clock benchmark
  mirroring the production path (evaluate.py stages + tracker + PCN completion
  timed per car cluster), `--stride` sampling, 3-frame warmup, 250-frame
  checkpoints, distinct JSON names per mode.
- **Sampling-bias lesson:** first-100-contiguous frames gave 675 ms/frame —
  28% understated (sparse opening scene; HDBSCAN cost ~doubles in dense
  segments). Uniform stride-40 (n=99) and stride-20 (n=201) agreed within
  0.3% at ~934 ms. ~100 frames suffice *if drawn across the whole drive*.
- **Full-sequence run (all 4,071 frames, ~70 min background):**
  `output/experiments/timing/timing_seq08_full_n4068.json` — **921.3 ms/frame
  mean (917.6 median), ≈1.1 frames/s** vs the sensor's 10 Hz. Stages: HDBSCAN
  502 ms (54%), RANSAC 163 ms (18%), load+preprocessing 136 ms (15%),
  classifier 74 ms (8%, 42.8 clusters/frame ≈ 1.7 ms each), geometric filter
  46 ms, tracker 0.3 ms. Completion: 18.6 ms per completed car (n=12,322),
  1.5 ms gate-rejected (n=14,823). Stride-20 estimate was within 1.4%.
  Scene variability: ~450-frame block means ≈770–1,040 ms.
- Added to the docx: "Inference time" section (after §1, table + prose) and a
  completion-metrics section — Coverage@0.1 m = primary quality metric (with
  out-of-box hallucination guard), BEV IoU = downstream utility, per-car-median
  aggregation, Chamfer synthetic-only; the two metrics cross-check each other.

### 3. Report restyle: Q&A tone → report tone (user request)
- Question heading "Which completion metric has to be good — IoU or Coverage?"
  → numbered "§5.3 Completion evaluation metrics"; old §5.3/§5.4 renumbered
  §5.4/§5.5 with all cross-references (incl. appendix findings table) updated.
- "Both, because…" answer paragraph, "Sampling note:", "Two observations."
  scaffolding, defensive "never a single vehicle or a single frame", and the
  "rev. July 19 (…)" subtitle all rewritten as declarative report prose.
  Zero question marks remain; doc is now usable as a school submission /
  paper draft base.
- Final timing numbers (921 ms, full sequence) swapped into the docx after the
  benchmark finished.

### 4. Runtime optimization plan (implementation deferred)
- `docs/perf/plan.md` created via /tweakable-plan: hypothesis ≤400 ms/frame
  reachable with metrics unchanged. Section A decisions (A1 regression budget,
  A2 HDBSCAN strategy, A3 ground removal, A4 preprocessing order) **left open —
  user deferred the decision to the next session**. Frame-parallelism rejected
  by user. Full-sequence baseline (921 ms) recorded in the plan.

### 5. Cleanup
- Deleted `docs/project_walkthrough.md` (user-approved): stale derived
  synthesis (last updated 2026-06-29, 27 findings, obsolete headline metrics
  P 0.956/R 0.731/F1 0.829); fully reconstructible from project_state +
  findings. Recoverable from git history.

## Files changed

- New: `docs/report/results_overview_2026_07_19.docx`, `docs/perf/plan.md`,
  `scratchpad/timing_benchmark.py` (git-ignored),
  `output/experiments/timing/timing_seq08{,_stride40,_stride20,_partial,_full_n4068}.json`
  (git-ignored)
- Modified: `docs/project_state.md`
- Deleted: `docs/project_walkthrough.md`

## Results / findings

- **Full-pipeline inference time (seq 08, all 4,071 frames): 921 ms/frame,
  ≈1.1 frames/s; 72% in classical CPU stages (HDBSCAN 54% + RANSAC 18%);
  PCN completion 19 ms per completed car.** Not yet a numbered finding —
  candidate for one when the perf work starts.
- Distinct-car counts: seq 08 = 393, seq 00 (100 frames) = 41.
- Contiguous-start timing samples are biased (−28%); sample uniformly.

## Next

- Send `results_overview_2026_07_19.docx` + the VN chat reply (use final
  figure **921 ms/frame**; the earlier draft said ~934).
- **Next session: lock Section A of `docs/perf/plan.md` (A1–A4), then
  implement Tier 1** (classifier batching, `core_dist_n_jobs`, copy trims —
  exact-output verification against a frozen TP/FP/FN reference).
- Carried: Direction 2 (far-end under-completion), pipeline diagram, thesis
  writing.

---
# Session — 2026-07-23 (runtime optimization landed; full post-opt report refresh + honesty pass)

## What was done

### Pipeline runtime optimization (Findings #33, #34)
Executed the locked perf plan (`.claude/plans/2-fluffy-crown.md`, `docs/perf/plan.md`).
- **Tier 1 (exact-output, per-frame TP/FP/FN bit-for-bit, #33):** batched classifier
  inference, `hdbscan_core_dist_n_jobs=-1`, trimmed the z-filter numpy→Open3D copy.
- **Tier 2/3 (behavior-changing, promoted, #34):** `ransac_iterations` 1000→300,
  `voxel_before_denoise=True`, `cluster_voxel_size=0.10` — now the `PIPELINE_CONFIG` defaults.
- Runtime **934.5 → 650.6 ms/frame (−30%, ≈1.54 fps)**. ≤400 ms target proven unreachable
  via the authorized tiers (sparse object cloud compresses only ~18%; ~185 ms Python-HDBSCAN
  floor). User accepted 650 ms — real-time out of scope; GPU HDBSCAN (cuML) / DBSCAN deferred.
- Detection improved on full seq 08: P 0.903→0.905, R 0.699→**0.730**, F1 0.788→**0.808**,
  mIoU 0.895→0.912 (TP 23823→25478). cv=0.10 closes intra-car density gaps (#23) → +1,655 TP —
  a research result, not just a speedup.
- `output/08` regenerated under the promoted config (old preserved at `output/08_preperf_backup/`);
  #29/#32 re-checked (isolate-config) — neither finding shifts.

### Report refresh: docs/report/results_overview_2026_07_23.docx
Created from the 07-19 overview; full refresh of §1–§6 + Summary to the promoted post-opt
operating point so the whole report sits at one consistent operating point.
- **§1 inference:** 921→623 ms figures + optimization narrative; disclaimer flipped
  ("§2–§5 reflect this optimized operating point").
- **§2 detection:** seq 08 → 0.905/0.730/0.808/0.912, TP/FP/FN 25,478/2,676/9,444;
  seq 00 → fresh 100-frame post-opt eval (0.967/0.777/0.862/0.962); per-car unit n=40 / 2,262 TP.
- **§5 completion (Table 4, post-opt):** BEV IoU 0.718→0.739 (p=.013), |ΔW| 0.252→0.177,
  |ΔH| 0.230→0.132, center 0.273→0.230, **|ΔL| 0.428→0.463 (p=.034, worse)**, yaw 3.3/3.3
  (neutral); Table 5 donor coverage raw 0.000 / mirror 0.051 / PCN 0.302; sparse BEV
  0.459→0.606; far-end coverage 0.125.

### Honesty pass (user: "be honest… softened framing seems hypocritic")
- **§3 Recall Ceiling — fully rewritten.** Dropped the obsolete "~0.74 hard ceiling / five
  mitigations all negative" framing. Now leads with the correction: five *post-hoc* repairs
  failed, but coarsening the clustering resolution (cv=0.10, production) cut the split rate
  37.7→29.9% (zero merges), raised the clean-cluster fraction 62→70%, lifted recall 0.699→0.730
  (+1,655 TP). "The earlier '~0.74 hard ceiling' was overstated." Residual structural limit
  kept but "smaller than first claimed." Summary bullet (1) rewritten to match.
- **Item 2 (§5.4):** prose now states completion *significantly worsens* length
  (|ΔL| 0.428→0.463 m, p=0.034), matching Table 4's "worse" verdict; conservative-bias
  mitigation retained.
- **Item 3a ([22]):** track-filter ablation "F1 0.762→0.801" kept (not re-run) + provenance
  caveat — pre-optimization operating point, isolates the filter's contribution, current endpoint 0.808.
- **Item 3b (#29 fine-tuned `stage_b_best.pth`):** deliberately left undisclosed — the paired
  raw-vs-completed design makes classifier drift immaterial to the delta; user scoped out.

## Files changed
Modified (uncommitted): `src/pipeline.py`, `src/classifier.py`, `src/evaluate.py`, `src/main.py`,
`src/analyze_clustering.py`, `docs/findings.md` (#33/#34), `docs/project_state.md`,
`.claude/skills/latex-report/SKILL.md`
New (untracked): `docs/report/results_overview_2026_07_23.docx`, `docs/approved_version.tex`,
`.claude/skills/latex-report/references/`, `.claude/skills/humanizer/`
(All 2026-07-23 code + report work is uncommitted.)

## Results / findings
- Runtime 934.5→650.6 ms/frame (−30%); detection F1 0.788→0.808, recall 0.699→0.730 (#33/#34).
- Report is now internally consistent at the post-opt operating point, with the recall-ceiling
  story corrected (partly a resolution artifact, not a hard limit).

## Next
- Pending user approval: correct `docs/findings.md` #24 wording (still says "hard limit").
- Commit the uncommitted 2026-07-23 runtime-opt + report work when asked.
- Resume Direction 2 (improve `complete()` geometry: far-end under-completion, coverage 0.125;
  signed ΔL −0.550 m).

---
# Session — 2026-08-01

## What was done

### Direction 2, Step 1 — longitudinal length prior for complete() (Finding #35)
- Root cause of far-end under-completion (#32 far_end cov 0.133; #29 signed ΔL
  −0.49→−0.55): `estimate_canonical_frame()` had width (X) + up (Y) priors but no
  length (Z) correction, so a partial truncated at the occluded far end normalizes
  around its near portion and PCN stops short of the unseen end.
- Fix (inference-only, no retraining): extend-only Z push toward the ego-far end,
  `center[2] += sign(center[2])·max(0.5·L_prior − 0.5·observed_len, 0)`,
  `COMPLETION_CAR_LENGTH_PRIOR = 4.14 m` (amodal-GT median). Mirrors the width
  prior. `src/completion.py`. Shipped ON (constructor `length_prior` default =
  constant; `None` disables for A/B). Planned via /tweakable-plan (A1 length prior,
  A2 ego-sign, A3 extend-only, A4 synthetic-first).

### Evidence — three paired lines, all positive
- Synthetic true-GT (`scratchpad/length_prior_synth_check.py`, n=40): far-quarter
  cov 0.42→0.59, CD/F improve, under-reach halved. Prior must be near true length
  (4.5≈ceiling on 4.5 m synthetic cars; 4.14 under-serves them).
- Real donor #32, A/B (`donor_metric_recompute.py` + step2/3 on
  `donor_metric_len_{off,414,450}`; #32-era detection, n=39, τ=0.15): far_end cov
  0.123→0.324, overall 0.307→0.428, out_of_box 0.0004→0.0014. **4.5 rejected** —
  out_of_box 0.0122 breaks the ≤ mirrored 0.0083 guard. Ship 4.14.
- Real box #29 (`length_prior_box_recheck.py`, n=39): signed ΔL −0.44→−0.32
  (completed now beats raw), |ΔL| 0.44→0.35. Ego-sign self-validated (donor
  far_end region is ego-defined; wrong sign would have lowered it).

### output/08 regenerated under the prior
- `main.py --seq 08 --frames 5000 --no-gui --save-output --out-tag _lenprior`
  (production defaults). 1040 tracks / 518 completed / 1558 PLYs. md5: only the 518
  completed clouds changed. Swapped by rename; old at `output/08_noprior_backup/`;
  GT artifacts copied in.

### #29/#32 production-config table refresh (prior OFF vs ON)
- Recomputed completions from `donor_metric_perf` (promoted config, stage_b_scratch)
  → `donor_perf_len{off,on}` (n=40 / 1508 pairs); donor step2/3 + box records via
  `build_box_records_from_donor.py` → `completion_box_eval_step2`. stage_b_scratch
  for both (resolves the #29 drift, published on stage_b_best); recompute-off
  reproduces the #34 `_perf` baselines.
- #32 donor: overall cov 0.301→0.403, far_end 0.121→0.346, out_of_box
  0.0004→0.0020 (guard holds).
- #29 box: |ΔL| 0.476→0.354 (off-prior completion worsens length, p=0.027; on
  reverses it), BEV IoU 0.743→0.747, width/center flat, height −1.2 cm.
- Compact-overshoot caveat: fixed 4.14 prior over-extends compacts (<3.6 m: signed
  ΔL −0.10→+0.25) while fixing normal cars (−0.55→−0.34).

## Files changed
Committed: `b7e2a4f` (`src/completion.py`, `docs/findings.md` #35,
`docs/project_state.md`, `docs/completion/plan.md`); `9f569d8`
(`docs/findings.md`, `docs/project_state.md` — production-config refresh).
New scratchpad scripts (gitignored, local): `length_prior_synth_check.py`,
`donor_metric_recompute.py`, `length_prior_box_recheck.py`,
`build_box_records_from_donor.py`.
New experiment outputs (gitignored, local):
`output/experiments/donor_metric_len_{off,414,450}/`, `donor_perf_len{off,on}/`,
`completion_box_eval/step{1_records,2_metrics}_08_len{off,on}.json`;
`output/08` regenerated (`output/08_noprior_backup/` preserved).

## Results / findings
- Finding #35: the length prior fixes far-end under-completion — far_end cov
  0.12→0.35, |ΔL| flips from a regression to a gain, guardrails intact, at both the
  #32-era and production operating points. First Direction-2 win; synthetic + two
  real-data metrics all agree.

## Next
- Direction 2 Step 1b (optional): per-car length estimate to remove the
  compact-overshoot (fixed 4.14 prior over-extends <3.6 m cars).
- Direction 2 Step 2: heading/center errors on diagonal/sparse views.
- Backlog ③: HDBSCAN vs DBSCAN vs Euclidean clustering benchmark.
- User plan: do ① (Direction 2) and ③ before ② (thesis writing).

# Session — 2026-08-02 (Step 1b shipped; d2 guard backfill; RNG fix; OLS detour reverted)

Crash-recovery day — resumed mid-task after a terminal crash; nothing lost. A
mixed session: one solid ship plus a guard fix and a reproducibility fix, but
also a self-inflicted frame-convention bug and one yes-man recommendation flip,
both caught and corrected. Logged here honestly.

## Shipped (committed 21430cf, 18:23)
- **Direction 2 Step 1b:** `track_length_estimate()` — tracks with ≥5 gate-passed
  frames get q90(fit_length)+0.12 m; sparser tracks fall back to the 4.14
  constant. `main.py` aggregates per-track `fit_length` and wires it in.
- **Findings #36** (per-car estimate fixes compact over-extension — SHIPPED) and
  **#37** (the #32 hallucination guard was blind to band-localized over-extension;
  pooled median hid a **100×** compact violation). **Step 1c** logged in plan.md
  (residual under-extension mechanism, deliberately deferred, not bolted on).
- Validation was frame-sound: `length_1b_box_eval.py:86` uses the correct
  `pts @ Rᵀ + t` world transform.

## d2 hallucination guard (Finding #37)
- Added band-split gate item **d2** to `donor_metric_step3.py`; **backfilled into
  all 5** existing donor summaries with `.pre_d2_backup.json` preserved.
  Integrity verified (only `d2_*` keys changed). `len450` confirmed #35's L=4.5 m
  rejection at ~165× baseline.

## RNG call-order fix (Finding #38, committed this session)
- Production `complete()` carried `self._rng` state across tracks → a track's
  completed cloud depended on **call order** (measured: OLD "identical across
  order" = False, NEW = True).
- Fix: `complete(sample_seed=...)`; `main.py` passes `pcn_sample_seed` per call,
  matching the donor-eval per-pair reset. **Reproducibility fix, not quality**
  (seed sweep: sd 0.0014 = noise). `evaluate.py` unaffected by construction — it
  never calls completion.

## Process failures — honest log
- **OLS sparse-track fallback (Finding #40):** implemented on top of 21430cf,
  then **REVERTED**. Frame-correct A/B (`clean_fallback.py`) showed it fires on
  only **2/40** cars and is a **wash** (`|ΔL|` 0.214→0.221). Negative result;
  kept the simpler constant.
- **Frame-convention bug (Finding #39):** four probe scripts used `(raw−t)@R`
  instead of `raw@Rᵀ+t`, producing **retracted** "5 fallback cars / halved-MAE"
  numbers. Caught via `frame_check.py`. `data["raw"]` in the donor caches is
  **sensor** frame; `world_box` needs `raw @ Rᵀ + t`. The published box eval was
  correct throughout.
- **Yes-man flip:** reversed the d2-backfill recommendation once with no new
  evidence; caught and corrected. Added global **Pushback** + **Evidence** rules
  to `~/.claude/CLAUDE.md`.

## Files changed
- Committed `21430cf`: `src/completion.py`, `src/main.py`, `docs/findings.md`
  (#36/#37), `docs/project_state.md`, `docs/completion/plan.md`.
- Committed this session (loose-ends): `src/completion.py` (`sample_seed`),
  `src/main.py` (per-call seed), `docs/findings.md` (#38/#39/#40),
  `docs/session_history.md`, `docs/project_state.md`.
- Gitignored/local: `output/experiments/{donor_metric*, donor_len1b_*,
  seed_sweep_q90off, donor_len1b_olsfb}`; scratchpad probes (`clean_fallback.py`,
  `frame_check.py`, `length_estimator_probe2.py`, `length_1b_box_eval.py`).

## Results / findings
- **Finding #36** shipped; **#37** (guard blind spot) + **#38** (RNG) + **#39**
  (frame trap) + **#40** (OLS negative result) logged.
- Day net in git: 21430cf (Step 1b) + one loose-ends commit; the OLS detour nets
  to zero (added then reverted).

## Next
- **Regenerate `output/08`** with `--save-output` if reproducible completed clouds
  are needed for the thesis — it was generated under the old call-order RNG path.
- Direction 2 **Step 1c:** residual under-extension (completion under-fills its
  normalized frame; a better length target doesn't translate to a better box).
- Direction 2 **Step 2:** heading/center errors on diagonal/sparse views.
- Backlog ③: clustering benchmark.

---
# Session — 2026-08-03 (Delegate brief execution: Tier 1 hygiene, T1–T7)

Executed Tier 1 of `docs/plans/delegate_brief_2026_08_02.md` (a full-repo
review follow-up plan the user had prepared, alongside
`docs/plans/personal_agenda_2026_08_02.md`), task by task, each gated on the
seq-00 reproduction baseline matching exactly (P 0.967/R 0.777/F1 0.862/mIoU
0.962, TP=1296/FP=44/FN=371).

## What was done
- **T1** — tracked 37 previously-gitignored `scratchpad/*.py` evidence
  scripts (`.gitignore`: `scratchpad/*` + `!scratchpad/*.py`); outputs remain
  ignored. Commit `9ba02f2`.
- **T2** — fixed `evaluate.py` CLI help drift (`--target` default text,
  `--ransac-iterations` default text) and added `--no-voxel-before-denoise`
  (the existing flag was a no-op with no off-switch since the config default
  is already `True`). Commit `4df200a`.
- **T3** — deduplicated `resolve_track_class` (identical in `main.py` and
  `evaluate.py`) into `tracker.py`. Commit `5303291`.
- **T4** — added `src/test_invariants.py`, 13 pytest cases: sensor↔global
  round-trip identity, `estimate_canonical_frame()` purity + extend-only
  length push, `track_length_estimate()` fallback/quantile branches,
  `complete()` order-independence under `sample_seed` (skippable if the PCN
  checkpoint is absent; ran green here). `pytest` wasn't installed in
  `.venv`; added to `requirements.txt`. Commit `f046443`.
- **T5** — filled the "TP/FP/FN not separately recorded" gap in
  `project_state.md`; added a disambiguation note distinguishing the
  label-scan "≥10 raw pts" car count from the eval recall denominator ("≥10
  points surviving preprocessing"). Flagged (didn't rewrite) two looser
  phrasings in `session_history.md` and `findings.md` per house convention.
  Commit `ef73981`.
- **T6** — added `length_estimate_source` (`"track_q90"`/`"fallback"`) and
  `n_gate_passed_frames` to each completed track's `tracks.json` entry,
  populated exactly where `track_length_estimate()` runs. Verified on a
  30-frame seq-08 smoke run before committing. Commit `bcf012f`.
- **T7** — regenerated `output/08` (`main.py --seq 08 --frames 5000 --no-gui
  --save-output --out-tag _regen`) under the per-car length estimate (#36) +
  the #38 RNG order-independence fix, superseding the 2026-08-01
  fixed-prior/pre-#38 version. Verified via new
  `scratchpad/verify_regen_08_t7.py` (md5): track set, all identity fields,
  all 518 `_partial.ply` inputs, and all non-completed `.ply` outputs
  byte-identical — no regression. Swapped in (old preserved as
  `output/08_fixedprior_backup/`). Fallback-frequency count: 119/518 (23.0%)
  — well above the brief's 5% threshold, so per the pre-registered decision
  rule, logged as a live limitation (**Finding #41**) rather than fixed.
  Commit `a4a65c5`.

## Process note
One operational hiccup: the first attempt to background the T7 run appended
a stray `&` on top of the harness's own `run_in_background`,
double-backgrounding the job — the harness reported "completed" after ~1s
(only the launcher shell had exited), while the actual `main.py` run kept
going untracked. Caught by checking process/CPU state directly (16 python
workers, `main.py` still burning CPU); recovered by starting a proper
polling watcher (`until [ -f tracks.json ]; do sleep 30; done`) that waited
for the real completion signal.

## Files changed
- New: `src/test_invariants.py`, `scratchpad/verify_regen_08_t7.py`, plus 37
  previously-untracked `scratchpad/*.py` files (T1).
- Modified: `.gitignore`, `src/evaluate.py`, `src/main.py`, `src/tracker.py`,
  `requirements.txt`, `docs/project_state.md`, `docs/findings.md`.
- `output/08` swapped (gitignored, not a git change): old preserved at
  `output/08_fixedprior_backup/`.
- Commits: `9ba02f2`, `4df200a`, `5303291`, `f046443`, `ef73981`, `bcf012f`,
  `a4a65c5` (all local; not pushed).

## Results / findings
- Reproduction baseline held exactly across all seven tasks.
- **Finding #41** (new): 23.0% of completed tracks in the regenerated
  `output/08` fall back to the fixed 4.14 m length prior rather than the
  per-car `track_q90` estimate — answers Finding #40's open caveat (b);
  logged as a live limitation, no fix implemented (out of scope per the
  delegate brief's decision rule).

## Next
- **T8** (Tier 2, delegate brief): held-out sequence selection (default seq
  05, never 08) + amodal GT construction (`scratchpad/amodal_gt.py` +
  `amodal_gt_viz.py`, same guards as seq 08), ~30–60 min. Fallback to seq 00
  if <15 well-observed static cars; STOP and escalate if that also fails.
- After T8: **T9a** — pre-registration draft, the one hard user-approval
  gate (`docs/plans/personal_agenda_2026_08_02.md` P0.1). Not reached yet.
- Full remaining roadmap: `docs/plans/delegate_brief_2026_08_02.md` (T8 →
  T9a/b/c → T10/T11 → T13/T14).

---
# Session — 2026-08-09 (Delegate brief Tier 2: T8 held-out GT, T9a pre-registration, T9b frozen evals)

## What was done

Executed `docs/plans/delegate_brief_2026_08_02.md` Tier 2 through T9b. All work
committed; three commits this session (c524612, 64d3e73, f08a0d3).

### T8 — held-out sequence selection + amodal GT
- Ran `scratchpad/amodal_gt.py --seq 05` (default). **Fell to the fallback rule:**
  only 11 well-observed static cars (< 15 threshold); dominant rejection
  `center_spread` 107/170 fitted. Survivors skew short (median L 3.61/W 1.82/H 1.50).
- Fell back to **seq 00** (brief: never 08): `amodal_gt.py --seq 00` → **46
  well-observed** (of 531 fitted / 537 observed). Median dims L 3.80/W 1.81/H 1.49
  (W/H within 2–6 cm of seq-08 ref; L shorter, inside sane range).
- Visual gate: `amodal_gt_viz.py --seq 00` → `output/00/amodal_gt_check.png`;
  boxes tight and correctly oriented on all 12 panels (4 typical + 8 outliers);
  short-L outliers are genuine compact clouds.
- Confound logged: seq 00 is in the classifier Stage-B train split (optimistic
  detection recall); completion claims unaffected (paired raw-vs-completed; PCN
  synthetic-trained). No labeled seq is both classifier- and completion-held-out.
- Pushback recorded: user asked whether to try other sequences before T9a;
  recommended against (all labeled non-08 seqs share the same leakage; shopping
  for a sequence is selection bias the pre-registration exists to prevent).

### T9a — pre-registration (the one hard user gate)
- Drafted in plan mode, **user-approved**, written verbatim to
  `docs/plans/preregistration_heldout.md` (dated 2026-08-09) and committed
  BEFORE any seq-00 eval (c524612, immutable timestamp).
- Primary refutation-bearing metrics: BEV IoU (#29) + donor cov@0.1 (#32),
  completed beats raw, per-car medians, Wilcoxon p<.05. R1 does-not-generalize,
  R2 08-specific length constants (per-band d2), R3 escalate uncovered results.
  Outcome taxonomy HOLDS / PARTIALLY HOLDS / DOES NOT GENERALIZE.
- Discretionary choice made: only BEV IoU + donor cov are refutation-bearing;
  |ΔW|/|ΔH|/|ΔL|/center-err secondary (per brief R1 wording).

### T9b — frozen-config evals
- Verified `track-q90off` recompute mode matches shipped production constants
  (quantile 90, offset 0.12, fallback 4.14, min_frames 5).
- Pipeline (production config + checkpoints, no tuning), seq-00-isolated dirs:
  `donor_metric_step1.py --seq 00 --out-dir output/experiments/donor_metric_00`
  → `donor_metric_recompute.py --length-mode track-q90off --out-dir
  output/experiments/donor_metric_00_lenon` → `donor_metric_step2.py` →
  `donor_metric_step3.py` → `length_1b_box_eval.py --variant on=...` →
  new helper `scratchpad/t9b_box_all_wilcoxon.py` (pooled ALL-cars box Wilcoxon).
- Results written to `docs/plans/t9b_results_heldout_seq00.md` (T9c input).
- Explicitly did NOT write the verdict (T9c = fresh session, executor/judge
  separation).

## Files changed
- New (committed): `docs/plans/preregistration_heldout.md`,
  `docs/plans/t9b_results_heldout_seq00.md`,
  `scratchpad/t9b_box_all_wilcoxon.py`.
- Modified (committed): `docs/project_state.md`, `docs/session_history.md`.
- New (gitignored outputs): `output/00/amodal_gt.json`,
  `output/00/amodal_gt_check.png`, `output/05/amodal_gt.json`,
  `output/experiments/donor_metric_00{,_lenon}/`.
- Untracked, intentionally left: `docs/plans/delegate_brief_2026_08_02.md`,
  `docs/plans/personal_agenda_2026_08_02.md`, `.codegraph/`.

## Results / findings (T9b, numbers only — verdict deferred to T9c)
- n_cars=45 (of 46; one all-rejected), n_pairs=2588 / 1592 gate-passed; bands
  compact 17 / normal 28 / **long 0**.
- Donor (#32), per-car medians, primary τ=0.15: cov@0.1 raw 0.000 / mirrored
  0.050 / **completed 0.413**; med novel-dist 0.417 / 0.280 / **0.123**; all
  completed-vs-raw and completed-vs-mirrored p≈0; win across every band, τ,
  region. Validation gate a/c/d pass.
- Box (#29), pooled ALL (n=45): BEV IoU raw 0.739 → **0.766** (p=1.6e-3),
  |ΔW| 0.203→0.165 (p=4.0e-4), |ΔH| 0.278→0.120 (p=7.8e-9), |ΔL| 0.359→0.227
  (p=2.0e-3); center_err 0.241→0.201 (p=0.082, n.s.); yaw neutral.
- Nuances flagged for T9c: BEV IoU win is normal-band-driven (normal 0.731→0.765
  p=3.2e-5; compact 0.783→0.771 p=0.85 n.s.); d2 compact pass-bit **fails**
  (completed 0.009 > mirrored 0.005) but ratio ~1.8× vs seq-08 ~16×; long band
  empty → long-band predictions untestable (R3-relevant).

## Next
- **T9c** — fresh Opus session (NOT the T9b executor). Reads ONLY
  `docs/plans/preregistration_heldout.md` + `docs/plans/t9b_results_heldout_seq00.md`;
  applies R1/R2/R3 verbatim; writes the finding draft. Any uncovered result
  (esp. empty long band) → STOP and escalate.
- Then T10/T11 (Sonnet, anytime after deps), T14 ch.1–3 (parallel), T13 last
  and conditional on T9c = HOLDS or PARTIALLY HOLDS.

---
# Session — 2026-08-09 (Delegate brief Tier 2/3: T9c verdict, T10 clustering, T11 movers, T13 Step 1c)

Second session of 2026-08-09 (the earlier T8/T9a/T9b entry is above). Executor/
judge separation honored for T9c. All work committed: ac03bed, fb50f15, 8b8be98.

## What was done

### T9c — held-out arbitration verdict (fresh judge session)
- Applied the pre-registered R1/R2/R3 criteria verbatim to the frozen seq-00
  T9b tables. R1 (does-not-generalize) not triggered; R2 (08-specific length
  constants) not triggered verbatim; R3 (uncovered result) triggered by the
  empty long band (0 cars ≥4.6 m).
- Escalated R3 to the user; resolved via the taxonomy's "caveat named" clause →
  **PARTIALLY HOLDS** (coverage gap, not a weak metric). Tier-3 gate satisfied.
- Wrote `docs/plans/t9c_verdict_heldout_seq00.md` + Finding #42.

### T10 — clustering benchmark (backlog #5)
- Added `_cluster_dbscan` (Open3D) + `_cluster_euclidean` (PCL-style) to
  `pipeline.py` + dispatch + `--clustering-method` choices in `evaluate.py`
  (additive; reproduction baseline re-verified EXACT: TP=1296/FP=44/FN=371).
- Benchmark (per-frame, no track filter; deviation: stride/tracker
  incompatible): HDBSCAN wins F1 on seq 00 (0.831 vs 0.767/0.768) and seq 08
  (0.735 vs 0.678/0.683) entirely via recall; 5–7× slower. eps sweep: no fixed
  radius reaches HDBSCAN's recall (best Euclidean eps=0.4, 0.684/0.800).
- Adopt nothing. Finding #43; scripts `scratchpad/t10_*.py`.

### T11 — moving-car plausibility
- On regenerated `output/08` (518 completed car tracks), split by kinematic
  net-displacement. Movers complete as plausibly as statics (57.9% 11/19 vs
  53.7% 225/419); every motion bucket 54–58%. Recipe caveat: axis-aligned `dims`
  understates diagonally-oriented cars. Finding #44; figure
  `output/figures/t11_mover_completions_bev.png`.

### T13 — Step 1c residual under-extension (Tier 3)
- `/tweakable-plan` → pre-registered `docs/completion/t13_step1c_plan.md` (D1
  Z-only radius decouple, D2 length-axis fill factor, D3 seq-08-only long-band;
  ship gate committed before any run).
- Implemented D1/D2 behind `PointCloudCompleter(decouple_radius, fill_z)`
  (default OFF); A/B flags through `donor_metric_recompute.py`.
- D2 calibration (synthetic, n=300): per-axis fill X 1.099 / Y 1.037 / Z 1.074 →
  length-only, fill_z=1.074 (X under the 1.10 widen threshold).
- Gate verbatim: primary MET (seq-08 normal |ΔL| 0.329→0.232 p=7.7e-3, long
  0.583→0.363; donor far_end cov up both seqs) but compact non-regression FAILS
  both seqs (|ΔL| +0.109 seq08 / +0.116 seq00; compacts over-extend).
- **Verdict: DO NOT SHIP** (pre-registered negative result). Confirms the
  compensation is irreducibly length-dependent (option 3, out of scope). Flags
  kept OFF; production unchanged. Finding #45.

## Files changed
- Modified: `src/pipeline.py`, `src/evaluate.py`, `src/completion.py`,
  `scratchpad/donor_metric_recompute.py`, `docs/findings.md` (#42–#45),
  `docs/project_state.md`, `docs/completion/plan.md`
- New: `scratchpad/t10_clustering_benchmark.py`, `scratchpad/t10_eps_sweep.py`,
  `scratchpad/t11_mover_plausibility.py`, `scratchpad/t13_fill_factor.py`,
  `docs/plans/t9c_verdict_heldout_seq00.md`,
  `docs/completion/t13_step1c_plan.md`
- Commits: ac03bed (T9c/T10/T11), fb50f15 (T13 plan), 8b8be98 (T13 result)

## Results / findings
- Findings #42 (T9c PARTIALLY HOLDS), #43 (HDBSCAN best clustering), #44 (movers
  ≈ statics), #45 (Step 1c negative).

## Next
- **T14 (thesis chapters)** — the only remaining delegate-brief task;
  user-supervised, new session. Ch. 1–3 writable now; ch. 4 (completion geometry
  + held-out replication) and ch. 5 (limitations: T11 movers, T13 negative,
  ~0.73 recall) now unblocked.
- Untracked and intentionally excluded: `.codegraph/`, `docs/plans/
  delegate_brief_2026_08_02.md`, `docs/plans/personal_agenda_2026_08_02.md`.

---

# Session — 2026-08-21

## What was done

### B2 evidence task — GT-eligibility count (Finding #48)
- Wrote `scratchpad/gt_eligibility_count.py`: counts GT car instances
  (sem in {10,252}, inst>0) excluded by the >=10-surviving-points recall
  denominator, reusing `get_frame_detections`'s preprocessing/`gt_masks` path
  (classifier not loaded — gt_masks independent of it) vs. a raw label scan.
- Ran seq 08 stride-20 (204/4071 frames):
  `.venv\Scripts\python.exe scratchpad/gt_eligibility_count.py --seq 08 --stride 20`.
  Pooled: raw_all=2332, raw_ge10=2033, eligible=1737. Exclusion 25.5% vs all
  annotated / 14.6% vs >=10-raw; implied recall-vs-all-annotated ~=0.54 (inferred).
  JSON: `output/experiments/gt_eligibility/gt_eligibility_08.json`.
- Committed `ca67a8a` (script + Finding #48 + plan/state ticks).

### Thesis writing started — Protocol milestone (Phase 3)
- Drafted 4.1 Evaluation Protocol -> `docs/writing/thesis/sec_4_1_evaluation_protocol.tex`
  (sequences/roles + leak disclosure, supported-vehicles targets + eligibility rule
  with B2 numbers, point-IoU greedy 1-to-1 matching + micro-averaging + mean-IoU-
  of-matched, headline table, IoU-threshold sensitivity from B1/#47).
- Drafted 7.2 Donor Metric -> `docs/writing/thesis/sec_7_2_donor_metric.tex`
  (occluded-side principle, definition, 4-item validation gate, #37 per-band guard
  lesson framed as design maturity per claim C7).
- Both compile clean under TeX Live 2026. Applied personal_agenda P1 phrasing rules.
  Fixed LaTeX float jank (-> `float` package `[H]` at paragraph boundaries; tables
  referenced by number). NOT yet advisor-reviewed.
- Committed `e1fabd3` (both .tex + THESIS_PLAN reconciliation).

### THESIS_PLAN.md reconciled to current state
- Ticked stale checkboxes (Section 14 actions 2/3/6/7; Phase 2); updated Section 1,
  Section 3 status rollup, Section 8 (Tab 4.2/4.5), and the end status block
  (Protocol milestone REACHED; next = Methods milestone).

## Files changed (this session's commits ca67a8a, e1fabd3)
- New: `scratchpad/gt_eligibility_count.py`,
  `docs/writing/thesis/sec_4_1_evaluation_protocol.tex`,
  `docs/writing/thesis/sec_7_2_donor_metric.tex`
- Modified: `THESIS_PLAN.md`, `docs/findings.md` (#48), `docs/project_state.md`
- (Build artifacts PDF/aux/log gitignored; `.codegraph/` + `docs/plans/*_2026_08_02.md`
  left untracked intentionally.)

## Results / findings
- Finding #48: eligibility rule excludes 25.5% of annotated seq-08 cars (14.6% of
  >=10-raw-point cars); reported recall 0.730 -> ~0.54 against all annotated cars.

## Next
- **Methods milestone:** Ch 3 (pipeline) + Ch 6 (completion method) + B5 pipeline
  diagram (from `output/experiments/timing/timing_seq08_full_n4068.json`).
- 4.1/7.2 pending advisor review (P1 rhythm). B3 (lit table, Ch 2) + B4 (optional)
  still open. Two substance Qs flagged: seq-00 dual-role framing in 4.1; whether
  ~0.54 recall-vs-all belongs in 4.1.

---

# Session — 2026-08-23

## What was done

### Methods milestone — Ch 3 + Ch 6 drafted
- **Ch 3** `docs/writing/thesis/sec_3_detection_pipeline.tex`: full pipeline
  (preprocess -> RANSAC ground -> HDBSCAN -> geometric filter -> PointNet classifier
  -> centroid tracker), frozen-config table, and the **B5 pipeline diagram** as an
  inline TikZ Fig 3.1 (per-stage ms from `timing_seq08_full_n4068.json`: HDBSCAN
  502 ms/54%, RANSAC 163 ms, learned stages only 10%). Production classifier
  written as the from-scratch checkpoint (#31); Stage A framed as Ch-4 ablation.
- **Ch 6** `sec_6_completion_method.tex`: completion-method debugging story
  (#15-19 -> #26 data fix -> #26 inference-bug fix -> #27 L-shape gate -> #35/#36
  length -> #45 closed Step-1c), eval deferred to Ch 7.
- Both compile clean under TeX Live 2026.

### Ch 6 figures (iterative, user-driven)
- Fig 6.1 naive-vs-KITTI-like partial -- `scratchpad/fig61_partial_compare.py`
  (both train_pcn generators on one ShapeNet car). Caption fixed for the
  "which is better" confusion (both are training inputs; KITTI-like = closer to
  deployment).
- Fig 6.2 evolved 3D-scatter -> **BEV box overlay**. The retired PCA+partial-radius
  "blob" path no longer exists in `completion.py`, so it was reconstructed in a
  scratchpad (freeze-safe). Final: `scratchpad/fig62_box_overlay.py` reuses the
  `completion_box_eval` machinery (`world_box`/`box_corners_xz`/`bev_iou`) to show
  GT amodal box vs old vs corrected completion box, quantified by BEV IoU:
  0.32->0.73 (sparse) / 0.25->0.74 (mid) / 0.47->0.85 (dense). Superseded and
  deleted the 3D-scatter script `fig62_deblob.py` (inlined `complete_old_blob` first).
- Fig 6.3 shipped completion output -- regenerated `output/figures/seq08_completion.png`
  (6-track) and `output/experiments/completion_shape_08.png` (4-track, embedded)
  under shipped config via `scratchpad/viz_completion.py` (reads shipped `output/08`).
- Other-sequences check (user ask): mined seq 00 (31 static cars) + seq 08 (15) at
  2500 frames; corroborates, diffuseness is model-level not sequence-level. Mining
  pickle-cached in `output/experiments/fig62_deblob/`.

### Completion-quality scope corrected
- First over-claimed "clean car", then over-corrected to "diffuse blob"; checking
  `output/figures/seq08_completion.png` settled it -- real completions are
  car-shaped (footprint + roofline) in the canonical frame, crispness varying with
  density, filled 4096-pt clouds not CAD-clean surfaces. Sec 6.5 + captions revised
  to claim pose+scale recovery + car-shaped output; crisp surface = synthetic, real
  quantitative gains = Ch 7.

### Ch 6 layout review
- Figure placement was leaving half-page gaps (`[H]` forbids floating) -> changed
  to `[htbp]`; trimmed the two long captions (8->7 pages); added the missing
  in-text reference to Fig 6.1. All three figures now referenced and building clean.

### Bookkeeping
- Fixed a commit-status inconsistency in `THESIS_PLAN.md` (4.1/7.2 were committed
  `e1fabd3` but listed uncommitted). Updated THESIS_PLAN + project_state (Methods
  milestone, Fig 6.1/6.2/6.3, scope corrections).

## Files changed
- New (untracked): `docs/writing/thesis/sec_3_detection_pipeline.tex`,
  `docs/writing/thesis/sec_6_completion_method.tex`,
  `scratchpad/fig61_partial_compare.py`, `scratchpad/fig62_box_overlay.py`
- Modified: `THESIS_PLAN.md`, `docs/project_state.md`
- Deleted (uncommitted, was untracked): `scratchpad/fig62_deblob.py` (superseded)
- output/ renders (fig61_partials/, fig62_deblob/, completion_shape_08.png,
  figures/seq08_completion.png) are gitignored, regenerable from the scripts.
- **No commits made this session** (commit was interrupted before greenlight).

## Results / findings
- No new experimental findings. Fig 6.2 BEV IoU numbers (0.32->0.73 / 0.25->0.74 /
  0.47->0.85) are illustrative panels from the existing #29/#36 box machinery, not a
  new result. Completion-quality characterization settled to "car-shaped, crispness
  varies with density" (see scope correction above).

## Next
- Commit this session's work (Ch 3/6 .tex + 2 scripts + doc edits).
- Ch 4/5/7 results chapters (Results milestone); optionally review Ch 3 layout.

---
# Session — 2026-08-24

## What was done

### Chapter 4 results half (§4.2–§4.5) drafted
- New `docs/writing/thesis/sec_4_results.tex` (§4.2 Detection Results, §4.3
  Ablations, §4.4 Detection Range, §4.5 Runtime); builds clean standalone under
  TeX Live 2026. Distinct label `ch:detection-eval-results` (folds under §4.1 at
  final assembly). Every number traced to a finding/artifact.
- §4.2 interprets the headline + two existing seq-08 figures (precision-saturated,
  recall-limited). §4.3: Tab 4.2 stage ablation, Tab 4.3 Stage-A redundancy +
  cross-domain, Tab 4.4 clustering benchmark. §4.4 distance table. §4.5 runtime.

### Two freeze-safe read-only runs (same category as B1/B2/B6)
- **Track-filter ablation** (Finding #49): `evaluate.py --seq 08 --frames 5000
  --no-track-filter` → `output/experiments/track_filter_ablation/seq08_notrackfilter.log`.
  Aggregate P 0.820 / R 0.677 / F1 0.741 / mIoU 0.921 (TP 23629/FP 5204/FN 11293)
  = Tab 4.2 middle row. Revealed recall is **non-monotonic**: classifier drops
  recall (0.775→0.677) as precision mechanism, track filter recovers it (→0.730)
  via temporal voting. §4.3 reframed as dip-then-recover.
- **B4-lite recall-by-distance** (Finding #50): new `scratchpad/distance_recall.py`
  (imports `evaluate.py` read-only, replicates greedy IoU≥0.3 matching). Seq 08
  stride-20, per-frame/track-filter-off → `output/experiments/distance_recall/
  distance_recall_08{,_fine}.json`. Pooled TP/FP/FN 1165/268/572 **byte-identical
  to T10 HDBSCAN seq-08** (#43) — validates the replication.

### §4.4 finer bins (user call) resolved a pre-registration miss
- Coarse 0–20/20–40/40+ bins showed monotone recall (no #23 dip) — contradicted
  the plan's pre-registered interpretation. Re-ran with 0–10/10–20 split: recall
  0–10 m (0.787) < 10–20 m peak (0.862) → the #23 near-range over-segmentation dip
  appears at #23's own granularity, plus far-range sparsity decline. 0–10 m also
  lowest matched-mIoU (0.902).

### §4.5 runtime anchor (user "report both")
- Found a full-run promoted-config timing artifact
  (`timing_seq08_full_n4068_combined.json`, 623.5 ms) initially overlooked. Table
  anchored on it (most robust); caption reconciles the stride-20 estimate
  (650.6 ms, cited elsewhere) and the 921 ms pre-opt baseline. Classifier-stage
  timing differs across runs (33.7 vs 73.7 ms) — measurement-condition, not config.

### Independent review (fresh-session judge) + fixes
- `/review-handoff` prepared `scratchpad/ch4_review_handoff.md`; a fresh session
  verified ~30 numbers (zero transcription errors) and returned 9 points; all
  processed against sources and conceded:
  - §4.2 "recall anti-correlated with density" was figure-only + mis-cited to #34
    → re-grounded in measured #23/#24 split rates (66% at 0–10 m).
  - 66%-split citation corrected #23→#24 (binned rates are in #24 line 558) in
    §4.4 AND Finding #50.
  - Cross-domain Tab 4.3 diagonal filled (in-domain car-F1 0.999 / 0.885, #30).
  - §4.1 arithmetic fixed: F1 drop 0.25→0.50 is 2.6 pts (not 2.3; 2.3 is 0.30→0.50
    per #47) — edits committed sibling.
  - Clustering ms column labelled "mean"; runtime "classical CPU" → ~94% (incl.
    geometric filter) + JSON cited; FP rate → ~0.7/frame.
  - Fixed a pre-existing 45pt overfull in §4.1's headline table ("Mean IoU
    (matched)" → "Mean IoU" in both §4.1 tables; captions already qualify).
- Both files rebuild clean (sec_4_results 7 pp, sec_4_1 3 pp).

## Files changed
- New (untracked): `docs/writing/thesis/sec_4_results.tex`,
  `scratchpad/distance_recall.py`
- Modified: `docs/writing/thesis/sec_4_1_evaluation_protocol.tex`,
  `docs/findings.md` (Findings #49/#50 added; #23→#24 correction),
  `docs/project_state.md`, `THESIS_PLAN.md`
- New output artifacts (gitignored, regenerable):
  `output/experiments/track_filter_ablation/`, `output/experiments/distance_recall/`
- Not committed (working tree left for review, as Ch 3/6 were).

## Results / findings
- Finding #49 (track-filter ablation): three-row stage decomposition; non-monotonic
  recall (classifier drops, track filter recovers).
- Finding #50 (B4-lite): near-range over-segmentation dip + far-range sparsity;
  pooled numbers byte-identical to T10.

## Next
- Draft Ch 5 (recall root-cause / HDBSCAN splitting) and Ch 7 (completion results).
- Optionally commit the Chapter-4 work (currently uncommitted).
- §4.1 is a committed sibling now edited in the working tree (arithmetic + table
  width) — include in the next commit.

---

# Session — 2026-08-25 (cross-file audit + Chapter 5 draft & external review)

## What was done

### Cross-file thesis audit — applied all six fixes
An external cross-file audit (5 thesis drafts vs. frozen evidence, ~30 numbers
verified) flagged one must-fix plus should/optional items. Verified the central
claim against the timing JSONs and §4.5 before editing, then applied all six.

- **#1 (must-fix) — Ch 3 timings regenerated from the promoted-config run.**
  `sec_3_detection_pipeline.tex` drew Fig 3.1 and all prose timings from
  `timing_seq08_full_n4068.json` (the pre-#34 run) while its Table 3.1 documents
  the promoted config (`ransac_iterations=300`, `cluster_voxel_size=0.10`) — a
  self-contradiction, and 1.1 fps vs §4.5's 1.6 fps. Rewrote every Ch 3 timing
  from `timing_seq08_full_n4068_combined.json` and updated the Fig 3.1 citation:
  total 921 ms/1.1 fps → 623.5 ms/1.6 fps; ground 163→56 ms (18%→9%); HDBSCAN
  502→370 ms (54%→59%); classifier 74→34 ms (1.7→0.75 ms/cluster); geometric
  46→36 ms; preprocessing 127→126 ms (splits 55/55/17 → 58/50/18); clusters/frame
  43→45; completion +19→+13 ms/car. Now matches Table 3.1 and §4.5. Qualitative
  claims survive (clustering still dominates, 370/623 ≈ 59%).
- **#2 — "learned stages" accounting corrected in both chapters.** Ch 3's "two
  learned stages = 93 ms (10%)" folded per-car completion into the per-frame
  budget → now "the only learned per-frame stage, the classifier, ~34 ms (~5%)",
  completion explicitly excluded as per-car. §4.5 (`sec_4_results.tex`) no longer
  calls the tracker a "learned component" → "the single learned component, the
  classifier, ~5.4% (the centroid/Hungarian tracker is classical, not learned)".
- **#3 — §7.2 n=39 vs n=40 reconciled.** `sec_7_2_donor_metric.tex`: tab:donor-gate
  caption marked as the metric's original acceptance run (Finding #32, n=39,
  pre-length-prior); guard-lesson paragraph marked as the promoted-config re-run
  (Finding #37, n=40 = 8+27+5 across compact/normal/long).
- **#4 — §7.2 inline Finding cites added** (#32 at validation, #37 at guard-lesson)
  to match house style in the other four files.
- **#5 — Ch 6 Fig 6.2 box-IoU triples sourced.** `sec_6_completion_method.tex`
  caption now cites the `fig62_box_overlay.py` render (0.32→0.73 / 0.25→0.74 /
  0.47→0.85 are per-panel render values, not an aggregate finding).
- **#6 — Ch 3 preprocessing-order framing softened.** "The order of the last two
  steps matters for cost" → "is behaviour-changing"; reframed as adopted under
  Finding #34 primarily for the detection gain, timing effect noted as small.

Items the audit checked and found correct (headline consistency, eligibility, IoU
sensitivity, ablations, distance table, cross-domain matrix, completion narrative)
were left untouched.

### (Later same-day session) Drafted Chapter 5 "The Recall Bottleneck"
New file `docs/writing/thesis/sec_5_recall_bottleneck.tex` — the mechanism +
negative-results chapter. Planned via the plan-mode workflow: 2 Explore agents
(thesis conventions + recall findings), then AskUserQuestion decisions — (a) Ch-4
overlap = cross-reference only (#43/#50 not re-tabulated); (b) temporal aggregation
included in the strategy table with a footnote (later moved to prose during review).
Plan file: `C:\Users\ngovi\.claude\plans\chapter-5-velvet-bubble.md`.
- Verified temporal-aggregation traceability before citing: primary record is
  `docs/session_history.md:898-906` (F1 0.844→0.073, R 0.039, `temporal_window=3`,
  code reverted). Not reproducible — feature reverted + research freeze.
- Structure §5.1 clustering- not classifier-bound (#22/#47) → §5.2 root cause
  HDBSCAN splitting (#23, Fig 5.1 `seq08_failure_zooms.png`) → §5.3 failed repairs
  (Tab 5.1: BEV #21, MCS/merge/adaptive #24) → §5.4 resolution fix (#34, recall
  0.699→0.730) + walk-back of the retracted "~0.74 hard limit" (#24 Correction) →
  §5.5 summary. Compiled clean (latexmk, TeX Live 2026 at `D:\texlive\2026`), 6 pp.

### (Later same-day session) Processed external review (/review-handoff, 12 points)
Verified each point against source (`docs/findings.md`, `src/tracker.py`,
`src/evaluate.py`, `src/analyze_clustering.py`) before conceding or pushing back.
- CONCEDED + fixed: Tab 5.1a rebuilt as a true partition (ok/split/noisy+all-noise
  sum to the denominator; `merged` demoted to a cross-cutting memo — old rows summed
  to 1639/810 vs headers 1631/811, confirmed via `analyze_clustering.py:165-182`);
  §5.2 reframed (recall EXCEEDS the ok fraction → floor not ceiling; split cars lost
  ~81%; dropped "usually still detected"/"recoverable ceiling"; added the shrinking
  clean-vs-recall gap 7.1→3.1 pt as the resolution-fix predictor); §5.1 "fragments
  not whole cars" softened to an inference (ablation checks GT-overlap, not survivor
  co-occurrence); §5.3 "post-hoc" over-generalisation fixed (only fragment-merge is
  post-hoc; real axis = alter-formation vs reassemble) and section retitled "Repair
  strategies fail to generalise"; temporal aggregation moved from table to prose;
  811/810 & 37.1/37.7 reconciled in a footnote; range bins shown (66/40/19/4).
- PUSHED BACK with evidence: "1600 TP" is correct (27051−25478=1573, same full-seq08
  sample, #47/#49) — added cite, disambiguated from #34's coincidental +1655;
  "monotonically" defensible (#24 has all four bins); the citation-drift flag was a
  handoff-note error (.tex already cited #24 correctly).
- Confirmed tracker is filter-only (`tracker.py` emits current-frame clusters, no
  interpolation; `evaluate.py:558-567`) → added a guard clause.
- Final: rebuilds clean, 7 pp, no undefined refs. `project_state.md` updated with a
  Ch-5 draft + review-revision block.

## Files changed
- Modified: `docs/writing/thesis/sec_3_detection_pipeline.tex`,
  `docs/writing/thesis/sec_4_results.tex`,
  `docs/writing/thesis/sec_6_completion_method.tex`,
  `docs/writing/thesis/sec_7_2_donor_metric.tex`, `docs/project_state.md`,
  `docs/session_history.md`
- New (later same-day session): `docs/writing/thesis/sec_5_recall_bottleneck.tex`
- Untracked, NOT part of this session (excluded from commit): `docs/LVTN.pptx`,
  `docs/LVTN_extracted.md` (a peer's defense deck, kept as a structure template).

## Results / findings
- Cross-file audit: no new experiments; documentation-consistency pass, no finding
  added to `docs/findings.md` (the frozen evidence was already correct — the defect
  was Ch 3 citing the wrong config's timing file). Text-only edits, grep sweep
  confirmed no stale pre-optimisation numbers remain in Ch 3.
- Chapter 5: no new experiments (research frozen); all numbers transcribed from
  `docs/findings.md` / `docs/session_history.md` and spot-checked against source.
  Chapter compiles clean under TeX Live 2026 (7 pp after review revisions).

## Next
- Draft Ch 7 (completion results) — the last remaining results chapter.
- All chapter drafts (incl. Ch 5) still NOT advisor-reviewed.
- Optional Ch 4 remainder: B3 literature table, B4 distance-recall figure.
- If desired, recompile the audit-edited `.tex` files (Ch 3/4/6/§7.2) under TeX Live
  2026 to reconfirm clean builds before advisor review.

---
# Session — 2026-08-26 (Ch 7 discovered drafted, verified, committed)

## What was done
- Session-start recap initially reported Ch 7 as unwritten, following a stale
  `docs/project_state.md` (whose "Next" line still pointed at Ch 4/5/7). User
  challenged this ("haven't we written it?").
- Inspected `docs/writing/thesis/` and found Chapter 7 **already drafted** across
  two files, both uncommitted (`sec_7_results.tex` untracked, `sec_7_2_donor_metric.tex`
  modified) and unrecorded in project_state:
  - `sec_7_2_donor_metric.tex` (§7.1–§7.2) — §7.1 "Why Chamfer distance is invalid
    on real cars" (#26) had been added; §7.2 leakage-free donor metric.
  - `sec_7_results.tex` (§7.3–§7.7) — coverage (#32), amodal-box utility
    (#29/#35/#36, #45 cross-ref), movers (#44), pre-registered held-out seq-00
    replication (#42), summary + limitations.
- Recompiled both under TeX Live 2026 via `latexmk`: **both clean** —
  `sec_7_2_donor_metric` 4 pp, `sec_7_results` 8 pp; zero undefined refs, zero
  overfull boxes >20 pt, zero LaTeX warnings.
- Reviewed the `sec_7_2_donor_metric.tex` diff (48 ins / 13 del): the §7.1 Chamfer
  subsection plus an §7.2 opener rework and a "fixed length prior" wording fix —
  legitimate content, no accidental changes.
- Updated `docs/project_state.md`: added a "Chapter 7 — DRAFTED; COMMITTED
  2026-08-26" block, flagged the stale 2026-08-23 "Next" line as superseded, fixed
  the "Remaining Ch 4 → then Ch 7" pointer, bumped the date to 2026-08-26.
- Committed Ch 7 + project_state as `6af89c9` (main).

## Files changed
- Modified: `docs/writing/thesis/sec_7_2_donor_metric.tex` (§7.1 added),
  `docs/project_state.md`
- New: `docs/writing/thesis/sec_7_results.tex` (§7.3–§7.7)
- This session: append to `docs/session_history.md`
- Untracked, NOT part of this session (excluded from commit): `docs/LVTN.pptx`,
  `docs/LVTN_extracted.md` (peer's defense deck, structure template).
- Build artifacts (`.aux/.log/.pdf/.fdb_latexmk/.fls`) gitignored, not committed.

## Results / findings
- No new experiments (research frozen). Ch 7 numbers transcribed from
  `docs/findings.md` #29/#32/#35/#36/#37/#41/#42/#44/#45,
  `docs/completion/donor_metric.md`, and `docs/plans/t9{b,c}_*heldout_seq00.md`.
- Both Ch 7 files verified to build clean under TeX Live 2026 (4 pp / 8 pp).

## Next
- Chapters 3–7 now all drafted; **none advisor-reviewed yet** — advisor review is
  the main open item.
- Optional Ch 4 remainder: B3 literature table, B4 distance-recall figure; B5
  pipeline diagram already embedded in Ch 3.
- Thesis assembly: merge the two Ch 7 files per the merge note in
  `sec_7_results.tex` (drop the duplicate `\section`, fold §7.3–§7.7 after §7.2).

---

# Session — 2026-08-26 (Ch 2 Tab 2.1 primary-source verification + reference archive)

## What was done
- Processed an external review of `docs/writing/thesis/sec_2_background.tex` (Ch 2,
  Background & Related Work). Diffed the review against the current `.tex`: found the
  major content-gap fixes (recall→paradigm reframe, 3D-detection omission paragraph,
  §4.1 forward pointer, tracking/AB3DMOT sentence, §4.5 runtime pointer) were ALREADY
  applied earlier in the Ch 2 drafting lineage — no rework needed.
- **Restarted the PARKED Tab 2.1 number verification from primary sources.** Sandbox
  `curl` cannot reach arXiv.org / CVF (network-filtered; google + semanticscholar OK),
  but `ar5iv.labs.arxiv.org` renders paper tables cleanly through WebFetch. Verified
  every published cell against each paper's own results table:
  - Semantic (test set): RangeNet++ 52.2 / car 91.4; RandLA-Net 53.9 / car 94.2
    (own paper; Cylinder3D's table lists it as 50.3 / 94.0 — `.tex` correctly uses the
    own-paper number); Cylinder3D 67.8 / car 97.1.
  - Panoptic (test set): RangeNet++&PointPillars PQ 37.1 / PQ^Th 20.2; Panoptic-PolarNet
    PQ 54.1 / PQ^Th 53.3 (RQ 65.0); DS-Net PQ 55.9 / mIoU 61.6; EfficientLPS PQ 57.4 /
    PQ† 63.2.
  - **Zero transcription errors.** The parked note's suspected Cylinder3D 97.1→96.4 fix
    was REJECTED — Cylinder3D's own Table I prints 67.8 and 97.1 in the same row (96.4
    was a secondary-source artifact). Test-vs-val closed: online single-scan eval track
    = test benchmark, so the semantic rows are test numbers.
- **Applied the four external-review minors** to `sec_2_background.tex` +
  `references.bib` (chapter rebuilds clean, 7 pp): Tab 2.1 caption footnote for the two
  RangeNet++ rows; Cylinder3D puffery→factual; §2.1 22-sequences clarification (11–21
  withheld); EfficientLPS bib year 2021→2022 (+volume 38, number 3).
- **Built the Ch 2 reference archive in `docs/papers/`** — all 25 cited papers, each
  pdfminer-verified as a readable copy of the correct paper. Downloads mostly via
  WebFetch (saves binary to tool-results temp → copied in); WebFetch hard-caps at 10 MB
  and cannot fetch arXiv PDFs >10 MB, and RangeNet++ is not on arXiv — those 5 stragglers
  (RangeNet++, EfficientLPS, SeedFormer, KITTI, DBSCAN, campello2013, JOSS-hdbscan) were
  filled by the user directly. Final: 25/25 present and readable.

## Files changed
- Modified: `docs/writing/thesis/sec_2_background.tex` (4 minor review fixes),
  `docs/references.bib` (EfficientLPS venue-of-record + prior-lineage Ch 2 entries),
  `docs/project_state.md` (PARKED→RESOLVED block; minors-applied + archive-complete),
  `THESIS_PLAN.md`.
- New (committed this session): `docs/writing/thesis/sec_2_background.tex`,
  `docs/writing/thesis/sec_8_discussion.tex` (Ch 8 draft, previously orphaned/uncommitted).
- Deleted: two superseded paper notes `docs/papers/AdaPoinTr …md` and
  `docs/papers/PoinTr Diverse …md` (old naming; replaced by `PoinTr.md`).
- Reference PDFs in `docs/papers/*.pdf` are gitignored (local archive, not committed).
- Excluded from commit: `docs/LVTN.pptx`, `docs/LVTN_extracted.md` (peer's defense deck).

## Results / findings
- No new experiments (research frozen). All Tab 2.1 numbers now primary-source verified
  (measured against paper tables, not WebSearch summaries) — supersedes the earlier
  overstated "web-verified" claim. Reference archive 25/25 complete.

## Next
- **Ch 1 (Introduction)** — next in plan order (THESIS_PLAN.md: Ch 8 → Ch 2 → Ch 1 →
  Ch 9 → abstract). Needs problem/approach/contributions, car-only+offline scope, and
  the proposal-drift reconciliation (dropped bus/motorcycle support; CD+EMD→CD-only).
- Then Ch 9 (conclusion) → abstract.
- None of the drafted chapters (2–8) are advisor-reviewed yet.

---
# Session — 2026-08-28 (Master/template assembly + reference pass)

## What was done
Implemented `~/.claude/plans/reference-pass-master-template-snug-lampson.md` (Steps 1–3
+ verification). Wired the 12 standalone chapter fragments in `docs/writing/thesis/`
into one `report`-class master that builds the whole thesis PDF, and converted every
hand-typed cross-reference to a real `\ref`.

- **Step 1 — master `docs/writing/thesis/main.tex` (new).**
  `\documentclass[a4paper,12pt]{report}`; union preamble of every fragment + the FPT
  template (babel/inputenc/fontenc/lmodern, amsmath/amssymb/amsfonts, bm, booktabs,
  float, siunitx, microtype, graphicx, multirow, array, url, `nth[super]`, geometry
  margin=2.5cm, fancyhdr; tikz + `\usetikzlibrary{...}` for the Ch 3 diagram;
  `\DeclareSIUnit{\frame}`/`{\fps}`; one master
  `\graphicspath{{../../../output/figures/}{../../../output/experiments/}}`). FPT
  title-page/`fancyhdr` style reused (logo referenced explicitly at
  `../report-template/Images/fpt.png`, no file duplication). Locked title, solo author
  Ngo Vi Viet Anh (MSE13205), advisor Dr. Doan Nhat Quang. Roman front matter
  (abstract → `\tableofcontents`/`\listoffigures`/`\listoftables`) → arabic body; single
  `\bibliographystyle{IEEEtran}` + `\bibliography{../../references}` at the end.
- **Step 2 — 12 fragments transformed in place.** Stripped each
  `\documentclass…\begin{document}` and trailing `\end{document}`; promoted
  `\section`→`\chapter`, `\subsection`→`\section`, `\subsubsection`→`\subsection`
  (subsubsections only in sec_4_1, sec_4_results, sec_7_2); dropped each per-file
  `\graphicspath`; stripped per-file `\bibliography` from sec_1/sec_2; dropped the
  duplicate `\section{…}\label{ch:*-results}` at the two folds (sec_4_results,
  sec_7_results). Comment headers left in place (harmless).
- **Step 3 — reference pass.** Converted all typed numerals to `\ref` against the
  now-shared label namespace: `Chapter~N` → `\ref{ch:*}`, `Section~N.N` → `\ref{sec:*}`,
  `Table~4.1` → `\ref{tab:headline}`, `Figure~3.1` → `\ref{fig:pipeline}`, and the two
  plural forms (`Chapters~6 and~7`, `Chapters~4 and~7`).

## Files changed
- New: `docs/writing/thesis/main.tex` (the master; the deliverable to track). Build
  artifacts also produced: `main.pdf`, `main.aux/.log/.toc/.lof/.lot/.bbl/.blg/.out`
  (generated; `main.lof`/`main.lot` currently show as untracked).
- Modified (in-place transform): all 12 `docs/writing/thesis/sec_*.tex` (sec_0, sec_1,
  sec_2, sec_3, sec_4_1, sec_4_results, sec_5, sec_6, sec_7_2, sec_7_results, sec_8,
  sec_9).
- Modified: `docs/project_state.md` (assembly DONE block; header date bumped to
  2026-08-28).
- Untracked, unrelated: `docs/LVTN.pptx` (peer deck, intentionally not tracked).

## Results / findings
- Build verified (`latexmk -pdf main.tex`, TeX Live 2026): exit 0; `main.pdf` 76 pp;
  all figures embedded (no file-not-found). **Zero undefined references, zero undefined
  citations, zero multiply-defined labels**, no rerun requested; BibTeX `.blg` clean.
- Folds render as one chapter each: Ch 4 = §4.1 (protocol) … §4.5 (runtime); Ch 7 =
  §7.2 (donor) … §7.6 (held-out). Converted refs print right numbers (from `main.aux`:
  tab:headline=4.1, fig:pipeline=3.1, sec:results-heldout=7.6, sec:donor-metric=7.2,
  sec:eval-protocol=4.1, sec:results-runtime=4.5).
- Overfull scan: 2 hboxes total, both ≤20pt (incl. the pre-existing 1.09pt EfficientLPS
  one) — no new >20pt overfulls from the report-class/geometry change.
- No experiments (research frozen). The 12 fragments no longer compile standalone
  (accepted trade-off; Step-0 checkpoint `8c0c637` is the git safety net).

## Next
- **Advisor review** of the assembled thesis (no chapter is advisor-reviewed yet).
- Optional cleanup: trim the now-stale MERGE/NUMBERING comment headers in the fragments
  (a few show `\ref` inside comments from the global ref pass — harmless).
- Commit the assembly (decide whether to gitignore `main.*` build artifacts vs. track
  only `main.tex`).
- Then Phase 8 defense prep.

---

# Session — 2026-08-29

## What was done

### LOSO cross-validation (answers reviewer "validation-as-test" critique)
Implemented **leave-one-sequence-out (LOSO)** detection cross-validation over all 11
labelled sequences (00–10) to remove classifier model-selection leakage (seq 08 was
both the classifier's val split and the headline sequence) and to supply the
per-sequence recall breadth the draft lacked.
- **Infra (freeze-safe; shipped `stage_b_scratch_best.pth` + `PIPELINE_CONFIG`
  untouched):** `train_classifier.py --held-out-seq` (re-partitions the mined Stage B
  clusters by sequence-prefix filename, no re-mining; deterministic 5% monitoring slice,
  `LOSO_MONITOR_FRAC=0.05`, seed 1234); `evaluate.py --json-out` + `--frame-fraction`,
  and a missing-checkpoint guard (now `parser.error`s instead of silently scoring
  geometric-only); `scripts/run_loso.ps1` (resumable 11-fold driver);
  `scripts/aggregate_loso.py` (per-fold table + pooled micro + pooled-excl-seq00 +
  macro mean±std → `results/loso/summary.json`).
- **Run:** all 11 folds, full-sequence eval, last-epoch checkpoint (no per-fold
  selection). Background sweep total ~10.5 hr. Per-fold classifiers written to
  `checkpoints/stage_b_fold*` (separate artifacts; gitignored).
- **Result:** pooled **P 0.874 / R 0.719 / F1 0.789 / mIoU 0.919**; per-sequence recall
  spread **0.336 (seq 01, sparse highway) → 0.761 (seq 00, dense urban)**, macro
  0.633±0.121 (precision steadier, 0.840±0.070). Leakage-free fold-08 R 0.737 vs shipped
  0.730 (recall NOT inflated); precision-only exposure ~2 pts (0.881 vs 0.905). Residual
  seq-00 classical-tuning bounded (drop-fold-00 pooled R 0.719→0.697). Logged as
  **Finding #51**.
- **Reconciliation** (added on user request): the two seq-08 rows are ONE model under
  two checkpoint-selection rules — same architecture, same 10 training sequences, same
  from-scratch 15-epoch regime; differ only in which epoch is frozen (shipped =
  best-on-seq08-val epoch 14; fold-08 = last epoch). `399316 = 420333 × 0.95` (the 5%
  monitoring slice seq 08 can't provide) verified. The deltas isolate the selection
  effect; the 5% confound removes training data from the leakage-free model, so it only
  strengthens "recall not inflated."

### Thesis edits (Option B — LOSO as validity + breadth layer, NOT a headline swap)
Seq-08 (0.730/0.905) stays the reported operating point the ablation / recall-root-cause /
completion chapters decompose (LOSO did not re-run those per fold); downstream numbers in
Ch 2/4-results/5/9 unchanged, zero ripple. Edits:
- `sec_4_1`: new §"Leave-one-sequence-out cross-validation" (per-fold + pooled + macro
  table), reframed §4.1.1 protocol, operating-point caption, eligibility reference
  scoped, + the one-model-two-rules reconciliation.
- `sec_8`: "validation-as-test" paragraph → "cross-validated, precision-only exposure";
  breadth caveat resolved; "what generalises" updated.
- `sec_0` / `sec_1`: one-line LOSO mentions.

### Three external-LLM review rounds processed (NOT advisor; user asked to push back)
- **Round 1 (addressed 2):** #1 23% length-fallback — added the defense that the
  0.725→0.771 BEV-IoU gain is measured on shipped output *including* the 119/518 fallback
  tracks, so the cost is already inside the headline (§8); #3 objectness ceiling — added
  the deliberate-trade justification (annotation-free/interpretable is the thesis premise;
  the ceiling is the measured price) (§8). Pushed back #4 (amodal GT already defended
  three ways) + #5 (empty long-band = correct "partially holds", no edit).
- **Round 2 (addressed 3):** #1 mover axis-aligned bias falls harder on movers → 57.9%
  is a stronger lower bound (§8, hedged as expected-direction not measured); #2 seq-01
  highway ODD limit — new paragraph (dense-scene ODD; cause mostly-intrinsic sparsity +
  partly urban-tuned params; conservative-miss, P 0.951) (§8); #3 Python-HDBSCAN ~185 ms
  = implementation floor not algorithmic — named C++/GPU HDBSCAN port, but **pushed back
  on cuML DBSCAN** (Finding #43: no fixed-radius clusterer reaches HDBSCAN recall) (§9.3).
- **Round 3 (pushed back all 3, NO edits):** #1 volume-gate/fragmentation — premise
  architecturally wrong (completion is downstream amodal surface-fill, not fragment
  reassembly; recall scored before completion) + already bounded by the geometric-only
  ablation (Finding #47, all gates removed → R 0.775, P collapses to 0.149); #2 hardcoded
  HDL-64E — KITTI *is* HDL-64E (in-scope non-issue) + already disclosed (§8 synthetic
  gap / §8.6 "re-check for a new sensor" / §9.3); #3 0.25 m ground-cut shift — minor
  empirical constant, residual already absorbed in shipped BEV IoU 0.771, corner-case
  out of scope.
- Reviewer responses drafted for the user to send onward each round; PDFs delivered
  (rounds 1–2 changed the build; round 3 no change).

### Build
`main.tex` rebuilt after each edit under TeX Live 2026: exit 0, **zero undefined
references** each time; grew 76→79→80 pp. Only residual warning is a pre-existing
~5.5 pt float overflow (invariant across LOSO-table font/caption edits → confirmed not
ours).

## Files changed
- **This session — modified:** `src/evaluate.py`, `src/train_classifier.py`,
  `docs/findings.md` (+#51), `docs/project_state.md`,
  `docs/writing/thesis/sec_0_abstract.tex`, `sec_1_introduction.tex`,
  `sec_4_1_evaluation_protocol.tex`, `sec_8_discussion.tex`, `sec_9_conclusion.tex`.
- **This session — new:** `scripts/run_loso.ps1`, `scripts/aggregate_loso.py`,
  `results/loso/` (11 `fold_NN.json` + logs, `summary.json`, `sweep.log`, plus reverted
  windowed-eval provenance `fold_NN_window.*`).
- **Also committed now (prior 2026-08-28 assembly session, was uncommitted):**
  `docs/writing/thesis/main.tex` (new master) + the in-place transforms of `sec_2`,
  `sec_3`, `sec_4_results`, `sec_5`, `sec_6`, `sec_7_2`, `sec_7_results`.
- **Deliberately NOT committed:** `docs/writing/thesis/main.lof` / `main.lot` (generated
  build artifacts, added to `.gitignore`); `docs/LVTN.pptx` (peer deck).

## Results / findings
LOSO per-sequence + pooled (full sequences, τ=0.3, shipped config):

| Held-out | P | R | F1 | mIoU |
|---|---|---|---|---|
| 00 | 0.913 | 0.761 | 0.830 | 0.932 |
| 01 | 0.951 | 0.336 | 0.496 | 0.953 |
| 02 | 0.793 | 0.700 | 0.743 | 0.903 |
| 03 | 0.733 | 0.623 | 0.673 | 0.904 |
| 04 | 0.797 | 0.501 | 0.615 | 0.835 |
| 05 | 0.861 | 0.731 | 0.791 | 0.910 |
| 06 | 0.861 | 0.637 | 0.732 | 0.906 |
| 07 | 0.905 | 0.729 | 0.808 | 0.929 |
| 08 | 0.881 | 0.737 | 0.803 | 0.912 |
| 09 | 0.824 | 0.646 | 0.724 | 0.911 |
| 10 | 0.725 | 0.559 | 0.632 | 0.871 |
| **Pooled (11)** | **0.874** | **0.719** | **0.789** | **0.919** |
| Macro | 0.840±0.070 | 0.633±0.121 | 0.713±0.097 | 0.906±0.030 |

## Next
- **Advisor review** of the assembled thesis (still no chapter advisor-reviewed).
- Optional, only if committee insists: a held-out **long-band completion eval on a
  second sequence** (would touch the frozen completion pipeline) — declined for now;
  "partially holds" is the honest outcome.
- Phase 8 defense prep.

---

# Session — 2026-08-30 (Front-matter flow pass + defense-deck planning)

## What was done

### Front-matter rhetorical-flow pass (committed `4638c37`)
Reworked three front-matter chapters for rhetorical flow, driven by forwarded external-LLM
reviews (hard-pushback mode). Prose-only; nothing frozen; `main.tex` rebuilds clean (exit 0,
zero undefined refs/citations, zero overfull ≥20 pt; 80→79 pp).
- **Abstract** (`sec_0_abstract.tex`): one ~245-word block → three paragraphs (approach /
  detection findings / completion contribution). Adopted the user's narrative rewrite but
  restored three dropped load-bearing elements: (1) the recall qualifier ("0.730 at a
  point-IoU of 0.3, on cars with enough returns to be scorable"; Finding #48 — bare 0.730
  reads as vs-all-annotated-cars ~0.54); (2) amodal-box scope ("on static, gate-passed cars";
  dropped puffery "significantly"); (3) the third contribution, corrected from an invented
  "mapping where hybrid pipelines succeed/fail" back to synthetic-pretraining redundancy
  (#25/#30), matching §1.4/§2.5/§9.3. LOSO compressed to qualitative; 0.000→0.304 and the
  LOSO numbers left to the body.
- **Intro §1.5** kept in place (pushed back on the review's move-to-appendix/rename — breaks
  §1.4/§9 cross-refs, weakens the proposal-drift defense): trimmed the multi-class + CD+EMD
  paragraphs one sentence each, compressed the "not a departure" note to one line.
- **Conclusion §9.2** compacted (pushed back on merging §9.1+§9.2 into a narrative — loses
  the RQ mapping examiners check for; the review's example sentence asserted a false
  recall→completion causal link). Kept §9.1's RQ answers, all finding citations, the locked
  three-contribution structure.

### Defense-deck planning — Phase 8 start (committed `d06c248`)
- Extracted the peer template `docs/LVTN.pptx` structure via python-pptx (63 slides, 16:9,
  four-act: Title/TOC → I. Introduction → II. Methodology → III. Experiment Evaluation →
  IV. Conclusion; roman-numeral dividers, running footer, numbered subsections,
  baselines-vs-ours walkthrough + result-analysis slides).
- **`docs/defense/plan.md`** (tweakable-plan format): goal + success criteria + Section-A
  decisions, all locked — **A1** = python-pptx onto a cleaned COPY of `LVTN.pptx` (inherits
  theme/footer; fallback fresh 16:9 theme, then manual); **A2** = 25–30 min → ~40 slides +
  backups; **A3** = completion-forward (donor metric is the headline per the thesis title +
  §8.1); A4 own-the-limitations; A5 reuse thesis figures + export the TikZ pipeline diagram.
- **`docs/defense/storyboard.md`**: ~40 content slides + 7 backups, slide-by-slide (message /
  source §-or-Finding / visual). Every P1 hedge carried onto its slide (recall denominator +
  ~0.54 beside the 0.730 headline, PARTIALLY HOLDS, movers plausibility-only, offline-as-scope);
  numbers sourced to §/Finding; claims ⊆ C1–C11. Dedicated LOSO slide (#51); donor-metric core
  = 4 slides (measurement problem / Chamfer-invalid #26 / donor metric #32 / guard #37).
- **Two external-LLM deck reviews processed with pushback:** (a) kept python-pptx over a
  fully-manual flip (the "manual saves time" argument assumes the user builds everything;
  flagged peer-content-leakage + python-pptx slide-clone corruption risks the review missed);
  adopted its LOSO-dedicated-slide + specify-the-three-limitations tweaks. (b) moved the ~0.54
  caveat from the Setup slide to beside the 0.730 headline (slide 21, not deferred to 23);
  merged the geometric-only ablation into the recall-decomposition slide; trimmed the
  completion-debug slide and demoted the L-shape gate to backup B7 (pushed back on splitting
  it into two slides — over-weights plumbing vs the metric).
- Updated the `defense-deck-baseline` memory to point at `docs/defense/plan.md`.

## Files changed
- **New:** `docs/defense/plan.md`, `docs/defense/storyboard.md`.
- **Modified:** `docs/writing/thesis/sec_0_abstract.tex`, `sec_1_introduction.tex`,
  `sec_9_conclusion.tex`, `docs/project_state.md`.
- Two commits this session: `4638c37` (flow pass), `d06c248` (defense docs). The
  `defense-deck-baseline` memory was updated outside the repo tree; `docs/LVTN.pptx` remains
  untracked.

## Results / findings
No new experiments. Manuscript builds clean at **79 pp** (was 80) after the flow pass.

## Next
- **Advisor review** of the assembled manuscript (Phase 7) — user-triggered; still no chapter
  advisor-reviewed.
- **Build the defense `.pptx`** (Section B of `docs/defense/plan.md`): generate onto a purged
  copy of `LVTN.pptx` per `storyboard.md`, export the TikZ Fig 3.1 to PNG, write speaker notes,
  verify; then Phase 8 rehearsal.

# Session — 2026-09-03 (External-LLM tone review: verify, push back, apply the defensible subset)

## What was done

A forwarded external-LLM review (prose/tone only — it reviewed no code) was checked in
hard-pushback mode (per the `external-review-pushback` memory). Every quote was verified
against source before acting. Prose-only; nothing frozen touched.

### Review verdict (mostly rejected / severity-inflated)
- **Fabricated quote found.** The review flagged a grammar error in *"it is never sees a
  labelled real car"* — that string does not exist. Actual text (`sec_1_introduction.tex:106`)
  is *"it never sees a labelled real car"*, grammatically correct. The reviewer invented the
  defect, lowering confidence in its unverified claims generally.
- **Rejected as harmful:** (1) delete/rename §1.5 to "Scope and Limitations" — collides with
  Ch 8 "Discussion and Limitations" and guts the deliberate proposal-drift defense (documented
  load-bearing job in the file header + §1.4/§9 cross-refs); (2) "frozen configuration" →
  "static/fixed" — "frozen" is the freeze-table term of art and "static" collides with "static
  cars" throughout Ch 7; (3) "deployed classifier" — nothing is deployed (offline research);
  (4) "blobs" → "amorphous geometries" — a defined scare-quoted phenomenon, the swap is
  pompous not rigorous.
- **Kept over the review's objection:** §7.1 *"This is a measured failure rather than a
  conjecture"* — it is exactly the measured/inferred distinction good empirical writing should
  make; the review's "empirically demonstrated rather than merely theoretical" was wordier for
  nothing.
- The **CRITICAL** rating on Chapter-6 heading wording was rejected as severity inflation (no
  heading is a critical thesis defect), but the underlying stylistic point was accepted.

### Edits applied (the defensible subset)
- **§7.1** (`sec_7_2_donor_metric.tex`): "The trouble is that the accumulation…" → "The
  fundamental limitation is that the accumulation…".
- **§6.6** (`sec_6_completion_method.tex`): "both feed the completer junk." → "both feed the
  completion network implausible input." (the one genuine dev-slang phrase).
- **Chapter 6 headings de-ticketed** (6.4–6.9), on request after flagging it as an authorial
  judgment call (the chapter is deliberately framed as a debugging narrative): Fix 1 → **Synthesising
  KITTI-Like Partials**; Fix 2 → **Correcting the Inference-Normalisation Error**; Fix 3 →
  **L-Shape Input Gating**; Fix 4 → **Per-Car Length Estimation**; "A closed negative…" → **The
  Second Under-Extension Mechanism**; "The shipped completion path" → **The Final Completion
  Path**. Departed from the review's proposed titles: British spelling (matches the thesis's
  `normalisation`/`voxelisation`), singular "Error" (one bug), dropped the review's "Limitations:"
  prefix (Ch 8 collision) and "Pipeline" (Ch 3 collision).
- **Full `shipped` → `final` sweep**, on request, for consistency: 35 occurrences across seven
  files (`sec_3`, `sec_4_1`, `sec_4_results`, `sec_6`, `sec_7_2`, `sec_7_results`, `sec_8`),
  including `\emph{shipped}` → `\emph{final}` and LaTeX author-note comments. The two
  `do not ship` / `do-not-ship` verdicts (Finding #45) were preserved (release-decision
  phrasing, no "shipped" substring). Uniform "final" preserves the term-of-art distinction
  (production/promoted artifact vs. rejected alternatives).

### Build + verification
`latexmk -pdf main.tex` (TeX Live 2026): exit 0, **79 pp**, zero undefined refs/citations, no
rerun pending. PDF-text extraction confirmed the edits landed in the output: `shipped` = 0
occurrences, `do not ship` = 2 (preserved), all six new Chapter-6 headings present in both the
table of contents and the chapter body.

## Files changed
- **Modified (prose only):** `docs/writing/thesis/sec_3_detection_pipeline.tex`,
  `sec_4_1_evaluation_protocol.tex`, `sec_4_results.tex`, `sec_6_completion_method.tex`,
  `sec_7_2_donor_metric.tex`, `sec_7_results.tex`, `sec_8_discussion.tex`, plus
  `docs/project_state.md` and this file.
- No `src/` / frozen artifact / checkpoint touched. NOT advisor-reviewed.

## Results / findings
No new experiments. Manuscript still builds clean at **79 pp**.

## Next
- **Advisor review** of the assembled manuscript (Phase 7) — user-triggered; still no chapter
  advisor-reviewed.
- **Build the defense `.pptx`** (Section B of `docs/defense/plan.md`).
- ~~Formalize the two `do not ship` / `do-not-ship` verdicts~~ **DONE (follow-up commit):**
  §6.8 (`sec_6`) "do not ship" → "not to adopt it"; §7.4 (`sec_7_results`) "do-not-ship" →
  "to reject it". Rebuilt clean (79 pp, exit 0); PDF now has zero release-jargon
  (`shipped` = 0, `do not ship` = 0).

---
# Session — 2026-09-04

## What was done

### Donor-metric contribution: pressure-test + prior-art hardening
Extended discussion stress-testing whether the donor-frame coverage metric is a
genuine contribution, prompted by doubt that the input-cleanliness finding (#27)
and the two secondary claims are "real" contributions. Conclusions:
- **#2 (recall → HDBSCAN splitting, #23/#34) and #3 (synthetic-pretraining
  redundancy, #25/#30) are findings, not contributions** — diagnostics of the
  system's own behavior, not methods others reuse. The donor metric is the one
  contribution-tier item.
- **Rejected a proposed pivot** to reframe the contribution as "a general
  detection pipeline that finds any object needing reconstruction, then completes
  it." Not viable: classifier is binary car/not-car, PCN is car-trained,
  cross-domain car F1 = 0.000 (#30); it also reopens the withdrawn multi-class
  decision (#20) and is not novel (standard detect-then-complete).

### Prior-art investigation (web, links recorded)
Verified the donor metric's novelty against the literature (was training-memory
before). Findings:
- **Closest prior work: Ren et al. 2022, arXiv 2203.10569** ("Self-supervised
  Point Cloud Completion on Real Traffic Scenes"). They *also* accumulate each
  tracked vehicle across frames as pseudo-GT — but score the **whole** accumulated
  cloud (Chamfer variants L-G/L-I/L-S), with **no** novel-set restriction, **no**
  symmetry baseline, **no** box hallucination guard. Confirmed via ar5iv full-text.
- SemanticKITTI SSC (Behley 2019): same accumulate-other-frames idea at scene-voxel
  level, not per-instance surface.
- KITTI reference-free toolkit — Fidelity / MMD / Consistency (from PCN, yuan2019) —
  references input / synthetic prior / own outputs, never held-out real surface.
- RealPC (2411.17580) does NOT threaten — manually builds paired GT, standard CD/HD.
- **Defensible novelty = the novel-set restriction** (score only donor surface the
  input never saw, so the raw partial scores 0 by construction) + mirrored baseline
  + per-band hallucination guard. Reframe from "a new reference-free metric" to "a
  novel-set-restricted refinement that fixes the under-completion reward inherent in
  accumulate-and-score approaches (Ren et al.; SSC)." Finding #26 is the evidence
  that whole-cloud scoring is inadequate.

### Thesis edits (prose only; freeze-safe; NOT advisor-reviewed)
- `references.bib`: +`ren2022selfsupervised`.
- `sec_2_background.tex` §2.4: new paragraph — reference-free KITTI toolkit
  (Fidelity/MMD/Consistency → PCN) + Ren et al. as closest prior work (whole-cloud
  scoring), foreshadowing the novel-set distinction.
- `sec_7_2_donor_metric.tex` §7.2: new "Relation to existing reference-free
  evaluation" paragraph distinguishing the donor metric from Ren et al. (whole-cloud
  vs novel-set), SSC (scene-voxel vs per-instance), and the KITTI toolkit.

### Open issue identified, not yet fixed
The metric is now **novelty-defended** but only **half-reusable** in the text:
§7.2 Definition still welds eligibility to the pipeline ("A pair counts only if the
input passes the L-shape gate") and there is no explicit general-form statement. This
is the gap between contribution-tier and internal-tool; de-welding it is the proposed
next move.

## Files changed
- **Modified (this session, prose only):** `docs/references.bib`,
  `docs/writing/thesis/sec_2_background.tex`,
  `docs/writing/thesis/sec_7_2_donor_metric.tex`.
- **Also present uncommitted in the tree (concurrent reframe/ablation work, not this
  conversation's prose edits):** `src/main.py` (+38 lines — additive `--no-gate`
  ablation flag + raw-partial save; `main.py` is not a frozen artifact), and untracked
  `scripts/run_gate_ablation.ps1`, `scratchpad/gate_ablation_analyze.py`,
  `docs/writing/reframe_plan.md`.
- Memory (outside repo): `~/.claude/.../memory/donor-metric-novelty.md` added.
- No frozen artifact touched (`src/pipeline.py`, `src/completion.py`,
  `src/classifier.py`, checkpoints, `output/08`).

## Results / findings
No new experiments run this session. Assembled `main.tex` rebuilds clean
(TeX Live 2026): exit 0, **81 pp** (was 79), zero undefined refs/citations, zero
multiply-defined labels, zero overfull >20 pt, `ren2022selfsupervised` resolved.

## Next
- **De-weld §7.2** for reusability: add a general-form metric definition and replace
  the gate-based qualification with a pipeline-agnostic one (qualify on novel-set
  point count alone). Closes the internal-tool gap. (Optionally sharpen §2.5's
  contribution wording to "isolates recovery of previously-unobserved surface.")
- **Gate-off ablation across all 11 sequences** (reframe_plan A1 = Opt B): the
  prerequisite for the input-quality headline; tooling staged
  (`scripts/run_gate_ablation.ps1` → `output/experiments/gate_ablation_v2/<seq>/`),
  ~4–5 h resumable background job. Status unverified this session.
- Advisor review (title change A2 = Opt 2 needs cover sign-off); then the reframe
  Section-B chapter-restructure build sequence.
- Commit this batch (thesis positioning edits) when ready — currently uncommitted.

---
# Session — 2026-09-05 (Debadge, 13->6 file consolidation, Ch1 rewrite, abstract/caption trims)

## What was done

### Dropped "leakage-free" / "reference-free" as contribution badges (committed ef8884c)
Removed both phrases where used as selling points (abstract, intro, conclusion,
discussion, contribution lists, the completion-metric section title). Where the
property is genuine hygiene (LOSO protocol, completion-eval independence) it is now
stated by mechanism instead of the badge word.

### Consolidated thesis 13 -> 6 files (committed ef8884c, pushed)
Merged the sec_* fragments into `ch0_abstract`, `ch1_introduction`, `ch2_background`,
`ch3_methodology` (detection + completion), `ch4_evaluation` (protocol + detection
results + recall + donor metric + completion results), `ch5_discussion_conclusion`.
Rewrote `main.tex` \input list + header; swept stale per-fragment header comments.

### Deleted THESIS_PLAN.md (committed ef8884c)
Stale after the 9->5 restructure. `docs/project_state.md` is now the plan of record;
updated `project_state.md` + `defense/plan.md` pointers and the `thesis-plan-authority`
memory. Claims map C1-C11 survives in git history + embedded chapter-comment labels.

### Introduction rewrite (this session)
- Expanded 1.1 into a motivation funnel: autonomous-driving significance + why LiDAR
  (KITTI/SemanticKITTI), a survey of existing deep-LiDAR methods (Qi/PointNet++/
  RandLA-Net; SqueezeSeg/RangeNet++/Cylinder3D; PointPillars/CenterPoint;
  Panoptic-PolarNet/DS-Net), then two honest limits (annotation cost, opacity);
  `lim2025longrange` added at the modular-approach transition. All citations already in
  `references.bib` -- none fabricated.
- Consolidated Ch1 from 6 sections to 3: 1.1 Problem and motivation (Approach folded
  in, contribution paragraph moved to close), 1.2 Contributions and scope (RQs +
  Contributions + Scope merged), 1.3 Thesis outline. Redirected the one internal \ref.

### List of Figures / Tables shortened
Added `\caption[short]{full}` to all 33 captions (1 ch2, 8 ch3, 24 ch4); full captions
preserved under the floats. LoF/LoT went from multi-line-per-entry (several pages) to
one line each.

### Abstract trimmed
~330 -> ~190 prose words, 4 -> 3 paragraphs. Cut the CAD/mesh aside, restatements, the
component-list aside, the input-quality ablation sentence, and the standalone
contributions list (folded to one clause). All honesty hedges kept (0.730 @ point-IoU
0.3 on scorable cars; cross-validated 11 seq; static/gate-passed; partially replicates;
0.905; "a valid metric", no "first").

## Files changed
- Committed ef8884c (pushed): 13->6 `ch*.tex`, `main.tex`, `THESIS_PLAN.md` (deleted),
  `docs/project_state.md`, `docs/defense/plan.md`, 2 figures.
- Modified this session, then committed in the bookkeeping commit: `ch0_abstract.tex`,
  `ch1_introduction.tex`, `ch2_background.tex`, `ch3_methodology.tex`,
  `ch4_evaluation.tex`, `docs/project_state.md`, `docs/session_history.md`.

## Results / findings
Build clean throughout (latexmk exit 0, 0 undefined refs/citations, 85 pp).

## Next
- Optional: front-matter acknowledgments/declaration.
- Ch1 prose + abstract NOT advisor-reviewed.
