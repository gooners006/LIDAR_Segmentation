# Project State

Last updated: 2026-06-27

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working (recall ceiling confirmed) |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Binary Stage B trained |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | Fixed inference (#26); single-frame completion in `main.py`; L-shape input gate (#27) → completion precision 38%→69% |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics + sweep flags), `src/visualize_gt.py` (GT vs pipeline toggle viz), `src/train_classifier.py`, `src/mine_stage_b.py`, `src/analyze_clustering.py` (filter ablation + merge/split), `src/explore_merge_strategies.py` (recall strategy exploration).

## Classifier — Binary (Complete)

- `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- Stage A: ShapeNet car (02958343) as positive, unknown_fraction=0.50. Best val macro F1: 0.9986
- Stage B: Mined from SemanticKITTI (train: seqs 00-07,09-10; val: seq 08; 5000 frames each, purity 0.75)
  - Train: 420,333 clusters (88,600 car / 331,733 not-car)
  - Val: 130,394 clusters (24,968 car / 105,426 not-car)
- Stage B best: epoch 13/15, macro F1 0.9225 (car P=0.837 R=0.900)
- Checkpoint: `checkpoints/stage_b_best.pth`
- **Stage A ablation done (Finding #25):** advisor's "too perfect synthetic data" concern resolved — from-scratch Stage B matches pretrained (macro F1 0.9285 vs 0.9225). Synthetic prior is redundant given 420k real clusters; pretrained keeps a weak pipeline precision edge. See `docs/classifier/`.

## Eval Metrics (Deterministic, seq 00, 100 frames)

| Metric | Value |
|--------|-------|
| Precision | 0.984 |
| Recall | 0.739 |
| F1 | 0.844 |
| Mean IoU | 0.943 |

RANSAC is deterministic (`np.random.default_rng(42)`). Results reproducible across runs.

## Recall Bottleneck — Fully Characterized

The ~0.74 recall ceiling is a **hard limit of density-based clustering** on voxelized LiDAR. Extensively investigated across two sessions:

### Root cause: HDBSCAN splitting (Finding #23)
- 31-37% of GT cars are split across multiple HDBSCAN clusters
- Large/close cars split most (66% split rate at 0-10m, 4% at 30-50m)
- Merging is negligible (0-0.5%)
- Recoverable ceiling (single-cluster GT cars): 63-68%, matching actual recall

### Geometric filter ablation (Finding #22)
- `min_volume` kills 68% of GT-matching rejected clusters, `min_points` kills 26%
- But these are sub-fragments from split cars, not independent missed detections

### Strategies attempted — all negative (Findings #21, #24)
- **BEV clustering:** F1 0.779 (vs 0.844 baseline). 2D projection merges overlapping objects.
- **Higher min_cluster_size:** MCS=20 → F1 0.852 on seq 00 but 0.801 on seq 08 (overfits).
- **Post-clustering fragment merge:** precision drops outweigh recall gains on held-out data.
- **Distance-adaptive HDBSCAN:** ring boundary artifacts; worse than global.
- **Temporal aggregation (prior session):** HDBSCAN on accumulated points → F1 collapsed to 0.073.
- **Lower cluster thresholds:** zero TP change.

**Conclusion:** All clustering-level interventions exhausted. Accept ~0.74 recall and focus on other thesis contributions.

## Alternative clustering implementations in `pipeline.py`

All disabled by default, CLI-toggleable for documentation:
- `--clustering-method bev` — BEV connected-component clustering
- `--merge-fragments` — post-clustering fragment merge
- `--adaptive-hdbscan` — distance-ring HDBSCAN with per-ring MCS

## Completion — KITTI-like PCN VERIFIED; the blobs were an inference bug (Finding #26)

Root cause of prior PCN failures (#15-19): synthetic partials were OOD from the
real post-pipeline input (voxelized 0.05 m, ground-removed, single-viewpoint).
Built a KITTI-like single-view partial generator (`_render_kitti_like` in
`src/train_pcn.py`, `--kitti-like`; see `docs/pcn/kitti_like_partial.md`) and
trained PCN on it: `checkpoints/pcn_kitti_best.pth`, best val 0.1246.

**Verdict (Finding #26): the data fix WORKED.** In-distribution synthetic eval is
clean (CD 0.16 m, F@0.1m 0.76 — real cars, not blobs). The "blobs on real data"
were **primarily an inference-normalization bug in `completion.py complete()`**:
it applies **3D PCA alignment + partial-radius/partial-centroid** normalization
that the model never saw in training (this breaks *every* PCN checkpoint, incl.
`pcn_best` — 3.5× worse CD even on in-distribution input). A corrected inference
path (no PCA; reorient gravity→Y, length→Z; scale ×1.137; full-car-center
estimate) de-blobs real seq-08 clusters into car-footprint shapes (see
`output/verify_pcn_step2/`). Scripts: `scratchpad/verify_pcn_step1.py` (synthetic,
calibration + ablation), `scratchpad/verify_pcn_step2.py` (real, multi-view + pseudo-GT).

Key sub-findings: scale is solved by the ×1.137 factor; **centroid estimation is
the dominant residual error**; training's `_augment_rotation` is roll-invariance
(about the length axis), not yaw. The static-car pseudo-GT metric is **invalid**
for completion — accumulated LiDAR is itself one-sided, so CD rewards
under-completion (raw partial scored lowest CD on every real example).

## Checkpoints

- `checkpoints/stage_b_best.pth` — binary Stage B classifier (current best)
- `checkpoints/classifier_best.pth` — binary Stage A classifier
- `checkpoints/pcn_kitti_best.pth` — PCN on KITTI-like partials (used by fixed `complete()`)
- `checkpoints/pcn_best.pth` — prior PCN (blobs on real data, #15-19)

## Immediate Next Steps

1. ~~Verify KITTI-like PCN on real data~~ — DONE (Finding #26). Data fix worked;
   blobs were an inference-normalization bug.
2. ~~Fix `completion.py complete()` + wire single-frame completion into `main.py`~~
   — DONE. Ported the corrected normalization (removed 3D PCA; reorient
   gravity→up, major horizontal axis→length; scale ×1.137; full-car-center
   estimate with up-shift + ego-side width push). Verified bit-for-bit identical
   to the validated step-2 path (`scratchpad/validate_completion_port.py`, max
   |Δ|=0). `main.py` now completes each track's **densest single frame** in the
   sensor frame and maps the result back to global (was: accumulated `all_pts`
   smear in global frame). End-to-end seq-08/100f: 18/29 car tracks completed
   (11 skipped `too_few_points`, single-frame <64 pts); outputs are car-sized
   (L≈3.5–4.2 m, W≈1.8 m, H≈1.3–1.5 m). Constants live in `completion.py`
   (`COMPLETION_SCALE_CORRECTION/CAR_WIDTH_PRIOR/UP_SHIFT`).
3. ~~Heading A/B (PCA vs L-shape) + input gating~~ — DONE (Finding #27).
   Heading method is **neutral** on real data (18/47 plausible cars either way);
   the dense-poor tracks were bad *inputs* (fragments/merges), not heading
   failures. Repurposed the L-shape fit as an **input-quality gate** (skip
   fragments fit-len<2.7 m and merges fit-width>2.3 m): completion precision
   38%→69%, all plausible cars retained (`output/08_ab_gated`). Gate is on by
   default in `completion.py`; heading default = `lshape`. **Methodology:** BEV
   diagnostics use the X–Z horizontal plane; the vertical axis is Y but the
   frame is **Y-down** (world up = −Y; `poses @ Tr` sends sensor +Z to
   (0, −1, 0)), so side-elevation plots must negate Y. See Finding #27 note 3.
4. ~~Re-run full seq-08 with the gate on~~ — DONE. Regenerated `output/08`
   with `--seq 08 --frames 5000` (4071 frames; pre-gate output preserved at
   `output/08_pregate`). 1005 accepted car tracks; **518 completed** (was 884
   pre-gate). The gate diverted 365 low-quality inputs from completion
   (294 `fragment_input`, 71 `merge_suspected`); plus 122 `too_few_points`.
   Skipped tracks are still saved as raw partial points. Accepted-track count
   unchanged (1005 vs 1006), confirming the gate filters what gets completed,
   not what gets detected.
5. **PoinTr completion — IN PROGRESS.** Targets the 8/26 implausible
   *clean*-input completions (genuine PCN model error), not heading or gating;
   transformer completers handle severe one-sided partiality better. Faithful,
   self-contained PoinTr core implemented (`src/pointr.py`, 8.9M params: FPS+DGCNN
   point proxies, geometry-aware block on the 1st enc/dec layer per the paper's
   model-E ablation, dynamic query generator, per-proxy FoldingNet,
   predict-missing-then-concat-input; loss = CD(coarse,GT)+CD(fine,GT), exact —
   no subsample). Trains via `src/train_pointr.py --kitti-like`, reusing
   `ShapeNetCompletionDataset`/`_render_kitti_like` verbatim so it shares the PCN
   baseline's data; AdamW lr 5e-4, batch 16 (~3.8 GB VRAM), 100 epochs →
   `checkpoints/pointr_kitti_best.pth`. `completion.py _load_model()` dispatches
   PoinTr vs PCN by checkpoint (`pointr_config` key); `complete()` is unchanged.
   **Next: finish training, then compare vs PCN (val CD/F-score, then real seq-08
   plausibility) per the experiment protocol before deciding to swap.**
6. **Thesis writing** — pipeline description, experiment results, discussion of recall ceiling
7. **Pipeline diagram** for thesis report

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
