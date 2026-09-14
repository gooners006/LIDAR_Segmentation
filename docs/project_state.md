# Project State

Last updated: 2026-09-15

> **Full chronological record: `docs/session_history.md`.** This file was compacted on
> 2026-09-07 — the historical per-session narrative blocks (thesis reframe/polish/review
> rounds, chapter-by-chapter drafting, delegate-brief T1-T13, completion Directions
> 1/2/4a, cross-file audits) were removed from here and a complete pre-compaction snapshot
> was appended to `session_history.md`. This file now holds current state only. Experiment
> detail lives in `docs/findings.md`.

## RESEARCH FREEZE — LIFTED 2026-09-09 (was declared 2026-08-21)

The write-up freeze is **lifted** (user, 2026-09-09): the pipeline may be re-run and models
retrained. Ordinary data-safety hygiene still applies (do not overwrite `output/08` or existing
checkpoints without cause; new work to fresh paths). The committed reference configuration
(`PIPELINE_CONFIG` at `src/pipeline.py:10`; checkpoints `stage_b_scratch_best.pth`,
`pcn_kitti_best.pth`; eval protocol point-level IoU ≥ 0.3 greedy 1-to-1, supported-vehicles,
track filter ON, micro-averaged) is documented in the sections below.

## Thesis Status (2026-09-15) — REFRAME DONE; figure/caption/prose polish DONE

Reframed across all 7 `.tex` files; title *"A Modular Detection and Completion Framework for
Occluded Vehicles in Automotive LiDAR."* Since 2026-09-14 a full figure + editorial pass landed:
TikZ pipeline/architecture/completion diagrams, geometric-filter and L-shape-gate figures, a
real-data §2.1.4 completion figure, background figures (`bg_*`), per-step Input/Output headers in
Ch3, two §4.1 dataset/sensor tables, a de-bloated caption set (all 41 ≤53w), and a run-on /
comma-splice / conversational-phrasing line-edit. Builds clean: `latexmk` exit 0,
**0 undefined refs/citations**, **71 pp**, `main.pdf` == `thesis_full.pdf` (byte-identical,
3197255 bytes). New bib key `zhang2017lshape`. Editorial only — no metric changes since the
2026-09-14 height-gate cascade.

## Next Session (updated 2026-09-15)

**Outstanding, in priority order:**
1. **Gemini structural + line review** of the thesis (prompt block below; feed
   `docs/writing/thesis/thesis_full.tex`). Not yet done.
2. **Numbers spot-check** vs `results/postfix_normalfix_2026-09-14/EDIT_MAP.md` (details below);
   not yet done.
3. Optional **de-bold/de-emph minimisation** pass (`\textbf`/`\emph` reduction, subjective).
4. **Advisor review** — the manuscript has never been advisor-reviewed.

Done as of 2026-09-15 (this commit): figure set (pipeline/architecture/completion TikZ diagrams,
geometric-filter + L-shape-gate figures, §2.1.4 real-data figure, `bg_*` background figures), Ch3
per-step Input/Output headers, §4.1 dataset/sensor tables, Table 4.3 seq-08 separator + PCN-role
row, 0.08°→0.09° reconciliation, `zhang2017lshape` citation, full caption-bloat trim, and the
run-on/comma-splice/conversational line-edit. Build is clean at 71 pp, PDFs byte-identical.

---

**Background: height-gate normal-sign fix + full re-run cascade (2026-09-14).** The RANSAC
ground-plane normal sign is now forced upward in `src/pipeline.py remove_ground`
so the signed height gate is no longer bypassed. Every seq08 detection number and
the LOSO aggregate were re-run at full sequence and propagated across
ch2/ch4/ch5/abstract. Post-fix canonical numbers: seq08 headline
**P0.911/R0.732/F1 0.812**; seq00 full **0.838**; LOSO pooled
**P0.888/R0.717/F1 0.793** (FP −2158). Provenance archived in
`results/postfix_normalfix_2026-09-14/` (per-fold LOSO, seq08 ablation/IoU/coarsening
JSONs, headline JSONs, split/gate `ac_seq08_*.txt`, `distance_recall_postfix.json`,
and `EDIT_MAP.md` with the full old→new map). `thesis_full.tex` is a flattened
single-file build (compiles byte-identical to `main.pdf`).

**Suggested spot-check (do first):** rebuild `latexmk main.tex` and skim the PDF, then
verify against `results/postfix_normalfix_2026-09-14/EDIT_MAP.md`:
1. Table cells: `tab:headline`, `tab:ablation-stage` (§4.2), `tab:iou-sensitivity`,
   the per-fold LOSO table, `tab:distance`, `tab:split`, `tab:splitrate`, `tab:clustering`.
2. Hand-recomputed prose deltas — §4.2 ablation ("104000 removed", FP 108831→4303,
   "1400 TP forgone / 106000 FP removed") and §4.4 split analysis (clean fraction 55.1%
   vs recall 0.710, gap 15.9→7.2 pts, #6 merge 56→284, "none newly merge" now corrected).
3. Cross-table seam: `tab:clustering` seq08 HDBSCAN and `tab:distance` pooled must both
   read 0.840/0.671/0.746.
4. Cross-file headline/LOSO consistency in ch2, ch5, abstract, project_state.

**Suggested next: run a Gemini structural + line review** on the thesis. Feed
`docs/writing/thesis/thesis_full.tex` as the Phase 1 "complete .tex file". Prompt to use:

```
Role: Senior Academic Editor & Peer Reviewer
Task: Perform a two-phase Macro-to-Micro Review (Structural & Line Edit) on my LaTeX thesis.

[PERSONA & GUIDELINES]
- Academic Rigor: Critique like a dissertation committee member. Identify logical leaps, unbacked claims, structural redundancies, and weak chapter transitions.
- Prose Stylist: Smooth out clunky, mechanical, or repetitive phrasing into natural, sophisticated academic prose. Maintain appropriate scholarly distance and proper hedging (e.g., "suggests" vs. "proves").
- Voice Preservation: Polish rhythm and sentence cadence without replacing my voice with generic AI buzzwords.

[STRICT LATEX PRESERVATION RULES]
- Command Safety: DO NOT alter, remove, or break any LaTeX markup, including citations (\cite{}), cross-references (\ref{}, \label{}), footnotes (\footnote{}), or custom macros.
- Environments & Math: Keep all environments (\begin{...}...\end{...}) and inline/display math intact and properly closed.
- Formatting Integrity: Return raw, compilable LaTeX syntax inside code blocks for easy copy-pasting.

[WORKFLOW]

Phase 1: Macro Review (Structural & Conceptual Edit)
I will provide: The complete .tex file or chapter outline.
Your output:
1. Narrative & Argument Flow: Evaluate how cleanly the core argument builds across chapters/sections.
2. Structural Gaps: Identify missing theoretical links, redundant points, or weak section transitions.
3. Recommendations: Provide 3–5 high-level structural adjustments before line editing begins.

Phase 2: Micro Review (Line & Prose Edit)
I will provide: Section-by-section chunks (\section or \subsection, approx. 500–1,500 words).
Your output:
1. Diagnostic: Briefly note 2–3 style or rhythm issues in the provided chunk.
2. Revised LaTeX Text: Provide the complete, polished line rewrite inside a code block with all LaTeX commands fully preserved.
3. Key Adjustments: Summarize 2–3 major stylistic improvements made.

Acknowledge these instructions and confirm you are ready to receive the Phase 1 submission (full .tex file or structural outline).
```

- **Contributions (new):** C1 = the modular detection-and-completion **framework** (integration
  of RANSAC/HDBSCAN/PointNet/PCN, end-to-end, inspectable); C2 = auto-mined supervision and
  synthetic completion (**SemanticKITTI + ShapeNet**), requiring zero additional manual
  annotation. Donor metric, amodal-box utility, and home-made L/W/H
  plausibility all **removed** (grep `donor|amodal` across `docs/writing/thesis/*.tex` = 0).
  Overclaims "near-annotation-free" / "trained without point-wise labels" / "sees no
  hand-labelled real cars" removed 2026-09-12: classifier labels are auto-mined from
  SemanticKITTI's existing semantic labels, so the honest claim is "zero additional manual
  annotation," not "annotation-free" (see memory [[annotation-framing]]).
- **RQs (new):** RQ1 = can the modular pipeline requiring zero additional manual annotation detect+complete occluded
  vehicles (answered by LOSO detection, pooled P0.888/R0.717/F10.793); RQ2 = does completion
  produce car-plausible shapes per an independent classifier + how it fares on fidelity.
- **Completion evaluation (new spine):** synthetic CD 0.16 / F@0.1 0.76 (network works) +
  Chen-2020 independent-classifier plausibility (#54: raw 0.253 → completed 0.785, at ceiling
  0.807) + reference-free fidelity UCD/UHD reported with the under-completion caveat (#52).
- **Done:** Ch1+main+abstract, Ch2 (UCD/UHD+plausibility math, completion figure, metrics table),
  Ch3 (donor section deleted, results migrated to Ch4, configured-formula cross-refs), Ch4
  (completion rebuilt on #54 + dataset table + supervised/unsupervised table + plausibility
  figure/table), Ch5 (reframed conclusions/limitations/future-work). Style: em-dashes removed
  (rendered PDF), colon-not-period on `\paragraph` leads.
- **Remaining:** the de-bold/de-emph minimization pass (`\textbf`/`\emph` reduction, subjective)
  and a full numbers spot-check. Figure polish is **done** (2026-09-15).
- **Plans (outside repo):** `~/.claude/plans/drop-the-whole-donor-frame-nifty-papert.md`
  (reframe, executed) and `~/.claude/plans/independent-classifier-plausibility.md` (Task G, done).

## Task G — Independent-Classifier Plausibility (DONE, 2026-09-12)

The prerequisite completion experiment (Finding #53 pre-registration -> **#54 results**; plans
`~/.claude/plans/independent-classifier-plausibility.md`). **Complete and FAVORABLE.** Next task
is the Ch4 reframe that cites #54.

**Headline (#54, pooled N=2822):** raw-partial car-acceptance 0.253 -> **completed 0.785**
(McNemar p~=0, Wilcoxon p~=0), reaching the static-accumulation ceiling 0.807; on car-prob
completed (0.745) is at/above ceiling (0.728). Movers complete as plausibly as statics (0.837).
Heading-flip sensitivity negligible (+0.009). Artifacts:
`output/experiments/plausibility_chen/{summary.json, per_seq.json, acceptance.png}`; harness
`scratchpad/plausibility/{build_ceiling,harmonize,score}.py`.

- **Judge — DONE.** In-repo PointNet++ SSG (Option A, built on `src/pointr.py` FPS/index/distance
  primitives + ball query; no new deps, no CUDA). Code `scratchpad/plausibility/{pointnet2_ssg,
  modelnet40,train_judge}.py` (gitignored). Trained from scratch on `modelnet40_ply_hdf5_2048`
  (`dataset/modelnet40/`, gitignored), seed 42, 100 epochs. **Best test top-1 = 0.9214** (gate
  >=0.88 PASS; reference-faithful, no fallback). Checkpoint `checkpoints/modelnet40_pointnet2.pth`.
- **Generation — DONE.** All 11 labeled sequences (00-10) run at production config to
  `output/experiments/plausibility_gen/<seq>/` (isolated from `output/08`). Per track:
  `<id>_partial.ply` (raw PCN input) + `<id>.ply` (completed, 4096 pts). **2,822 completed tracks**
  total (paired-test N). `tracks.json.tracks[]` has `centroid_history` (static/mover), `completed`,
  `ref_fit_length/width`.
- **Ceiling (step 4) — DONE (2026-09-12).** Static-accumulation upper bound built by
  `scratchpad/plausibility/build_ceiling.py` (reuses `scratchpad/amodal_gt.py` accumulation +
  `well_observed` gate unchanged; it adds the point-cloud dump amodal_gt.json omits). **140
  qualifying dense clouds** (of 1861 observed) across 8 contributing seqs (highway 01/04 → 0, as
  expected) → `output/experiments/plausibility_ceiling/<seq>/{<inst_id>.ply, manifest.json}`.
  Clouds saved object-centered, **world (cam0) orientation preserved**; manifest carries
  `yaw_deg`/`dims_lwh`/`center_world` for step-5 reorientation. 00/05/08 counts (46/11/40)
  reproduce the cached `amodal_gt.json` exactly (deterministic).
- **Harmonization (step 5) — orientation MEASURED; state-doc claim CORRECTED.** The earlier
  "completed and ModelNet40 both y-up, no rotation needed" was **WRONG**. Verified against the
  data (`scratchpad/plausibility/figs/orientation_check.png`): ModelNet40 (judge) is **up=+Y**
  (84/100 bottom-wider; sedan renders roof-up), but BOTH pipeline sources are **up=-Y** (cam0
  frame) — completed (skew -0.29 vs MN40 +0.41; same pipeline frame as ceiling) and ceiling
  (0/61 bottom-wider, decisive; renders roof-down). All three have Y vertical / Z length / X
  width EXCEPT ceiling is yaw-rotated in X-Z. **The judge trained with NO rotation aug, so it is
  orientation-sensitive** — scoring unflipped would feed upside-down cars and yield a false
  honest-negative (the Finding-#26 semantics trap). **Fix:** completed → 180deg about length axis
  Z `(x,y,z)->(-x,-y,z)`; ceiling → de-rotate by `yaw_deg` about Y, reorder to X-wid/Y/Z-len,
  then the same 180deg-about-Z flip. Then FPS-to-1024 + unit-sphere + min-point gate on raw
  partials. Residual front/back **heading sign** (L-shape leaves 180deg ambiguous) handled by
  scoring both ways and reporting sensitivity (low-risk for a binary car judge).
- **Steps 5-8 DONE (2026-09-12):** harmonization (`harmonize.py`, orientation-verified), scoring
  (`score.py`, 2,822 pairs + 140 ceiling), stats (McNemar/Wilcoxon/MW), and Finding #54 recorded.
- **NEXT — the thesis reframe** (`~/.claude/plans/drop-the-whole-donor-frame-nifty-papert.md`):
  Task G's #54 supplies the favorable real-data completion number. Execution order per the reframe
  plan: Ch1+main contribution -> Ch2 -> Ch3 -> Ch4 (cite #54: plausibility + fidelity, drop donor
  metric) -> Ch5 -> global style passes.
- **Source layout / build:** unchanged — six files under `docs/writing/thesis/` (`ch0_abstract`,
  `ch1_introduction`, `ch2_background`, `ch3_methodology`, `ch4_evaluation`,
  `ch5_discussion_conclusion`) wired by `main.tex`; clean build, ~76 pp. Still NOT advisor-reviewed.

## Evidence Tasks — all closed

Read-only analyses against the frozen config (freeze-safe). Full detail in `docs/findings.md`.

- **B1** (IoU sensitivity) + **B6** (geometric-only ablation) — DONE (#47).
- **B2** (GT-eligibility count) — DONE (#48): ~25.5% of annotated cars excluded by the
  >=10-surviving-points rule -> implied recall vs all annotated cars ~=0.54.
- **LOSO cross-validation** — DONE (#51, `results/loso/`): leakage was precision-only
  (~2 pts); pooled over 11 seqs P 0.888 / R 0.717 / F1 0.793 (post height-gate fix, 2026-09-14). Answers the
  "validation-as-test" critique; shipped checkpoint + `PIPELINE_CONFIG` untouched.
- **UCD/UHD reference-free comparison** — DONE (#52): standard input-fidelity metrics rank
  raw partial = mirror = 0 and the completion worst, while donor cov@0.1 credits the
  completion. Strengthens the metric's novel-set-restriction novelty.
- B3 (literature table) / B4 (distance recall) / B5 (pipeline diagram) — writing/figure
  work, folded into the chapters.

## Defense Deck (Phase 8) — planned, not yet built

Planning artifacts committed (`docs/defense/plan.md` + `docs/defense/storyboard.md`),
grounded in the peer template `docs/LVTN.pptx` (same advisor) and the frozen numbers.
Locked: python-pptx onto a cleaned copy of `LVTN.pptx`; 25-30 min / ~40 slides + 7 backups;
completion-forward (donor metric headline); own-the-limitations; reuse thesis figures.
**Resume at plan Section B** to build the `.pptx` (purge peer content from the template copy;
python-pptx slide-clone corruption is the known risk -> fallback fresh 16:9 theme). Deck NOT
yet generated.

## Current Architecture

| Stage | Description | File | Status |
|-------|------------|------|--------|
| 1-3 | Z-filter, denoise, downsample | `src/pipeline.py` | Working |
| 4 | HDBSCAN clustering | `src/pipeline.py` | Working (recall ceiling confirmed) |
| 5 | Geometric filtering (ground-plane-relative) | `src/pipeline.py` | Tuned |
| 6 | Classification (dual-branch PointNet, binary car/not-car) | `src/classifier.py` | Binary Stage B trained |
| — | Centroid tracker + track-level filtering | `src/tracker.py`, `src/evaluate.py` | Working |
| 7 | Point completion | `src/pcn.py`, `src/completion.py` | Fixed inference (#26); single-frame completion in `main.py`; L-shape input gate (#27) → completion precision 38%→69%; length prior (#35) → far_end cov 0.13→0.32; per-car length estimate (#36) → box \|ΔL\| 0.354→0.304, compact overshoot fixed |

Key files: `src/main.py` (runner), `src/evaluate.py` (metrics + sweep flags), `src/visualize_gt.py` (GT vs pipeline toggle viz), `src/train_classifier.py`, `src/mine_stage_b.py`, `src/analyze_clustering.py` (filter ablation + merge/split), `src/explore_merge_strategies.py` (recall strategy exploration),
`src/test_invariants.py` (pytest invariant tests, T4).

## Classifier — Binary (Complete)

- `CLASS_LABELS = ["car", "not-car"]`, `NUM_CLASSES = 2`
- **Production (since 2026-07-14, Finding #31): `checkpoints/stage_b_scratch_best.pth`** —
  trained on real mined clusters only, from random init. Stage A synthetic
  pretraining dropped from the final pipeline (kept as thesis ablation, #7/#25/#30).
- Stage A (ablation material): ShapeNet car (02958343) as positive, unknown_fraction=0.50. Best val macro F1: 0.9986
- Stage B: Mined from SemanticKITTI (train: seqs 00-07,09-10; val: seq 08; 5000 frames each, purity 0.75)
  - Train: 420,333 clusters (88,600 car / 331,733 not-car)
  - Val: 130,394 clusters (24,968 car / 105,426 not-car)
- Scratch best: epoch 14/15, macro F1 0.9285; fine-tuned (A→B) best: epoch 13/15, macro F1 0.9225
- **Stage A ablation done (Finding #25):** advisor's "too perfect synthetic data" concern resolved — from-scratch Stage B matches pretrained (macro F1 0.9285 vs 0.9225). Synthetic prior is redundant given 420k real clusters; pretrained keeps a weak pipeline precision edge. See `docs/classifier/`.
- **Cross-domain matrix done (Finding #30, advisor-requested):** sim-to-real gap is total and symmetric — car F1 = 0.000 in every off-diagonal cell (synthetic-trained on real, real-trained on synthetic); fine-tuning forgets synthetic entirely. Script: `scratchpad/cross_domain_classifier_eval.py`; results: `output/experiments/cross_domain_classifier/`.

## Eval Metrics

Headline numbers below use the production scratch checkpoint (Finding #31;
prior fine-tuned-checkpoint numbers preserved there).

### Seq 00 (deterministic, 100 frames) — headline baseline

| Metric | Value |
|--------|-------|
| Precision | 0.984 |
| Recall | 0.761 |
| F1 | 0.859 |
| Mean IoU | 0.942 |

TP=1242 FP=20 FN=389. RANSAC is deterministic (`np.random.default_rng(42)`). Results reproducible across runs.
Promoted-config 100-frame eval (2026-07-23, report refresh): P 0.967 / R 0.777 / F1 0.862 / mIoU 0.962. TP=1296 FP=44 FN=371 (TP/FP/FN not separately recorded 2026-07-23; re-run and filled in 2026-08-02, same command/config, metrics unchanged — `.venv\Scripts\python.exe src/evaluate.py`).

### Seq 08 (full, 4071 frames) — generalization check (promoted config, updated 2026-09-14 after height-gate fix)

| Metric | Value |
|--------|-------|
| Precision | 0.911 |
| Recall | 0.732 |
| F1 | 0.812 |
| Mean IoU | 0.913 |

TP=25568 FP=2513 FN=9354. Promoted `PIPELINE_CONFIG` (voxel_before_denoise,
ransac_iterations=300, cluster_voxel_size=0.10; Finding #34). Height-gate
normal-sign fix (2026-09-14, `src/pipeline.py remove_ground`) cut false
positives (FP 2676→2513) and lifted F1 0.808→0.812; the fix cascaded through
every seq08 detection table in Ch4 (headline, ablation, IoU sweep, coarsening,
recall-by-distance, gate ablation, split analysis) plus the LOSO aggregate.
Pre-fix numbers (0.905/0.730/0.808/0.912, TP=25478 FP=2676 FN=9444) and the
earlier pre-opt numbers (0.903/0.699/0.788/0.895, TP=23823 FP=2565 FN=10240) preserved in #34.
Command: `.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000`.
Confirms the seq-00 story at 40× scale: precision-saturated, recall-limited;
per-frame recall anti-correlated with GT-car density, FP flat ~1/frame.
Figures: `output/figures/seq08_{bev_detections,failure_zooms,timeseries}.png`.

## Recall Bottleneck — Characterized; partly lifted by coarse-voxel clustering (#34)

The recall shortfall was root-caused to HDBSCAN **splitting cars into fragments**
(not a classifier problem). Six repair strategies all failed to generalise
(four with table outcomes + temporal aggregation and threshold lowering in prose;
see Ch 5 reconciliation note above) — **but** coarsening the clustering resolution (cv=0.10, promoted
2026-07-23, Finding #34) closed some intra-car density gaps and lifted recall
0.699→0.730, so the earlier "~0.74 hard limit" was **partly a resolution
artifact**, not fundamental. A smaller structural limit remains (density-based
clustering has no notion of objectness; coarser voxels would start merging
adjacent cars). Extensively investigated across multiple sessions:

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

**Conclusion:** Post-hoc (reassemble-after-splitting) interventions exhausted —
all negative. The one lever that worked was clustering *resolution* itself
(cv=0.10, now in production, #34), which recovered part of the loss (recall
0.699→0.730). Residual ~0.73 recall accepted; focus on other thesis contributions.

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
trained PCN on it: `checkpoints/pcn_kitti_best.pth`, best val loss 0.1246
(= coarse CD + 0.5·fine CD; val fine-CD 0.066).

**Verdict (Finding #26): the data fix WORKED.** In-distribution synthetic eval is
clean (CD 0.16 m, F@0.1m 0.76 — real cars, not blobs). The "blobs on real data"
were **primarily an inference-normalization bug in `completion.py complete()`**:
it applies **3D PCA alignment + partial-radius/partial-centroid** normalization
that the model never saw in training (this breaks *every* PCN checkpoint, incl.
`pcn_best` — 3.5× worse CD even on in-distribution input). A corrected inference
path (no PCA; reorient gravity→Y, length→Z; scale ×1.137; full-car-center
estimate) de-blobs real seq-08 clusters into car-footprint shapes (see
`output/experiments/verify_pcn_step2/`). Scripts: `scratchpad/verify_pcn_step1.py` (synthetic,
calibration + ablation), `scratchpad/verify_pcn_step2.py` (real, multi-view + pseudo-GT).

Key sub-findings: scale is solved by the ×1.137 factor; **centroid estimation is
the dominant residual error**; training's `_augment_rotation` is roll-invariance
(about the length axis), not yaw. The static-car pseudo-GT metric is **invalid**
for completion — accumulated LiDAR is itself one-sided, so CD rewards
under-completion (raw partial scored lowest CD on every real example).

## Checkpoints

- `checkpoints/stage_b_scratch_best.pth` — binary classifier, real-data-only from scratch (**production**, Finding #31)
- `checkpoints/stage_b_best.pth` — binary Stage B fine-tuned from Stage A (kept for reproducibility)
- `checkpoints/classifier_best.pth` — binary Stage A classifier (ablation material)
- `checkpoints/pcn_kitti_best.pth` — PCN on KITTI-like partials (used by fixed `complete()`)
- `checkpoints/pcn_best.pth` — prior PCN (blobs on real data, #15-19)
- `checkpoints/modelnet40_pointnet2.pth` — ModelNet40 PointNet++ SSG plausibility judge (Task G,
  test top-1 0.9214; gitignored, real-data-independent, not part of the detection/completion pipeline)

## Completion — Frozen Results Summary

Mechanism is described above; full result tables and worked verdicts live in
`docs/findings.md` and `docs/completion/donor_metric.md`. Frozen headline numbers
(seq 08, promoted config, per-car medians):

- **Donor coverage @0.1** (novel-side surface recovery): raw 0.000 / mirrored baseline
  0.043 / **completed 0.304** (#32/#35/#36; ~7x the symmetry-mirror baseline).
- **Amodal-box utility:** completed beats raw partial — BEV IoU 0.725 -> **0.771**,
  |dL| -> 0.304, center err -> 0.184 (#29/#35/#36). Length uses a per-car q90 estimate
  (`COMPLETION_LENGTH_TRACK_QUANTILE=90` + 0.12 m), 4.14 m fallback below 5 frames;
  fallback fires on 119/518 (23%) of completed tracks on seq 08 (#41).
- **Movers** complete as plausibly as statics (57.9% vs 53.7%, #44) — plausibility only,
  not accuracy-validated.
- **Held-out (seq 00) pre-registered replication: PARTIALLY HOLDS** (#42) — both primaries
  (BEV IoU, donor cov@0.1) are decisive wins; the downgrade is a coverage gap (empty long
  band), not a weak metric.
- **T13 Step-1c** (decouple radius / fill_z) tested NEGATIVE — do not ship; flags default
  OFF (#45).
- **Roadmap / open ideas:** `docs/completion/plan.md`, `docs/completion/next_ideas.md`.

## Maintenance / Delegate Brief

Delegate-brief tasks **T1-T13 all complete** (one commit each; full record in the
`session_history.md` snapshot and `docs/findings.md`). Only **T14 (thesis, user-supervised)**
remains.

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. ~~Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN~~ — done (Finding #43, T10)
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
