# Project State

Last updated: 2026-10-16

> **Full chronological record: `docs/session_history.md`.** This file holds current state only.
> Experiment detail lives in `docs/findings.md`.

## PHASE: THESIS DEFENSE PREP — thesis SUBMITTED

- **Thesis submitted.** Source of truth for every defense claim: `docs/writing/thesis/`
  (`main.tex` → `ch1`–`ch5`; `main.pdf` built 2026-09-16). Title: *"A Modular Detection and
  Completion Framework for Occluded Vehicles in Automotive LiDAR."* **No further manuscript
  edits.** `ch0_abstract.tex` is NOT in the submitted PDF. Submitted state committed (`fb19c4f`).
  Student ID: 24MSE13205 is official (deck); thesis says MSE13205 — fix only in a post-defense
  final copy if asked.
- **Defense: 2026-10-25.** 15 min talk + 30 min Q&A. English slides, Vietnamese talk,
  non-specialist committee.
- **All defense work lives in `docs/defense/`:** `outline.md` (plan of record + slide table +
  diagram inventory + schedule), `s1_*`/`s2_*`/`s3_*.md` study notes, `deck_v2.pptx`,
  `scripts/`, `assets/`, `figures/`, `qa_backup/` (private checks), `reference/`,
  `archive/canva/`.
- **Deck is built by code:** `docs/defense/scripts/build_deck.py` → `docs/defense/deck_v2.pptx`
  (61 slides, I–IV structure, ★ talk path ≈ 14:50, ○ support slides for Q&A; thesis source in
  speaker notes). If PowerPoint has the deck open, the build writes `deck_v2_new.pptx` —
  close PowerPoint, delete it, rebuild. Refresh the slide table in `outline.md` with
  `scripts/deck_table.py` (splice lines 40–102 while the deck is 61 slides).
- **Every defense figure/diagram/helper script goes in `docs/defense/scripts/`**, never only in
  the scratchpad (user rule 2026-10-16).
- **Weak spots stay off the deck** (slides and notes); they live in study notes, `qa_backup/`,
  and the list below.
- **Diagram workflow:** user draws technical diagrams in draw.io from a Claude scaffold
  (`assets/<name>.drawio` with "?" placeholders) and exports `assets/<name>.drawio.png`
  (Zoom 200%, Transparent off, Appearance Light, diagram embedded). Claude makes data
  figures/charts (matplotlib, Segoe UI) and native PPTX tables; thesis TikZ figures are cropped
  with `scripts/crop_thesis_figure.py <fig num> <name>`.
- **Numbers spot-check vs `results/postfix_normalfix_2026-09-14/EDIT_MAP.md`: PASSED
  2026-10-12.**

## Next Session

**S4 (scheduled Oct 17):** tracking + track filter, evaluation protocol (eligibility ≥10
surviving points, IoU ≥ 0.3 matching, LOSO), results, recall analysis → notes
`docs/defense/s4_*.md` (same format: concepts → computation → numeric example → why; Q&A traps;
glossary) + quiz. Diagrams: D8 tracking timeline (user, draw.io scaffold first), D10 = thesis
`seq08_failure_zooms.png` frame-250 row. S4-scope placeholders: slides 16, 23, 26, 27, 30–33,
41, 42, 46.

**Deck status:** S1–S3 slides all filled (4–15, 22, 29, 44, 45, 47, backups 54–55). Remaining
placeholders: S4 set above; S5/S6 (17–20, 25, 35–39, 48); backups 56–61.

**Open study gaps (unverified — user skipped the retries):**
- S1: classifier labels are *mined from existing SemanticKITTI GT*; seq 08 is *validation* (why
  LOSO exists); no held-out set because seqs 11–21 are unlabelled; PCN trains on *rendered
  partials of synthetic CAD meshes*. Accuracy-comparison answer: S1 notes Part B #2.
- S2: Q3b (0.10 m grid trade: 3,700 un-split vs 230 newly merged) and Q5 (tab:clustering
  numbers, slide 45).
- S3: memorise 0.199 → 0.846 precision, 0.773 recall ceiling, purity 0.75, cross-domain 0.000
  both ways. Spoken EN/VI answers in S3 notes Part E.

**Other carry-over:**
- Fonts: Be Vietnam Pro not installed → Segoe UI fallback; export the final PDF with fonts
  embedded and check it (open item).
- Vietnamese speaker notes: Oct 20–21 (only S3 Part E drafts so far).
- Nothing committed since `70bfa40`; commit when the user asks.

Remaining schedule (`outline.md`): S4 Oct 17 · S5 Oct 18 (PCN, partials, normalisation,
L-shape) · S6 Oct 19 (completion evaluation) · Oct 20–21 fill deck + Vietnamese notes · Oct 22
`docs/defense/qa_bank.md` · Oct 23–24 mock defenses · Oct 25 defense.

## Known Thesis Weak Spots (Q&A prep — collect here as found)

- **Classifier trained on pre-promotion clusters (found 2026-10-16, S3).** ch3 §classifier says
  the training clusters come from "the same clustering and filtering the classifier meets at
  inference". But the 420,333 clusters were mined 2026-06-02/03 (`dataset/stage_b/`), the
  production checkpoint `stage_b_scratch_best.pth` is dated 2026-06-25, and the final config
  (0.10 m clustering grid, voxel-before-denoise, 300 RANSAC) was promoted 2026-07-23 (#34) with
  no re-mine or retrain; `src/mine_stage_b.py` still denoises before downsampling. LOSO folds
  reuse the same clusters. All reported numbers are measured with this checkpoint on the final
  pipeline, so they stand; whether re-mining would help is untested (no re-runs during defense
  prep). Answer prepared in `docs/defense/s3_filter_classifier_mining.md` Part B #8.
- **OBB wording contradiction:** ch3 §3.1 says detection "emits an oriented bounding box";
  ch2 §2.2.5 says the pipeline "never fits an oriented box". Reality: a tight OBB around the
  observed points is fitted per cluster (geometric filter + classifier features, ch3 §3.2.5);
  it isn't scored, because the evaluated output is the point cluster. Answer prepared in
  `docs/defense/s1_problem_data_framework.md` Part B #3.
- **"Completed car = dense surface + box" (ch3 fig:architecture, fig:completion-pipeline,
  §End-to-End Completion Path) — the box is never output or defined.** Checked 2026-10-16:
  `complete()` (`src/completion.py:529`) returns points only; `src/main.py` saves `.ply` + track
  metadata, no box. The only box is the implied canonical frame (centre, L-shape heading, 1.9 m
  width prior, per-car length) used to normalise for PCN; likely a remnant of the box-overlap
  metric removed in the 2026-09-12 reframe (ch3 still says "the box overlap gains it produces").
  Answer: "the completion estimates a full-car frame that implies a box, but the evaluated output
  is the point cloud, judged by plausibility; boxes aren't scored." D3 says "dense 4096-point
  surface" instead.
- **L-shape gate vs I-shaped scans (not discussed in the thesis; checked 2026-10-16).** "L-shape"
  is the fitting method (Zhang 2017 rectangle search, `_lshape_axes`, `src/completion.py:402`),
  not a requirement on the scan; the longer fitted side is called "length". Ran the production
  gate on synthetic footprints (4.3 × 1.8 m car): L → pass (4.32 / 1.83); side-on I → pass
  (4.25 / 0.10; the 1.9 m width prior recentres toward the hidden side); **rear/front-on I →
  rejected as `fragment_input`** (1.73 m < 2.7 m). `main.py:364–372` completes the track's
  **densest frame, chosen before the gate, with no fallback**, so a car followed from directly
  behind is never completed. The thesis only says the gate "drops a fraction of otherwise usable
  cars as false rejections" (ch4 §abl-completion); that rear-on cars are among them is inference,
  not counted (the saved `tracks.json` doesn't record fit dims for rejected tracks). Answer:
  "Side-on works: the fit finds the one visible edge, and the width prior places the hidden
  side. Rear-on, the visible face is only ~1.8 m, so the gate treats it as a fragment and skips
  it. That's deliberately conservative: skip rather than complete a car rotated 90°. The fix
  would be to fall back to the densest frame that passes the gate, or take the heading from the
  track's motion; that's future work, not in the thesis."
- **RANSAC: 300 tries and the "rare" wall fail-safe (private check, 2026-10-16; NOT thesis
  numbers; `docs/defense/qa_backup/ransac_seq08.md`).** Seq 08: in 204/4071 frames (5.0%) a wall
  wins and no ground is removed (192 still rejected at 1000 tries), because the near-vertical
  check is applied to the winner only, not to each candidate. 300 vs 1000 tries: median +1.4%
  inliers / 0.37°, but ~20% of frames gain > 5% or tilt > 1°. The thesis says the fail-safe is
  "rare" and the fit "converges well before 1000": don't say "always". Effect of the no-ground
  frames on recall: not measured.
- **Seq-00 full F1 0.838 is not in the thesis.** For seq 00 quote the LOSO fold-00 row
  (P 0.921 / R 0.758 / F1 0.831).
- **Recall denominator:** eligibility requires ≥10 surviving points. ~25.5% of annotated cars
  are excluded. **Quote ≈0.55** vs *all* annotated cars (ch4 l.234 and ch5 both say 0.55;
  Finding #48's 0.54 is not the thesis number). Slide 3.1.2 notes carry it.
- **Objective 2 has no matching contribution:** ch1 Objective 2 is "establish a robust
  methodology for evaluating completion on real data", but ch1's two contributions (framework;
  zero additional annotation) don't claim the evaluation method (it's Chen-2020-style
  plausibility, #54). ch5 "Principal Outcomes" does claim a "non-circular methodology". Prepare
  the answer: the method is adopted, our part is applying it with an independent judge and a
  pre-registered gate. Affects the 3→2 mapping on slide 8 (1.5).
- **"Annotation-free" framing:** always "zero additional manual annotation"; classifier labels
  are auto-mined from SemanticKITTI GT.

## Canonical Numbers (as stated in the submitted thesis)

- **Detection, seq 08 headline** (`tab:headline`): P 0.911 / R 0.732 / F1 0.812 / mIoU 0.913;
  TP 25568 / FP 2513 / FN 9354. Held-out LOSO fold-08: 0.888 / 0.736 / 0.805.
- **LOSO pooled, all 11 seqs** (`tab:loso`): P 0.888 / R 0.717 / F1 0.793 / mIoU 0.919.
- **Stage ablation** (`tab:ablation-stage`): geometric-only P 0.199 / R 0.773 (FP 108,831) →
  +classifier 0.846 / 0.676 (FP 4,303) → +track filter 0.911 / 0.732 (FP 2,513).
- **Geometric filter** (ch4 recall analysis, seq 08): 903,901 clusters → 127,654 pass (14.1%);
  27,718 rejected overlap a GT car (73.9% min-volume, 12.7% min-points gate).
- **Classifier:** mined 420,333 train (00–07, 09, 10) / 130,394 val (08), ~21% car, purity
  ≥ 0.75. Stage A: macro-F1 0.9285 scratch vs 0.9225 pretrained; cross-domain car-F1 0.999 /
  0.000 / 0.000 / 0.885.
- **Clustering fragmentation:** 33.7% of seq-08 cars split at 0.10 m (44.8% at 0.05 m); merges
  56 → 284 (0.2% → 0.8%); coarsening lifted recall 0.710 → 0.732 (+756 TP).
- **Distance** (`tab:distance`, stride-20 per-frame): pooled 0.840 / 0.671 / 0.746; 40 m+ recall
  0.196.
- **Completion, synthetic:** CD 0.16 m, F@0.1 m 0.76; val loss 0.1246.
- **Completion, real plausibility** (`tab:plausibility`, N=2822): acceptance raw 0.253 →
  completed **0.785** vs static-accumulation ceiling 0.807 (N=140). McNemar flips 1615 vs 115.
  Movers 0.837. Heading flip Δ 0.009. Judge: PointNet++ SSG on ModelNet40, 0.9214 test acc
  (pre-registered gate 0.88).

## Current Architecture (frozen reference configuration)

`PIPELINE_CONFIG` at `src/pipeline.py:10`. Eval protocol: point-level IoU ≥ 0.3, greedy 1-to-1,
supported vehicles, ≥10 surviving points, track filter ON, micro-averaged.

| Stage | Description | File |
|-------|------------|------|
| 1–3 | z-filter, voxel downsample, denoise | `src/pipeline.py` |
| — | RANSAC ground removal (normal sign forced up, 2026-09-14 fix) | `src/pipeline.py remove_ground` |
| 4 | HDBSCAN clustering @ 0.10 m | `src/pipeline.py` |
| 5 | Geometric OBB filter (ground-relative) | `src/pipeline.py` |
| 6 | Dual-branch PointNet, binary car/not-car | `src/classifier.py` |
| — | Centroid tracker + track-level filter | `src/tracker.py`, `src/evaluate.py` |
| 7 | Completion: L-shape gate, canonical normalisation, per-car length, PCN | `src/completion.py`, `src/pcn.py` |

## Checkpoints

- `checkpoints/stage_b_scratch_best.pth` — production classifier (real mined clusters, from
  scratch; macro F1 0.9285). Stage A synthetic pretraining = ablation only (redundant, #25/#30/#31).
- `checkpoints/pcn_kitti_best.pth` — production PCN (KITTI-like synthetic partials).
- `checkpoints/modelnet40_pointnet2.pth` — plausibility judge (not part of the pipeline).
- `checkpoints/stage_b_best.pth`, `classifier_best.pth`, `pcn_best.pth` — ablation / historical.

## Data Safety

Don't overwrite `output/08` or existing checkpoints; new work goes to fresh paths. No pipeline
re-runs are planned during defense prep. Any number for the defense must come from the
submitted thesis, not from re-runs.
