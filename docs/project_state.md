# Project State

Last updated: 2026-10-14

> **Full chronological record: `docs/session_history.md`.** This file holds current state only.
> Experiment detail lives in `docs/findings.md`. Pre-submission detail removed from here on
> 2026-10-12 (Gemini-review prompt, Task G step log, donor-metric summary, delegate brief,
> old defense-deck plan status) remains in `session_history.md` and git history.

## PHASE: THESIS DEFENSE PREP — thesis SUBMITTED

- **Thesis submitted.** Source of truth for every defense claim: `docs/writing/thesis/`
  (`main.tex` → `ch1`–`ch5`; `main.pdf` built 2026-09-16 after the last `.tex` edit). Title:
  *"A Modular Detection and Completion Framework for Occluded Vehicles in Automotive LiDAR."*
  **No further manuscript edits.** `ch0_abstract.tex` is NOT in the submitted PDF (unwired
  2026-09-05). `thesis_full.tex/.pdf` deleted from the working tree. The submitted-state `.tex`
  / `main.pdf` edits are still **uncommitted**.
- **Defense: 2026-10-25.** 15 min talk + 30 min Q&A. Slides in English, talk in Vietnamese.
  Committee assumed non-specialist. Built **locally** (python-pptx + PowerPoint; Canva dropped 2026-10-14); diagrams drawn by the user
  (simple → Canva, complex → draw.io → SVG). No template requirement.
- **Plan of record: `docs/defense/outline.md`.** 2026-10-14: deck restructured to **Deck v2**
  (59 slides, peer FPT deck's I–IV layout/structure, section-prefixed numbering 1.x / 2.x.y /
  3.x.y / 4.x, ★ talk path ≈ 14:50 with no slack, ○ support slides in place for Q&A).
  **Built by code:** `docs/defense/scripts/build_deck.py` → `docs/defense/deck_v2.pptx`
  (★/○ + thesis source in speaker notes). Canva AI edits failed at this scale; change
  structure/text in the script, not via Canva AI. D3 pipeline strip dropped.
  The old `docs/defense/plan.md` + `storyboard.md` (2026-08-30, pre-reframe, donor-metric
  headline) were deleted 2026-10-12; they're recoverable from git (commit d06c248) if needed.
- **Numbers spot-check vs `results/postfix_normalfix_2026-09-14/EDIT_MAP.md`: PASSED
  2026-10-12.** All post-fix tables and prose deltas verified; no stale pre-fix values.

## Next Session

**Session S2 (scheduled Oct 15):** preprocessing, RANSAC ground removal, voxel downsampling,
HDBSCAN → diagrams D4 (RANSAC plane fit) and D5 (clusters before/after) + quiz. Same format as
S1: notes file `docs/defense/s2_*.md` (concepts in core → computation → numeric example → why
order; Q&A traps; step-by-step drawing; glossary additions).

Carry-over from S1 / 2026-10-14:
0. **Canva dropped (2026-10-14): the deck is finished locally** — structure/text in
   `build_deck.py`, then fill diagrams/tables into `deck_v2.pptx`. Fonts: Be Vietnam Pro is not
   installed on this PC, so PowerPoint falls back; install it (or switch the script's FONT) before
   judging line wraps, and embed fonts when saving the defense PDF.
1. **S1 quiz:** Q1 ✓ (mostly), Q3/Q4 half right, Q2/Q5/Q6 missed; reading list given; the
   reworded retry (Q2–Q6, posed 2026-10-14) is unanswered — user skipped it to work on Canva.
2. **Diagrams:** user draws only *technical* diagrams (D3, D4, D6, D7, D8, D11, D12-schematic);
   D1 was drawn by Claude as SVG (`docs/defense/figures/d1_occlusion.svg`, labels enlarged) and
   is already in the deck. D3 still to draw.
3. Early **Canva → PPTX/PDF export test** (fonts, animations). PPTX half done 2026-10-14:
   Canva embeds "Be Vietnam" (.fntdata) but PowerPoint on this PC renders a bold Arial-like
   fallback, so line breaks shift; PPTX page is 1440×810 pt (Canva size N = N pt, so body 32 ≈
   21 pt on a standard 960×540 slide). → Don't present from PPTX. PDF half PASSED same day:
   Be Vietnam subsets embedded; PyMuPDF render vs Canva PNG differs on ≤0.9% of pixels per
   slide (anti-aliasing). **Defense-day file = PDF** (export test was on the Canva deck; re-check fonts on the local PDF).
   Animations not tested (plan uses build slides instead).

Remaining schedule (`outline.md`): S3 Oct 16 (geo filter, classifier, auto-mining) · S4 Oct 17
(tracking, eval protocol, results, recall analysis) · S5 Oct 18 (PCN, partials, normalisation,
L-shape) · S6 Oct 19 (completion evaluation) · Oct 20–21 deck + Vietnamese speaker notes ·
Oct 22 `docs/defense/qa_bank.md` + backup slides · Oct 23–24 mock defenses.

## Known Thesis Weak Spots (Q&A prep — collect here as found)

- **OBB wording contradiction:** ch3 §3.1 says detection "emits an oriented bounding box";
  ch2 §2.2.5 says the pipeline "never fits an oriented box". Reality: a tight OBB around the
  observed points is fitted per cluster (geometric filter + classifier features, ch3 §3.2.5);
  it isn't scored, because the evaluated output is the point cluster. Answer prepared in
  `docs/defense/s1_problem_data_framework.md` Part B #3.
- **Seq-00 full F1 0.838 is not in the thesis.** For seq 00 quote the LOSO fold-00 row
  (P 0.921 / R 0.758 / F1 0.831).
- **Recall denominator:** eligibility requires ≥10 surviving points. ~25.5% of annotated cars
  are excluded, so implied recall vs *all* annotated cars is ≈0.54 (Finding #48). Verify how
  ch4 states this before S4.
- **Objective 2 has no matching contribution:** ch1 Objective 2 is "establish a robust
  methodology for evaluating completion on real data", but ch1's two contributions (framework;
  zero additional annotation) don't claim the evaluation method (it's Chen-2020-style
  plausibility, #54). ch5 "Principal Outcomes" does claim a "non-circular methodology". Prepare
  the answer: the method is adopted, our part is applying it with an independent judge and a
  pre-registered gate. Affects the 3→2 mapping on slide 4.
- **"Annotation-free" framing:** always "zero additional manual annotation"; classifier labels
  are auto-mined from SemanticKITTI GT.

## Canonical Numbers (as stated in the submitted thesis)

- **Detection, seq 08 headline** (`tab:headline`): P 0.911 / R 0.732 / F1 0.812 / mIoU 0.913;
  TP 25568 / FP 2513 / FN 9354. Held-out LOSO fold-08: 0.888 / 0.736 / 0.805.
- **LOSO pooled, all 11 seqs** (`tab:loso`): P 0.888 / R 0.717 / F1 0.793 / mIoU 0.919.
- **Stage ablation** (`tab:ablation-stage`): geometric-only P 0.199 / R 0.773 (FP 108,831) →
  +classifier 0.846 / 0.676 (FP 4,303) → +track filter 0.911 / 0.732 (FP 2,513).
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
