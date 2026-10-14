# Defense Outline — v1 (draft for sign-off)

Defense: **2026-10-25**. 15 min talk + 30 min Q&A. Slides in English, talk in Vietnamese.
Committee assumed **non-specialist** → explain LiDAR / point clouds / clustering from scratch;
fewer equations, more pictures.
Thesis title: *A Modular Detection and Completion Framework for Occluded Vehicles in Automotive LiDAR.*

**Source of truth: the submitted thesis** = `docs/writing/thesis/` (`main.tex` → ch1–ch5,
`main.pdf` built 2026-09-16 after the last .tex edit). Every slide number must be traceable to
a table/section there. Note: `ch0_abstract.tex` is **not** in the submitted PDF (unwired since
2026-09-05) — don't quote it as "the abstract".

**Numbers spot-check (2026-10-12): PASSED.** Checked against
`results/postfix_normalfix_2026-09-14/EDIT_MAP.md`: tab:headline, tab:ablation-stage,
tab:iou-sensitivity, tab:loso (pooled recomputed: P 0.888 / R 0.717 from TP/FP/FN),
tab:distance, tab:split, tab:splitrate, tab:clustering seam (0.840/0.671/0.746 in both), prose
deltas (104k FP removed, ~1400 TP forgone, ~106k FP, 15.9→7.2 pts, 56→284 merges / ~230 newly
merge, +756 TP). No stale pre-fix values found. Caveat for Q&A: the seq-00 full-sequence
F1 0.838 is **not** in the thesis — for seq 00 quote the LOSO fold-00 row (F1 0.831).

## The story in one breath

> LiDAR sees a car from only one side. We build a pipeline of small, inspectable steps that
> (1) finds the cars and (2) fills in their missing side, without any new manual labelling.
> Detection: precision 0.91, recall 0.73 on seq 08 (cross-validated pooled F1 0.79); the
> missing recall is a clustering problem, not a classifier problem.
> Completion: there's no ground truth on real data, so an independent judge decides
> whether each output looks like a car. It accepts 25% of raw partials, 79% of completions,
> and 81% of dense reference cars.

Every slide should support one sentence of that paragraph.

## Deck v2 — friend's structure (decided 2026-10-14; supersedes the two tables below)

Mirrors the peer FPT deck (`docs/LVTN.pptx`): I–IV sections, sub-dividers inside III, hierarchical
numbering ("2.1 …"), THE END, then backups. Their talk was also 15 min; support slides sit in
place for the 30-min Q&A instead of all at the end. ★ = talk path (timed), ○ = support (skip in
the talk, jump to in Q&A). D-numbers = diagram inventory below. Sources = thesis labels.

| # | Slide | Talk path / source (from speaker notes) |
|---|---|---|
| 1 | Cover | ★ TALK 0:15 — title, name, supervisor. |
| 2 | TABLE OF CONTENTS | ★ TALK 0:10 — four parts. |
| 3 | ▌I. INTRODUCTION | divider: click through |
| 4 | 1.1 BACKGROUND | ★ TALK 1:00 — source: D1; fig:bg-lidar-scan (b); say the 3 point-cloud properties aloud (backup slide) |
| 5 | 1.2 CURRENT APPROACHES | ★ TALK 1:00 — source: ch1 §1.1; ch2 §2.2.2–2.2.3 |
| 6 | 1.3 LITERATURE REVIEW | ○ SUPPORT (skip in talk) — source: ch2 §2.2; tab:positioning |
| 7 | 1.4 RESEARCH GAP | ★ TALK 0:40 — source: ch1 §1.1–1.2 |
| 8 | 1.5 OBJECTIVES & CONTRIBUTIONS | ★ TALK 0:45 — source: ch1 §Objectives, §Contributions (Objective-2 weak spot: project_state.md) |
| 9 | ▌II. PROPOSED METHODOLOGY | divider: click through |
| 10 | 2.1 SYSTEM ARCHITECTURE | ★ TALK 1:00 — source: fig:architecture |
| 11 | 2.2 DETECTION PIPELINE | ○ SUPPORT (skip in talk) — source: ch3 §detection pipeline; fig:pipeline |
| 12 | 2.2.1 PREPROCESSING AND GROUND REMOVAL | ★ TALK 0:45 — source: ch3 §preprocessing, §RANSAC |
| 13 | 2.2.2 CLUSTERING (HDBSCAN) | ★ TALK 0:45 — source: fig:clustering |
| 14 | 2.2.3 GEOMETRIC FILTER | ○ SUPPORT (skip in talk) — source: fig:geo-filter |
| 15 | 2.2.4 CAR / NOT-CAR CLASSIFIER | ★ TALK 1:00 — source: fig:classifier |
| 16 | 2.2.5 TRACKING AND TRACK FILTER | ★ TALK 0:30 — source: fig:tracker |
| 17 | 2.3 OBJECT COMPLETION | ○ SUPPORT (skip in talk) — source: ch3 §object completion; fig:completion-pipeline |
| 18 | 2.3.1 COMPLETION NETWORK (PCN) | ★ TALK 1:00 — source: fig:pcn |
| 19 | 2.3.2 SYNTHETIC TRAINING PARTIALS | ○ SUPPORT (skip in talk) — source: fig:kitti-like-partial |
| 20 | 2.3.3 NORMALISATION, L-SHAPE GATE AND LENGTH | ○ SUPPORT (skip in talk) — source: fig:lshape-gate, tab:completion-config |
| 21 | ▌III. EXPERIMENT EVALUATION · 3.1 DATASET AND PROTOCOL | divider: click through |
| 22 | 3.1.1 DATASETS | ○ SUPPORT (skip in talk) — source: tab:dataset-roles, tab:dataset, tab:sensor-spec |
| 23 | 3.1.2 TARGET ELIGIBILITY AND GROUND TRUTH | ○ SUPPORT (skip in talk) — source: ch4 §eligibility (recall-denominator weak spot) |
| 24 | ▌3.2 EXPERIMENT SETUP | divider: click through |
| 25 | 3.2.1 BASELINES | ○ SUPPORT (skip in talk) — source: ch4 §baselines |
| 26 | 3.2.2 METRICS | ★ TALK 0:30 — source: fig:bg-iou; ch4 §metrics; ch4 §plausibility |
| 27 | 3.2.3 DATA SPLIT: LEAVE-ONE-SEQUENCE-OUT | ○ SUPPORT (skip in talk) — source: ch4 §splits; tab:loso |
| 28 | ▌3.3 DETECTION RESULTS | divider: click through |
| 29 | 3.3.1 OVERALL DETECTION RESULTS | ★ TALK 1:00 — source: D9; tab:headline, tab:loso, tab:ablation-stage |
| 30 | 3.3.2 WHAT LIMITS RECALL | ★ TALK 1:00 — source: tab:split, fig:failure-zooms |
| 31 | 3.3.3 ACCURACY VS RANGE | ○ SUPPORT (skip in talk) — source: tab:distance |
| 32 | 3.3.4 DETECTION RESULT ANALYSIS | ○ SUPPORT (skip in talk) — source: ch4 §result analysis |
| 33 | ▌3.4 COMPLETION RESULTS | divider: click through |
| 34 | 3.4.1 JUDGING COMPLETION WITHOUT GROUND TRUTH | ★ TALK 1:30 — source: D12; tab:plausibility, fig:plausibility |
| 35 | 3.4.2 COMPLETION EXAMPLES: RAW VS COMPLETED | ○ SUPPORT (skip in talk) — source: fig:completion-shape, fig:plausibility |
| 36 | 3.4.3 SYNTHETIC FIDELITY | ○ SUPPORT (skip in talk) — source: ch4 §synthetic fidelity |
| 37 | 3.4.4 PLAUSIBILITY ROBUSTNESS | ○ SUPPORT (skip in talk) — source: tab:plausibility-seq; ch4 §plausibility |
| 38 | 3.4.5 COMPLETION RESULT ANALYSIS | ○ SUPPORT (skip in talk) — source: ch4 §scope of the completion claim; ch5 |
| 39 | ▌3.5 RUNTIME EVALUATION | divider: click through |
| 40 | 3.5.1 RUNTIME BY STAGE | ○ SUPPORT (skip in talk) — source: tab:runtime |
| 41 | 3.5.2 RUNTIME OPTIMISATION | ○ SUPPORT (skip in talk) — source: ch4 §runtime evaluation (921.3 → 623.5 ms) |
| 42 | ▌3.6 ABLATION STUDY | divider: click through |
| 43 | 3.6.1 STAGE ABLATION | ○ SUPPORT (skip in talk) — source: tab:ablation-stage |
| 44 | 3.6.2 CLUSTERING ALTERNATIVES AND RESOLUTION | ○ SUPPORT (skip in talk) — source: tab:clustering, tab:splitrate |
| 45 | 3.6.3 IOU-THRESHOLD SENSITIVITY | ○ SUPPORT (skip in talk) — source: tab:iou-sensitivity |
| 46 | 3.6.4 SYNTHETIC PRETRAINING | ○ SUPPORT (skip in talk) — source: tab:cross-domain |
| 47 | 3.6.5 COMPLETION ABLATIONS | ○ SUPPORT (skip in talk) — source: ch4 §completion ablations |
| 48 | ▌IV. CONCLUSION & FUTURE WORK | divider: click through |
| 49 | 4.1 RESEARCH SUMMARY | ★ TALK 0:45 — source: ch5 §principal outcomes, §diagnostic findings |
| 50 | 4.2 LIMITATIONS | ★ TALK 0:30 — source: ch5 §limitations |
| 51 | 4.3 DISCUSSION & FUTURE WORK | ★ TALK 0:30 — source: ch5 §future work |
| 52 | THE END. · THANKS FOR ATTENDING AND LISTENING. | ★ END — leave up during Q&A; jump to slides by number. |
| 53 | LIDAR POINT CLOUD PROPERTIES | ○ SUPPORT (skip in talk) — source: ch2 §LiDAR point cloud; tab:sensor-spec; S1 notes A1 |
| 54 | REPAIR STRATEGIES AND WHY THEY FAIL | ○ SUPPORT (skip in talk) — source: backup |
| 55 | THE NORMALISATION BUG | ○ SUPPORT (skip in talk) — source: backup |
| 56 | IS THE JUDGE INDEPENDENT? | ○ SUPPORT (skip in talk) — source: backup |
| 57 | FIDELITY METRICS: UCD AND UHD | ○ SUPPORT (skip in talk) — source: backup |
| 58 | WORKED EXAMPLE: IOU MATCHING | ○ SUPPORT (skip in talk) — source: backup |
| 59 | PIPELINE CONFIGURATION | ○ SUPPORT (skip in talk) — source: backup |

Numbering: section-prefixed hierarchy (user decision 2026-10-14, replacing the friend's per-section
restart, which repeated "2.1" in II and III): I → 1.x, II → 2.x / 2.x.y, III → 3.x sub-dividers with
3.x.y slides (results split into 3.3 Detection and 3.4 Completion), IV → 4.x. Max depth 3.

**Built by code** (2026-10-14, after Canva AI edits failed at this scale):
`docs/defense/scripts/build_deck.py` → `docs/defense/deck_v2.pptx` (59 slides, ★/○ + thesis
source in each slide's speaker notes). Canva dropped 2026-10-14: the deck is finished locally. Assets in `docs/defense/assets/`.
To change structure or text, edit the script and rebuild rather than re-prompting Canva AI.

The D3 "you are here" pipeline strip was **dropped** 2026-10-14 (user decision); orientation comes
from the footer section name + hierarchical numbering. D3 appears only on II.1 and THE END.

Talk path ≈ 14:50 incl. dividers, so **no slack**: the first mock defense must check it; first cuts
if over are slide 25 (fold metrics into 28) and merging 45+46.

## Main deck v1 (~14 min, leaves 1 min slack) — SUPERSEDED by Deck v2 above

| # | Slide | Time | One message | Visual | Tool |
|---|---|---|---|---|---|
| 1 | Title | 0:20 | — | Title, name, advisor, one striking point-cloud render | Canva |
| 2 | The problem: one-sided view | 1:00 | The sensor only hits the near side of a car, so half of every car is missing | **D1** top-down sketch: sensor, rays, car with near side lit; + real scan screenshot | Canva (animated) |
| 3 | Two ways to solve it | 1:00 | End-to-end networks are accurate but need dense labels and act as black boxes; we take the modular route | **D2** two parallel pipelines: "one big black box" vs "chain of glass boxes" | Canva |
| 4 | Objectives & contributions | 0:45 | 3 objectives → 2 contributions | Clean 3→2 mapping graphic, no paragraphs | Canva |
| 5 | Framework overview | 1:15 | Raw scan → ground removal → clustering → filters → classifier → tracker → completion | **D3** master pipeline strip (reused later as a "you are here" bar) | draw.io → SVG, or Canva |
| 6 | Finding object candidates | 1:30 | Remove the ground plane (RANSAC), then group the remaining points into blobs (HDBSCAN) | **D4** RANSAC: plane fit to points, inliers vs outliers; **D5** before/after real frame coloured by cluster | Canva + screenshots |
| 7 | Keeping only cars | 1:30 | Size/shape rules drop obvious junk; a small PointNet says car / not-car; its training labels are mined automatically from SemanticKITTI | **D6** mining flow (labelled scans → clusters → car/not-car); **D7** simplified PointNet (points → shared MLP → max-pool → yes/no) | draw.io |
| 8 | Tracking removes flicker | 0:45 | A car seen across many frames is kept; one-frame junk is dropped | **D8** timeline frames t1…t5, a stable track vs a blinking false positive | Canva (animated) |
| 9 | Detection results | 1:30 | Each stage has a measurable job: FP 108,831 → 4,303 → 2,513; final P 0.911 / R 0.732 / F1 0.812; cross-validated (LOSO) F1 0.793 | **D9** funnel of false positives per stage + small headline table | Canva chart |
| 10 | Why recall stops at ~0.73 | 1:00 | HDBSCAN splits 33.7% of cars into fragments; the classifier isn't the limit | **D10** real example: one car split into 2 colours + schematic | Screenshot + Canva overlay |
| 11 | Completing the car | 1:30 | PCN is trained only on synthetic ShapeNet cars, using simulated LiDAR-style partial views, and must work on real scans it never saw | **D11** ShapeNet mesh → simulated partial → PCN (encoder → global code → decoder) → full car; canonical frame note | draw.io |
| 12 | Judging completion without ground truth | 1:30 | Distance metrics reward doing nothing; instead an independent judge (PointNet++ on ModelNet40) accepts 0.253 raw → **0.785** completed, vs a 0.807 ceiling | **D12** three groups of shapes → judge → bar chart | draw.io + Canva chart |
| 13 | Conclusions, limitations, future work | 1:00 | 2 contributions + 2 findings; car-only, KITTI-only, clustering ceiling | 3 columns of short bullets | Canva |
| 14 | Thank you / Q&A | — | — | Master pipeline (D3) again as a visual anchor for questions | Canva |

## Backup slides v1 — SUPERSEDED (now in-place ○ slides + backups in Deck v2)

Grouped by the question each one answers.

- **"How do you count a detection as correct?"** IoU ≥ 0.3, greedy 1-to-1 matching, supported-vehicle eligibility, micro-averaging. Include a worked example.
- **"Is seq 08 cherry-picked?"** Per-fold LOSO table and the pooled results.
- **"How does it do at range?"** Accuracy-versus-distance table.
- **"Why HDBSCAN, not DBSCAN or k-means?"** Density diagram plus the clustering comparison table.
- **"Did you try to fix fragmentation?"** The repair strategies and why they fail (they merge neighbouring cars).
- **"Why not PointPillars / a deep detector?"** Positioning against related work, and the annotation cost.
- **"Why is synthetic pretraining redundant?"** Stage A vs Stage B classifier comparison.
- **Classifier details:** input size, architecture, training data counts.
- **L-shape gate and length estimation:** which clusters are allowed into completion.
- **The normalisation bug:** the train/inference frame mismatch and the lesson learned.
- **Fidelity metrics (UCD/UHD):** what they reward and why they penalise added surface.
- **Plausibility robustness:** results per sequence, moving vs static cars, the 180° flip test, McNemar statistics.
- **Is the judge independent?** ModelNet40 vs ShapeNet, the pre-registered 0.88 gate, 0.9214 accuracy.
- **Runtime.**

## Diagram inventory (what you'll draw)

The order follows the study sessions, not the deck order.

**Split (decided 2026-10-14):** the user hand-draws only the *technical* diagrams (the point is
understanding the mechanism): D1, D3, D4, D6, D7, D8, D11, D12-schematic. Presentational slides
(D2, title, objectives, closing) go to Canva AI. D5/D10 are real screenshots and D9/D12-bars are
charts from typed-in thesis numbers; neither is AI-generated. Every number on an AI-made slide
must be checked against the thesis.

| ID | Diagram | Difficulty | Tool | Study session |
|---|---|---|---|---|
| D1 | Occlusion / one-sided view | easy | Canva | S1 |
| D2 | End-to-end vs modular | easy | Canva | S1 |
| D3 | Master pipeline | medium | draw.io | S1 |
| D4 | RANSAC plane fit | easy | Canva | S2 |
| D5 | Clusters before/after | screenshot | Open3D + Canva | S2 |
| D6 | Auto-mined labels flow | medium | draw.io | S3 |
| D7 | PointNet simplified | medium | draw.io | S3 |
| D8 | Tracking timeline | easy | Canva | S4 |
| D9 | False-positive funnel | easy | Canva chart | S4 |
| D10 | Fragmentation example | screenshot | Open3D + Canva | S4 |
| D11 | PCN training & inference | hard | draw.io | S5 |
| D12 | Plausibility evaluation | medium | draw.io + chart | S6 |

## Schedule (13 days)

| Date | Work |
|---|---|
| Oct 12–13 | Sign off this outline. Numbers spot-check against `results/postfix_normalfix_2026-09-14/EDIT_MAP.md`. Canva export test. |
| Oct 14 | **S1** Problem, data (SemanticKITTI, ShapeNet), framework → D1–D3 + quiz |
| Oct 15 | **S2** Preprocessing, RANSAC, voxels, HDBSCAN → D4–D5 + quiz |
| Oct 16 | **S3** Geometric filter, PointNet classifier, auto-mining, Stage A/B → D6–D7 + quiz |
| Oct 17 | **S4** Tracking, evaluation protocol, results, ablation, recall analysis → D8–D10 + quiz |
| Oct 18 | **S5** PCN, synthetic partials, normalisation, L-shape gate → D11 + quiz |
| Oct 19 | **S6** Completion evaluation: synthetic CD/F, plausibility, fidelity → D12 + quiz |
| Oct 20–21 | Assemble the Canva deck, write Vietnamese speaker notes, time each slide |
| Oct 22 | Question bank (`docs/defense/qa_bank.md`) + backup slides |
| Oct 23 | Mock defense #1 (timed talk + 30 min hostile Q&A) |
| Oct 24 | Mock #2, final fixes, export PDF backup |
| Oct 25 | Defense |

## Open items

- [x] Outline sign-off (2026-10-12; user agreed incl. eval details + normalisation bug → backup)
- [x] Numbers spot-check — passed 2026-10-12 (see top)
- [ ] Vietnamese ↔ English glossary of key terms (built during S1–S6)
- [ ] Canva → PPTX/PDF export fidelity test (fonts, animations)
