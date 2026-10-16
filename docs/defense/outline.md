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

Mirrors the peer FPT deck (`docs/defense/reference/LVTN.pptx`): I–IV sections, sub-dividers inside III, hierarchical
numbering ("2.1 …"), THE END, then backups. Their talk was also 15 min; support slides sit in
place for the 30-min Q&A instead of all at the end. ★ = talk path (timed), ○ = support (skip in
the talk, jump to in Q&A). D-numbers = diagram inventory below. Sources = thesis labels.

| # | Slide | Talk path / source (from speaker notes) |
|---|---|---|
| 1 | Cover | ★ TALK 0:15 — title, name, supervisor. |
| 2 | TABLE OF CONTENTS | ★ TALK 0:10 — four parts. |
| 3 | ▌I. INTRODUCTION | divider: click through |
| 4 | 1.1 BACKGROUND | ★ TALK 1:00 — source: D1; fig:bg-lidar-scan (b); say the 3 point-cloud properties aloud (backup slide) |
| 5 | 1.2 CURRENT APPROACHES | ★ TALK 1:00 — source: D2; ch1 §1.1 (examples); ch2 §2.2.2–2.2.3. If asked how we compare: different metrics (their all-class mIoU vs our car-only point F1), not one scale; see 1.3 / tab:positioning |
| 6 | 1.3 LITERATURE REVIEW | ○ SUPPORT (skip in talk) — source: ch2 §2.2 (bg-deep, bg-classical, bg-completion); tab:positioning for the per-method table. No accuracy column on purpose: their mIoU / PQ over all classes and our car-only point F1 are different measurements (ch2 §positioning). |
| 7 | 1.4 RESEARCH GAP | ★ TALK 0:40 — source: ch1 §1.1–1.2 |
| 8 | 1.5 OBJECTIVES & CONTRIBUTIONS | ★ TALK 0:45 — source: ch1 §Objectives, §Contributions |
| 9 | ▌II. PROPOSED METHODOLOGY | divider: click through |
| 10 | 2.1 SYSTEM ARCHITECTURE | ★ TALK 1:00 — source: D3; fig:architecture, fig:pipeline, fig:completion-pipeline |
| 11 | 2.2 DETECTION PIPELINE | ○ SUPPORT (skip in talk) — source: ch3 §detection pipeline; fig:pipeline (crop: scripts/crop_thesis_figure.py 3.2 pipeline) |
| 12 | 2.2.1 PREPROCESSING AND GROUND REMOVAL | ★ TALK 0:45 — source: D4 (seq 08 frame 2500, one-frame illustration: counts are NOT thesis results); ch2 §RANSAC; ch3 §preprocessing, §RANSAC. |
| 13 | 2.2.2 CLUSTERING (HDBSCAN) | ★ TALK 0:45 — source: D5 = same seq 08 frame 2500 as D4 (scripts/make_d5_clusters.py, pipeline's own preprocess, remove_ground, cluster_objects); one-frame illustration: the 205 clusters are NOT a thesis result. Concept: fig:clustering; fig:bg-pipeline-stages. |
| 14 | 2.2.3 GEOMETRIC FILTER | ○ SUPPORT (skip in talk) — source: fig:geo-filter (crop: scripts/crop_thesis_figure.py 3.4 geo_filter); gates = ch3 §Geometric Filtering; P 0.199 / R 0.773 = tab:ablation-stage geometric-only row. If asked which gates cost cars: 27,718 rejected clusters overlap a GT car, 73.9% by the minimum-volume and 12.7% by the minimum-points gate, i.e. small splinters of already-split cars (the thesis states this as an inference). |
| 15 | 2.2.4 CAR / NOT-CAR CLASSIFIER | ★ TALK 1:00 — source: D6+D7 merged. Top lane (training) = ch3 §training data (mined, not hand labelled); bottom lane (inference) = fig:classifier, simplified. Story: the top lane is the answer key, looked up from SemanticKITTI labels; the classifier learns to give the same answer from the points alone, so on a new scan, with no labels, only the bottom lane runs. Say 'zero additional manual annotation', not 'annotation-free'. Counts: 420,333 training clusters (seq 00-07, 09, 10), 130,394 validation (seq 08), about 21% car / 79% not-car. |
| 16 | 2.2.5 TRACKING AND TRACK FILTER | ★ TALK 0:30 — source: fig:tracker |
| 17 | 2.3 COMPLETION PIPELINE | ○ SUPPORT (skip in talk) — source: ch3 §object completion; fig:completion-pipeline |
| 18 | 2.3.1 COMPLETION NETWORK (PCN) | ★ TALK 1:00 — source: fig:pcn |
| 19 | 2.3.2 SYNTHETIC TRAINING PARTIALS | ○ SUPPORT (skip in talk) — source: fig:kitti-like-partial |
| 20 | 2.3.3 L-SHAPE GATE, NORMALISATION AND LENGTH | ○ SUPPORT (skip in talk) — source: fig:lshape-gate, tab:completion-config |
| 21 | ▌III. EXPERIMENTAL EVALUATION · 3.1 DATASET AND PROTOCOL | divider: click through |
| 22 | 3.1.1 DATASETS | ○ SUPPORT (skip in talk) — source: tab:dataset-roles, tab:sensor-spec, tab:dataset, verbatim. Seq 08 is the classifier's validation sequence, not a test set; LOSO (3.2.3) is the leakage control. |
| 23 | 3.1.2 TARGET ELIGIBILITY AND GROUND TRUTH | ○ SUPPORT (skip in talk) — source: ch4 §eligibility — ≥10 surviving points; 25.5% of annotated cars excluded; recall vs ALL annotated cars ≈ 0.55 (ch4, ch5) vs 0.732 eligible |
| 24 | ▌3.2 EXPERIMENTAL SETUP | divider: click through |
| 25 | 3.2.1 BASELINES | ○ SUPPORT (skip in talk) — source: ch4 §baselines; tab:positioning |
| 26 | 3.2.2 METRICS | ★ TALK 0:30 — source: fig:bg-iou; ch4 §metrics; ch4 §plausibility |
| 27 | 3.2.3 DATA SPLIT: LEAVE-ONE-SEQUENCE-OUT | ○ SUPPORT (skip in talk) — source: ch4 §splits; tab:loso |
| 28 | ▌3.3 DETECTION RESULTS | divider: click through |
| 29 | 3.3.1 OVERALL DETECTION RESULTS | ★ TALK 1:00 — source: D9 = tab:ablation-stage, numbers verbatim (scripts/make_d9_funnel.py); table = tab:headline + tab:loso pooled row. Story, left to right: the geometric filter keeps most cars (recall 0.773) but 4 of 5 detections are false positives (precision 0.199); the classifier removes about 104,000 false positives (precision 0.846) and pays some recall (0.676); the track filter wins back 1,968 true positives and drops more false positives (0.911 / 0.732 / F1 0.812). Counts are per-frame matches over the sequence, not distinct cars. Table row 2 = tab:loso pooled (each fold's classifier never saw its test sequence). If asked whether we selected and report on seq 08: the seq-08 fold of LOSO (classifier blind to seq 08) gets 0.888 / 0.736 / 0.805, so selection bought about 2 precision points and no recall (tab:headline). |
| 30 | 3.3.2 DETECTIONS ON A REAL FRAME | ○ SUPPORT (skip in talk) — source: fig:bev-detections |
| 31 | 3.3.3 WHAT LIMITS RECALL | ★ TALK 1:00 — source: tab:split, fig:failure-zooms |
| 32 | 3.3.4 ACCURACY VS RANGE | ○ SUPPORT (skip in talk) — source: tab:distance |
| 33 | 3.3.5 DETECTION RESULT ANALYSIS | ○ SUPPORT (skip in talk) — source: ch4 §result analysis |
| 34 | ▌3.4 COMPLETION RESULTS | divider: click through |
| 35 | 3.4.1 JUDGING COMPLETION WITHOUT GROUND TRUTH | ★ TALK 1:30 — source: D12; tab:plausibility, fig:plausibility |
| 36 | 3.4.2 COMPLETION EXAMPLES: RAW VS COMPLETED | ○ SUPPORT (skip in talk) — source: fig:completion-shape, fig:plausibility |
| 37 | 3.4.3 SYNTHETIC FIDELITY | ○ SUPPORT (skip in talk) — source: ch4 §synthetic fidelity |
| 38 | 3.4.4 PLAUSIBILITY ROBUSTNESS | ○ SUPPORT (skip in talk) — source: tab:plausibility-seq; ch4 §plausibility |
| 39 | 3.4.5 COMPLETION RESULT ANALYSIS | ○ SUPPORT (skip in talk) — source: ch4 §scope of the completion claim; ch5 |
| 40 | ▌3.5 RUNTIME EVALUATION | divider: click through |
| 41 | 3.5.1 RUNTIME BY STAGE | ○ SUPPORT (skip in talk) — source: tab:runtime |
| 42 | 3.5.2 RUNTIME OPTIMISATION | ○ SUPPORT (skip in talk) — source: ch4 §runtime evaluation (921.3 → 623.5 ms) |
| 43 | ▌3.6 ABLATION STUDY | divider: click through |
| 44 | 3.6.1 STAGE ABLATION | ○ SUPPORT (skip in talk) — source: tab:ablation-stage, verbatim. Same numbers as the slide 29 chart, plus mean IoU and the raw counts. Counts are per-frame matches over the full sequence, not distinct cars. |
| 45 | 3.6.2 CLUSTERING ALTERNATIVES AND RESOLUTION | ○ SUPPORT (skip in talk) — source: tab:clustering, tab:splitrate. Mean IoU column of tab:clustering left out. If asked 'why not an even coarser grid, e.g. 0.20 m?': merges grow as the grid coarsens (56 → 284), so we expect more fused cars; 0.20 m was NOT tested, say 'expect', not 'found'. If asked 'did you tune DBSCAN's radius?': yes, an epsilon sweep; the best fixed radius (Euclidean, 0.4 m) reaches F1 0.800 on seq 00, still below HDBSCAN's 0.835 (ch4 §clustering). |
| 46 | 3.6.3 IOU-THRESHOLD SENSITIVITY | ○ SUPPORT (skip in talk) — source: tab:iou-sensitivity |
| 47 | 3.6.4 SYNTHETIC PRETRAINING | ○ SUPPORT (skip in talk) — source: ch4 sec:ablation-stagea (macro-F1 0.9285 vs 0.9225; pipeline F1 identical to three decimals); tab:cross-domain. Don't mix the metrics: 0.9285 is macro-F1 (mean of car and not-car F1), 0.885 is car-class F1. Why keep it in the thesis: it changes the recipe, so it is reported as a negative result. |
| 48 | 3.6.5 COMPLETION ABLATIONS | ○ SUPPORT (skip in talk) — source: ch4 §completion ablations (sec:abl-completion) |
| 49 | ▌IV. CONCLUSION & FUTURE WORK | divider: click through |
| 50 | 4.1 RESEARCH SUMMARY | ★ TALK 0:45 — source: ch5 §principal outcomes, §diagnostic findings, Final Remarks |
| 51 | 4.2 LIMITATIONS | ★ TALK 0:30 — source: ch5 §limitations (disc-architectural, disc-completion-scope, disc-eval-validity) |
| 52 | 4.3 DISCUSSION & FUTURE WORK | ★ TALK 0:30 — source: ch5 §future work |
| 53 | THANK YOU. · QUESTIONS ARE WELCOME. | ★ END — leave up during Q&A; jump to slides by number. |
| 54 | LIDAR POINT CLOUD PROPERTIES | ○ SUPPORT (skip in talk) — source: ch2 §LiDAR point cloud; tab:sensor-spec; S1 notes A1 |
| 55 | HOW HDBSCAN WORKS | ○ SUPPORT (skip in talk) — source: backup for 'how does HDBSCAN work?'. TOY 2D example (scripts/make_hdbscan_explainer.py), real hdbscan library with our settings (min_samples 5, min_cluster_size 10); NOT a thesis result. Say: (1) each point measures how crowded its neighbourhood is: the distance to its 5th nearest point (itself included); (2) HDBSCAN then looks at every radius at once: at a small radius only the dense near car forms, at a medium one both cars form, at a large one they fuse; (3) shrinking the radius, it keeps a group whole if it holds together longer than its pieces would. Each car is judged only against its own pieces, never against the other car, so the dense near car and the sparse far car both survive; a fixed-radius DBSCAN would have to pick one column. Toy numbers: near car 335.7 vs its pieces 19.2 -> whole; far car 24.1, no pieces -> kept (S2 notes A4). Precision if pressed: the 'radius' is the mutual reachability distance max(core a, core b, distance a-b), and 'longest' is stability, lifetime measured in 1/radius and summed over points; groups under 10 points are noise. ch2 §density-based clustering; S2 notes A4. |
| 56 | REPAIR STRATEGIES AND WHY THEY FAIL | ○ SUPPORT (skip in talk) — source: backup |
| 57 | THE NORMALISATION BUG | ○ SUPPORT (skip in talk) — source: backup |
| 58 | IS THE JUDGE INDEPENDENT? | ○ SUPPORT (skip in talk) — source: backup |
| 59 | FIDELITY METRICS: UCD AND UHD | ○ SUPPORT (skip in talk) — source: backup |
| 60 | WORKED EXAMPLE: IOU MATCHING | ○ SUPPORT (skip in talk) — source: backup |
| 61 | PIPELINE CONFIGURATION | ○ SUPPORT (skip in talk) — source: backup |

Numbering: section-prefixed hierarchy (user decision 2026-10-14, replacing the friend's per-section
restart, which repeated "2.1" in II and III): I → 1.x, II → 2.x / 2.x.y, III → 3.x sub-dividers with
3.x.y slides (results split into 3.3 Detection and 3.4 Completion), IV → 4.x. Max depth 3.

**Built by code** (2026-10-14, after Canva AI edits failed at this scale):
`docs/defense/scripts/build_deck.py` → `docs/defense/deck_v2.pptx` (61 slides, ★/○ + thesis
source in each slide's speaker notes). Canva dropped 2026-10-14: the deck is finished locally. Assets in `docs/defense/assets/`.
To change structure or text, edit the script and rebuild. The table above is generated from the
built deck's speaker notes.

**Thesis ↔ reference ↔ deck comparison (2026-10-16, user-approved):** structure kept; fixes:
4.2 Limitations rewritten to ch5's limitations (dense scenes only / clustering ceiling /
plausibility-only completion evidence / offline); 4.1 Summary = the four Final-Remarks claims with
numbers; 3.2.1 Baselines states why there is no head-to-head with supervised detectors
(tab:positioning); 3.6.5 key message = "input quality and normalisation, not the network";
new ○ 3.3.2 "Detections on a Real Frame" (fig:bev-detections), our analogue of the reference's
"Baselines vs Ours — Case N" walkthroughs; 2.3 renamed "Completion Pipeline" to parallel 2.2.
Kept as-is on purpose: auto-mining stays half of 2.2.4 (no talk-time slack for its own slide);
repair strategies stay a backup.

The D3 "you are here" pipeline strip was **dropped** 2026-10-14 (user decision); orientation comes
from the footer section name + hierarchical numbering. D3 appears only on 2.1 and THE END.

Talk path ≈ 14:50 incl. dividers, so **no slack**: the first mock defense must check it; first cut
if over is folding 3.2.2 Metrics (★ 0:30) into 3.3.1.

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

**Split (revised 2026-10-16; Canva dropped):** three owners.

1. **User draws in draw.io** — the *technical* diagrams, so the mechanism is understood, not
   just shown. Use the thesis TikZ figure as the reference and simplify it for a non-specialist
   committee. **Workflow (from D3, 2026-10-16):** Claude generates a scaffold `.drawio` in
   `assets/` (shapes, arrows, colours, "?" placeholders); the user fills in the content and
   exports `assets/<name>.drawio.png` (File → Export as → PNG: Zoom 200%, Transparent off,
   Appearance **Light**, "Include a copy of my diagram" on). That one file is both the deck image
   and the editable source; `build_deck.py` uses it directly.
2. **Claude draws** — *presentational* visuals: SVG (like D1) or native PPTX charts/tables in
   `build_deck.py`, with every number typed from the thesis.
3. **Thesis figures reused as-is** — real-data plots and ○-slide figures are cropped from the
   thesis (`docs/writing/thesis/figures/*.png`, or TikZ figures cropped from `main.pdf`). No
   Open3D screenshots or re-runs.

| ID | Diagram | Owner | Tool / source | Slide | Session |
|---|---|---|---|---|---|
| D1 | Occlusion / one-sided view | Claude ✓ done | SVG (`figures/d1_occlusion.svg`) | 1.1 | S1 |
| D2 | End-to-end vs modular | Claude ✓ done | SVG (`figures/d2_end_to_end_vs_modular.svg`) | 1.2 | S1 |
| D3 | Master pipeline | **User** ✓ done | `assets/d3_master_pipeline.drawio.png`; ref `fig:architecture` | 2.1, THE END | S1 |
| D4 | RANSAC on a real scan | Claude ✓ done (data figure, replaced the draw.io scaffold 2026-10-16) | `assets/d4_ransac_real.png` from `scripts/make_d4_ransac.py` (seq 08 frame 2500) | 2.2.1 | S2 |
| D5 | Clusters before/after | Claude ✓ done | `assets/d5_ground_then_clusters.png` from `scripts/make_d5_clusters.py` (seq 08 frame 2500, same as D4; regenerated 2026-10-16, replaced the thesis (b)+(c) crop) | 2.2.2 | S2 |
| D6 | Auto-mined labels flow | **User** ✓ | filled `assets/d6_mining_flow.drawio`; merged into D6+D7 `assets/d67_classifier_training.drawio` (training lane) 2026-10-16; PNG exported, on slide 15 | 2.2.4 | S3 |
| D7 | PointNet simplified | **User** ✓ | filled `assets/d7_classifier.drawio`; merged into D6+D7 (inference lane, 'trains' arrow from the training lane) | 2.2.4 | S3 |
| D8 | Tracking timeline | **User** | draw.io; ref `fig:tracker` | 2.2.5 | S4 |
| D9 | False-positive funnel | Claude ✓ | `assets/d9_stage_funnel.png` from `scripts/make_d9_funnel.py` (tab:ablation-stage verbatim) + native tab:headline/tab:loso table, 2026-10-16 | 3.3.1 | S4 |
| D10 | Fragmentation example | Thesis | `seq08_failure_zooms.png`, frame-250 row | 3.3.3 | S4 |
| D11 | PCN training & inference | **User** | draw.io; ref `fig:pcn` | 2.3.1 | S5 |
| D12a | Judge schematic | **User** | draw.io; ref ch4 §plausibility | 3.4.1 | S6 |
| D12b | Acceptance bars | Claude | PPTX chart from `tab:plausibility` | 3.4.1 | S6 |

Also Claude: LOSO diagram (3.2.3), eligibility example (3.1.2), all results tables as native PPTX
tables (**3.6.2 done 2026-10-16**: tab:clustering + tab:splitrate), and the backup **How HDBSCAN
Works** (done 2026-10-16: toy figure from `scripts/make_hdbscan_explainer.py`, slide 55). Also thesis-as-is: `fig:pipeline` (2.2), `fig:geo-filter` (2.2.3),
`fig:completion-pipeline` (2.3), `fig:kitti-like-partial` (2.3.2), `fig:lshape-gate` (2.3.3),
`bg_iou.png` (3.2.2), `seq08_bev_detections.png` (3.3.2), `completion_shape_08.png` (3.4.2).

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
| Oct 20–21 | Fill the deck (diagrams, tables, figures), write Vietnamese speaker notes, time each slide |
| Oct 22 | Question bank (`docs/defense/qa_bank.md`) + backup slides |
| Oct 23 | Mock defense #1 (timed talk + 30 min hostile Q&A) |
| Oct 24 | Mock #2, final fixes, export PDF backup |
| Oct 25 | Defense |

## Open items

- [x] Outline sign-off (2026-10-12; user agreed incl. eval details + normalisation bug → backup)
- [x] Numbers spot-check — passed 2026-10-12 (see top)
- [ ] Vietnamese ↔ English glossary of key terms (built during S1–S6)
- [x] Thesis ↔ reference ↔ deck comparison (2026-10-16; fixes above)
- [ ] PDF export of the local `deck_v2.pptx` with fonts embedded (Canva test is obsolete)
