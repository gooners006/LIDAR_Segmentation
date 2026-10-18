# Project State

Last updated: 2026-10-18

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
  diagram inventory + schedule), `s1_*`–`s6_*.md` study notes, `qa_bank.md`, `talk_script.md`, `deck_v2.pptx`,
  `scripts/`, `assets/`, `figures/`, `qa_backup/` (private checks), `reference/`,
  `archive/canva/`.
- **Deck is built by code:** `docs/defense/scripts/build_deck.py` → `docs/defense/deck_v2.pptx`
  (61 slides, I–IV structure, ★ talk path ≈ 14:50 incl. dividers (★ timings sum to 14:35), ○ support slides for Q&A; thesis source in
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

**All defense material is built (2026-10-18).** S1–S6 study notes; deck filled with no
placeholders (main 4–53, backups 54–61); Q&A bank; talk script with VI speaker notes; PDF +
font-embedded exports. Committed (see `git log`). What's left needs the user:

1. **Timed rehearsal (Vietnamese, stopwatch).** Estimated length ≈ 10:00 (8:44–11:38) at an
   ASSUMED 3.5 syll/s (3.0–4.0), budget 14:35. Pre-registered rule: < 12:00 → add 0:30 (slide 38
   "the gain holds on every sequence" into slide 36); 12:00–14:50 → keep; > 14:50 → cut. Record the
   measured pace and re-run `scripts/talk_timing.py --vi-rate <pace>`.
2. **Mock defenses Oct 23–24:** say hook answers H1–H5 aloud (`qa_bank.md` Part 1), then random
   Part 3 questions; avoid Part 2 (do-not-open list).
3. **Optional:** S6 quiz (not run); S5 quiz was dropped by the user.
4. **Defense day:** take `deck_v2_embedded.pptx` (Segoe UI embedded) + `deck_v2.pdf` (backup) to
   the presentation laptop.

**Where things are (`docs/defense/`):**
- Study notes `s1_*`–`s6_*.md` (Part A concepts, Part B traps, Part C diagram answers, Part D
  glossary, Part E spoken EN/VI answers). S4: tracking + evaluation (D8 on 16, D10 on 31, LOSO
  grid on 27). S5: completion method, 2.3 slides 17–20 (D11 ★, D14, D15, gate + constants).
  S6: completion evaluation, slides 25, 36 (D12a + D12b chart), 37 (D16), 38, 39, 48.
- Backups 56–61: repair strategies, normalisation ablation on car C (naive path from git
  `896ff9c` vs production), judge independence, UCD/UHD toy, IoU worked example, pipeline config.
- `qa_bank.md`: Part 1 = five planted hooks H1–H5 (★ notes on slides 17, 26, 31, 36, 51 carry the
  line to say and stop on; committee is non-specialist); Part 2 = do-not-open list; Part 3 = 18
  generalist questions, EN/VI, jump slides, traps.
- `talk_script.md`: `### <n> <title> (<m:ss>)` + `EN:` / `VI:` for the 21 ★ slides; hooks ⚑.
  `build_deck.py add_vi_script` copies each VI line into that slide's notes and asserts the
  title matches the slide.
- Exports: `pwsh docs/defense/scripts/export_deck.ps1` after every rebuild → `deck_v2.pdf`
  (gitignored) + `deck_v2_embedded.pptx` (gitignored); never quits PowerPoint.

**Open study gaps (unverified — user skipped the retries):**
- S1: classifier labels are *mined from existing SemanticKITTI GT*; seq 08 is *validation* (why
  LOSO exists); no held-out set because seqs 11–21 are unlabelled; PCN trains on *rendered
  partials of synthetic CAD meshes*. Accuracy-comparison answer: S1 notes Part B #2.
- S2: Q3b (0.10 m grid trade: 3,700 un-split vs 230 newly merged) and Q5 (tab:clustering
  numbers, slide 45).
- S3: memorise 0.199 → 0.846 precision, 0.773 recall ceiling, purity 0.75, cross-domain 0.000
  both ways. Spoken EN/VI answers in S3 notes Part E.
- S4 (quiz 5/7): recall denominator (≈ 0.55 vs all annotated cars, not "three in four"), LOSO
  argument (0.736 vs 0.732, only the epoch differs), range table caveat (per frame, tracker off,
  0.671). Spoken answers: S4 notes Part E.

**Other carry-over:**
- Fonts: **Segoe UI everywhere** (deck, figures, diagrams; user decision 2026-10-18).
  `build_deck.py` `FONT = "Segoe UI"`.

Remaining schedule: Oct 19–22 rehearsal + buffer (deck, notes and Q&A done ahead of the
`outline.md` plan) · Oct 23–24 mock defenses · Oct 25 defense.

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
- **Judge input is not length-aligned (private check, 2026-10-18; NOT thesis numbers;
  `docs/defense/scripts/check_completion_heading.py`).** ch4 §plausibility says every cloud is
  harmonised into the judge's frame (width, up, length) and that harmonisation "fixes each cloud's
  up axis and length axis". Code (`scratchpad/plausibility/harmonize.py reorient`): partial and
  completed clouds only get a 180° flip about Z (fixes up); the saved clouds keep their scene
  heading. Measured over the 2822 scored completions: main axis within 20° of Z 37.6%, within 20° of
  X 38.2%, in between 24.2%. The ceiling clouds *are* de-rotated by `yaw_deg`. Inferred, not
  measured: raw vs completed is still a fair paired test (same heading on both sides), so
  0.253 → 0.785 stands; completed vs ceiling compares unaligned vs aligned clouds; the 180° heading
  flip check (Δ 0.009) only tests front/back for the ~38% that are Z-aligned. The thesis
  `fig:completion-shape` caption ("canonical car frame") has the same issue (19292, 8387 are
  not). **Settled 2026-10-18 (S6; measured, not in thesis):** aligned re-score
  (`docs/defense/scripts/check_judge_aligned.py` → `docs/defense/qa_backup/judge_aligned_rescore.json`;
  each pair turned so the completed cloud's main horizontal axis lies on the judge's Z; unaligned
  rerun reproduces 0.253 / 0.785 exactly): **raw 0.311 → completed 0.852**, McNemar 1,644 vs 117,
  per seq 0.79–0.98. Pre-registered reading (≥ 0.75 / ≤ 0.35) = headline holds; the thesis number is
  conservative, and aligned completed sits above the (aligned) ceiling 0.807. Judge is orientation
  sensitive (unaligned completed: 0.873 when Z-aligned vs 0.744 when X-aligned). Answer: S6 notes
  trap #10.
- **Synthetic CD 0.16 m / F 0.76 uses the training placement (S6, 2026-10-18; findings #26/#28,
  reproduced this session: 0.161 ± 0.022 / 0.754 on 30 val cars).** Each held-out scan is placed
  with the true centre and true radius; ch4 just says "a held-out synthetic set". With a GT-free
  placement it is worse (≈ 0.19 m with the width prior, before the length step, findings #35;
  0.50 m with the old naive centre, #28). Answer: S6 notes trap #6 (the test isolates the network;
  real placement is judged end to end by plausibility).
- **"96% of the gap" pairs the static-subset 0.770 with the pooled ratio (S6; arithmetic, not
  measured).** Pooled (0.785 − 0.253)/(0.807 − 0.253) = 96%; static subset (0.770 − 0.248)/
  (0.807 − 0.248) = 93%. Say "93 to 96%". S6 notes trap #11.
- **UCD formula vs reported value (S6, 2026-10-18; [code]).** ch2 eq. defines UCD with *squared* distances, but tab:fidelity's 0.090 m is the plain mean distance (`scratchpad/ucd_uhd_compare.py` `ucd_mean`; the squared version is stored separately as `ucd_sq`, m²). Units in the table (m) match the plain version. S6 notes trap #18.
- **"Real data fine-tuning" of PCN was never run (S6, 2026-10-18; inferred from absence: findings, checkpoints, scripts).** ch4 §abl-completion says sim-to-real remedies "(LiDAR noise augmentation and real data fine-tuning) failed" and "fine-tuning PCN on mined real pairs is ... reported as a negative result". The only fine-tuning runs (findings #16/#17/#19, 2026-05-28 to 06-02) used LiDAR-like *simulated* partials (noise aug, virtual Velodyne → `pcn_lidar_best.pth`, sparse-input); real-data fine-tuning appears only as a proposed option (#17) and future work (#26). No real-data PCN checkpoint exists; `session_history.md` mislabels #16/#17/#19 as "PCN real fine-tuning" (likely source). Deck fixed: slide 48 row = "KITTI-like training scans | fine-tune on noisier simulated scans"; slide 52 keeps real-data fine-tuning as future work (now consistent). Answer: qa_bank.md Part 2. If the user remembers an untracked real-data run, revisit.
- **Tracker gate vs fast cars (S4, 2026-10-18; speculative, from code, not measured).** 2 m
  centroid gate at 10 Hz = 72 km/h; above that every frame would start a one-frame track and the
  track filter (≥ 2 car votes) would drop it. May add to seq-01 (highway) recall 0.336, which the
  thesis attributes to range. Check: seq 01 with `--tracker-max-distance 4.0` vs default (~25 min);
  >0.03 recall gain = the gate costs highway cars. Answer in S4 notes trap #8.
- **Tracker uses dataset poses (S4; [code], not in thesis).** Centroids go to a world frame via
  `poses[i] @ Tr`; the thesis never mentions poses. Source of SemanticKITTI `poses.txt` (likely
  SLAM-estimated, since test seqs have them too) not verified. S4 notes trap #9.
- **PCN trained on cars + buses + motorcycles, thesis says cars (S5, 2026-10-18; [code] +
  checkpoint metadata, not in thesis).** `src/train_pcn.py` `TRAIN_CONFIG["categories"]` = car,
  bus, motorcycle since 2026-05-06; no CLI override; the config snapshot inside
  `pcn_kitti_best.pth` lists all three. Rebuilt dataset: train 3,835 = 2,813 cars (73%) + 752 bus
  + 270 motorcycle; val 955 (701 cars). Thesis: "ShapeNet car", `tab:dataset-roles` "~3,150 models
  (Car category)" (matches no count found). Synthetic CD 0.16 / F 0.76 likely over the mixed val
  set (inferred). Effect on car completion untested. Slide 22 keeps "~3,150 car models"
  (thesis verbatim; user decision 2026-10-18). Answer in S5 notes trap #9.
- **PCN training rolled every car about its length axis (S5, 2026-10-18; [code], measured).**
  `_augment_rotation` rotates about Z as if Z were up, but ShapeNet cars are Y-up / Z-length, so
  each training sample got a random roll. ch3 claims a fixed canonical frame (X width, Y up,
  Z length) in training: only the length axis was fixed. Inference inputs are upright (inside the
  training range). Quality cost untested. S5 notes trap #13.
- **Completion coverage and length fallback (S5; measured from `plausibility_gen`, not in
  thesis).** 5,496 car tracks over 11 seqs: 2,822 completed (51%), 2,022 rejected with ≥ 64 points
  (gate/degenerate, 37%), 652 too few points (12%). Length fallback (4.14 m): 119/518 (23%) on
  seq 08, 843/2,822 (30%) overall; ch3 promises "the exact fallback rate" but ch4 only says
  "non-trivial fraction".
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
