# Project State

Last updated: 2026-08-26

## RESEARCH FREEZE — declared 2026-08-21 (write-up phase)

Research is complete; the repository is now **evidence, not a lab**. Thesis
write-up (T14) is the only open task. Execution plan: `THESIS_PLAN.md` (Section 2
is the authoritative freeze table). Frozen items, all repo-verified 2026-08-21:

| Item | Frozen value | Source |
|---|---|---|
| Pipeline config | `PIPELINE_CONFIG` as committed: `voxel_before_denoise=True`, `ransac_iterations=300`, `cluster_voxel_size=0.10`, `clustering_method="hdbscan"` | `src/pipeline.py:10` |
| Checkpoints | `stage_b_scratch_best.pth` (classifier), `pcn_kitti_best.pth` (completion) | this doc, Checkpoints |
| Completion constants | per-car length estimate q90 (`COMPLETION_LENGTH_TRACK_QUANTILE=90`) + 0.12 m (`_OFFSET`), fallback 4.14 m below 5 frames; T13 flags (`decouple_radius`, `fill_z`) default OFF | `src/completion.py:206,242-243,303-304` |
| Eval protocol | point-level IoU ≥ 0.3 greedy 1-to-1, supported-vehicles, track filter ON, micro-averaged | `src/evaluate.py` |
| Authoritative sequences | **seq 08 = headline** (4,071 frames); seq 00 = tuning/replication (dual role, in classifier train split); seq 05 = fallback-trigger record only | Findings #18, #42; T8 |
| Closed experiments | recall repairs (#21/#24), Step 1c (#45), OLS fallback (#40), PoinTr (#28), Stage A production use (#31), PCN real fine-tuning (#16/#17/#19), merge-centroid fix (#46) | `docs/findings.md` |

**Freeze rule:** no edits to `src/pipeline.py`, `src/completion.py`,
`src/classifier.py`, checkpoints, or `output/08` until after submission. New work
goes to `scratchpad/` + new `output/experiments/` subfolders only (house
data-safety rule). The remaining evidence tasks (B1–B6 in `THESIS_PLAN.md` §3)
are read-only analyses against this frozen config — they do not change it.

**Evidence-task progress:** B1 (IoU sensitivity) + B6 (geometric-only ablation)
DONE 2026-08-21 (Finding #47, `output/experiments/iou_sensitivity/`): headline
F1 threshold-stable (0.811/0.808/0.785 at IoU 0.25/0.30/0.50); geometric-only
like-for-like F1 0.250 under promoted config (C2 corrected from the pre-promo
0.305). B2 (GT-eligibility count) DONE 2026-08-21 (Finding #48,
`output/experiments/gt_eligibility/`): seq 08 stride-20, 25.5% of all annotated
cars (14.6% of ≥10-raw-point cars) excluded by the ≥10-surviving-points
eligibility rule → implied recall against all annotated cars ≈0.54. **All
mandatory experimental items (B1/B2/B6) now closed;** B3 (literature table)/B4
(distance recall, optional)/B5 (pipeline diagram) are writing/figure work.

**Thesis writing (T14) — STARTED 2026-08-21 (Protocol milestone).** §4.1
(evaluation protocol) and §7.2 (donor metric) first-drafted as compilable LaTeX
in `docs/writing/thesis/` (`sec_4_1_evaluation_protocol.tex`,
`sec_7_2_donor_metric.tex`; build clean under TeX Live 2026; committed `e1fabd3`).
Both apply the `personal_agenda` P1 phrasing rules; NOT yet advisor-reviewed.
Execution plan + full checklist: `THESIS_PLAN.md`.

**Methods milestone — REACHED 2026-08-23 (committed `3ec88be`).** Ch 3
(`sec_3_detection_pipeline.tex`) + Ch 6 (`sec_6_completion_method.tex`)
first-drafted as compilable LaTeX (build clean, TeX Live 2026) in
`docs/writing/thesis/`.
- **Ch 3:** every pipeline stage + frozen-config table; production classifier as
  the from-scratch checkpoint (#31), Stage A as Ch-4 ablation; **B5 pipeline
  diagram** embedded as inline TikZ Fig 3.1 (per-stage ms from
  `timing_seq08_full_n4068_combined.json`, i.e. the promoted config — corrected
  2026-08-25 from the pre-#34 `_n4068.json`; see the cross-file audit block).
- **Ch 6:** completion debugging story (#15-19 → #26 data fix → #26 inference-bug
  fix → #27 L-shape gate → #35/#36 length → #45 closed Step-1c), eval deferred to
  Ch 7. Three figures, all referenced + building clean (7 pp): Fig 6.1
  naive-vs-KITTI-like partial (`scratchpad/fig61_partial_compare.py`); Fig 6.2 BEV
  box overlay of old-vs-corrected-vs-GT completion boxes, BEV IoU 0.32→0.73 /
  0.25→0.74 / 0.47→0.85 (sparse/mid/dense) (`scratchpad/fig62_box_overlay.py`,
  reconstructs the retired PCA+partial-radius path — freeze-safe); Fig 6.3 shipped
  completion output, partial-vs-completed canonical BEV+side
  (`scratchpad/viz_completion.py` from shipped-config `output/08`).
- **Completion-quality scope (settled):** real completions are car-shaped
  (footprint + roofline, dims L≈4.2/W≈1.9/H≈1.6), crispness varying with density,
  filled 4096-pt clouds not CAD-clean surfaces (sparse width-inflation #44). Ch 6
  claims pose+scale recovery + car-shaped output; crisp surface = synthetic, real
  quantitative gains = Ch 7. Other-sequences check: seq 00 mined (31 static cars vs
  seq 08's 15) corroborates — diffuseness is model-level.

**Next (historical, as of 2026-08-23):** Ch 4/5/7 results chapters (Results
milestone). Methods-milestone work (Ch 3/6 .tex + `fig61_partial_compare.py` +
`fig62_box_overlay.py` + THESIS_PLAN/project_state edits) committed 2026-08-23 in
`3ec88be`; working tree clean. output/ renders gitignored, regenerable from
scripts. **Superseded: Ch 4, Ch 5, and Ch 7 are all now drafted — see below.**

**Ch 4 results half (§4.2–§4.5) — DRAFTED + REVIEWED 2026-08-24; COMMITTED `151bde9`.**
`docs/writing/thesis/sec_4_results.tex` written (§4.2 results, §4.3 ablations,
§4.4 distance, §4.5 runtime); builds clean standalone under TeX Live 2026 (7 pp,
no undefined refs). Distinct label `ch:detection-eval-results` (folds under §4.1's
chapter at final assembly). Execution plan was
`C:\Users\ngovi\.claude\plans\draft-ch-4-modular-valley.md`. Two freeze-safe
read-only runs ran first (both same category as B1/B2/B6, touched nothing frozen):
1. **Track-filter ablation DONE** (Finding #49,
   `output/experiments/track_filter_ablation/seq08_notrackfilter.log`): full seq 08
   `--no-track-filter` middle row of Tab 4.2 = P 0.820 / R 0.677 / F1 0.741 /
   mIoU 0.921 (TP 23629 / FP 5204 / FN 11293). Reveals recall is **non-monotonic**:
   classifier drops recall (0.775→0.677) as the precision mechanism, track filter
   recovers it (→0.730) via temporal voting → §4.3 reframed as dip-then-recover.
2. **B4-lite recall-by-distance DONE** (Finding #50,
   `output/experiments/distance_recall/distance_recall_08{,_fine}.json`): per-frame
   / track-filter-off trend table; pooled TP/FP/FN 1165/268/572 **byte-identical
   to T10 HDBSCAN seq-08** (#43, validates the greedy-match replication). Finer
   0–10/10–20 bins (user call) surfaced the #23 near-range over-segmentation dip
   (0–10 m R 0.787 < 10–20 m 0.862) + far-range sparsity decline. Supersedes #5.
**Runtime table (§4.5) anchor decision (user "report both"):** table on the
full-run promoted timing `timing_seq08_full_n4068_combined.json` (623.5 ms, most
robust), caption reconciling the stride-20 estimate (650.6 ms, the figure cited
elsewhere) and the 921 ms pre-opt baseline; the 650.6-vs-623.5 gap is
stride-20-vs-full-run sampling, both the promoted config.
**Independent review DONE 2026-08-24** (fresh-session judge, `/review-handoff`;
handoff at `scratchpad/ch4_review_handoff.md`). ~30 numbers verified — zero
transcription errors in §4.2–§4.5. Fixes applied to the working tree: (1) §4.2
recall-limited claim re-grounded in measured #23/#24 split rates (was an
unquantified figure-only "anti-correlated" claim mis-cited to #34); (2) 66%-split
citation corrected #23→#24 (distance-binned rates live in #24 line 558) in both
§4.4 and Finding #50; (3) cross-domain Tab 4.3 diagonal filled (in-domain car-F1
0.999 / 0.885, #30) so the 0.000 collapse is credible; (4) §4.1 arithmetic fixed
(F1 drop 0.25→0.50 is 2.6 pts not 2.3; 2.3 is the 0.30→0.50 drop per #47) — edits
the committed sibling, flagged; (5) clustering ms column labelled "mean", runtime
"classical CPU" clarified to ~94% incl. geometric filter + JSON cited, FP rate
corrected to ~0.7/frame. Both files rebuild clean (sec_4_results 7 pp, sec_4_1 3 pp). Also fixed a
pre-existing 45pt overfull in §4.1's headline table (shortened "Mean IoU (matched)"
→ "Mean IoU" in both §4.1 tables — captions already qualify "matched" — matching
the §4.2 convention; +`\tabcolsep` on tab:headline).
**Remaining Ch 4:** advisor feedback if any. (Ch 7 completion results now DRAFTED
2026-08-26 — see the Chapter 7 block below.)

**Chapter 5 (The Recall Bottleneck) — DRAFTED 2026-08-25.**
`docs/writing/thesis/sec_5_recall_bottleneck.tex` written; builds clean standalone
under TeX Live 2026 (6 pp, no undefined refs). Mechanism + negative-results chapter,
cross-referencing (not re-tabulating) Ch 4's #43/#50 tables per user decision. Owns:
§5.1 recall is clustering- not classifier-bound (#22/#47), §5.2 root cause = HDBSCAN
splitting (#23, Fig 5.1 = `seq08_failure_zooms.png`), §5.3 Tab 5.1 failed post-hoc
repairs (BEV #21, MCS/merge/adaptive #24, temporal aggregation footnoted from
`session_history.md:898-906` — code reverted, pre-promotion baseline; per user call),
§5.4 the resolution fix (#34, recall 0.699→0.730) + explicit walk-back of the
retracted "~0.74 hard limit" (#24 Correction), §5.5 summary. All numbers traced;
no `src/`/frozen artifact touched. Plan:
`C:\Users\ngovi\.claude\plans\chapter-5-velvet-bubble.md`.

**Ch 5 external review processed — REVISED 2026-08-25** (`/review-handoff`, 12
points). Verified each against source before acting (findings.md, tracker.py,
evaluate.py, analyze_clustering.py). Fixes applied to the working tree: (1) Tab 5.1a
(tab:split) rebuilt as a true partition (ok/split/noisy+all-noise sum to the
denominator; merged demoted to a cross-cutting memo row, not summed) — the old rows
summed to 1639/810 vs headers 1631/811; (2) §5.2 reframed — dropped "usually still
detected" and "recoverable ceiling" (recall EXCEEDS the ok fraction, so it's a floor
not a ceiling; split cars are lost ~81% of the time; a large point-fraction ≠ a
match), added the shrinking clean-vs-recall gap 7.1→3.1 pt as the resolution-fix
predictor; (3) §5.1 "fragments not whole cars" softened to an inference
(analyze_clustering.py checks GT-overlap, not co-occurrence with a surviving
cluster); (4) §5.1 "1600 TP" pinned to 27051 vs 25478 (#47/#49, same full-seq08
sample) + disambiguated from #34's coincidental +1655; tracker-is-filter-only guard
clause added (tracker.py emits current-frame clusters only, no interpolation);
(5) §5.3 "post-hoc" over-generalisation fixed (only fragment-merge is post-hoc; the
real axis is alter-formation vs reassemble; objectness is the shared *ceiling*, §5.4)
— section retitled "Repair strategies fail to generalise"; (6) temporal aggregation
moved from Tab 5.1 to prose (reverted feature, session-log-only); (7) minors: all
four range-split bins shown (66/40/19/4, justifies "monotonically"); 811-vs-810 /
37.1-vs-37.7 reconciled in a footnote. NOTE: reviewer's citation-drift flag (66→4%
#24 vs #23) was a handoff-note error, not a .tex error — .tex already cited #24
correctly. Rebuilds clean (7 pp, no undefined refs). Still NOT advisor-reviewed.

**Chapter 7 (Completion Evaluation) — DRAFTED; COMMITTED 2026-08-26.** The full
completion-results chapter is written across two standalone-compilable files in
`docs/writing/thesis/` (both build clean under TeX Live 2026, zero undefined refs /
overfull >20pt / warnings; fold together at final assembly via the merge note in
`sec_7_results.tex`):
- **`sec_7_2_donor_metric.tex` (§7.1–§7.2, 4 pp)** — metric *definition* half:
  §7.1 why plain Chamfer is invalid on real cars (one-sided pseudo-GT rewards
  under-completion; raw partial scores best, #26), §7.2 the leakage-free donor
  metric (occluded-side principle, definition, validation battery, per-band
  hallucination guard #37). §7.1 was added this session; §7.2 dates to the
  2026-08-21 Protocol milestone.
- **`sec_7_results.tex` (§7.3–§7.7, 8 pp)** — *results* half: §7.3 real-data
  surface coverage (#32), §7.4 downstream amodal-box utility + length geometry
  (#29/#35/#36, #45 cross-ref), §7.5 moving cars (#44), §7.6 pre-registered
  held-out replication on seq 00 (#42, PARTIALLY HOLDS), §7.7 summary +
  limitations. Figures 7.1–7.3 via `\graphicspath` into gitignored `output/`
  (regenerable). Numbers traceable to Findings #29/#32/#35/#36/#37/#41/#42/#44/#45.
Both files were previously untracked/uncommitted; committed together this session.
Still NOT advisor-reviewed.

**Chapter 8 (Discussion & Limitations) — DRAFTED 2026-08-26.**
`docs/writing/thesis/sec_8_discussion.tex` written; builds clean standalone under
TeX Live 2026 (5 pp, zero undefined refs / zero overfull). Synthesis-then-limitations
chapter that names every audit-identified weakness so the committee cannot surface one
first. §8.1 what the thesis establishes (donor metric + completion value = strongest
evidence; detection = engineering result not SOTA; #25/#30 negative replaces two-stage
contribution); §8.2 eval validity (val-as-test for detection #18/Q1 — bounded because
recall ceiling is upstream of the classifier; held-out coverage gaps — empty long band
+ compact d2 fail, PARTIALLY HOLDS #42); §8.3 completion scope (statics-only accuracy /
movers plausibility-only #44; #41 23% length-estimate fallback; #45 length-dependent
residual under-extension; #37 band-blind guard as metric-design lesson, ratios reported);
§8.4 GT/dataset (amodal GT unvalidatable — tracklet cross-check impossible, 3 defences;
B2 recall-vs-all-annotated ≈0.54 / two distinct ≥10-point rules #48; SemanticKITTI label
quality assumed); §8.5 architectural (clustering-objectness ceiling from Ch 5 #34/#43;
offline runtime 623.5 ms, real-time out of scope #33/#34); §8.6 what generalises
(mechanisms transfer, constants are seq-08-derived, accuracy scoped to static compact+
normal cars). All six mandatory named limitations present; P1 phrasing rules applied.
First of the four remaining chapters (plan order: Ch 8 → Ch 2+B3 → Ch 1 → Ch 9 → abstract).

**Ch 8 external review processed — REVISED 2026-08-26** (`/review-handoff`, verified each
point against findings.md before acting; conceded all, no pushback). Fixes applied to the
working tree (now 6 pp, rebuilds clean): (A1, the load-bearing fix) §8.3 mislabelled the
0.0433/100× compact hallucination as "the shipped length prior" — it is the fixed 4.14 m
prior (#37 table; now only the 23% fallback), while the shipped per-car estimate is 16×
(0.0065); reattributed and connected to the #41 fallback. (A2) §8.1 BEV IoU 0.771 stated
"against the raw partial" → corrected to "against the amodal GT box, up from 0.725 for the
raw partial." (A3) §8.3 guard direction made explicit (out-of-box ≤ mirrored baseline;
>1 fails, higher worse; seq 08 16×/100× fails worst, seq 00 1.8× confirms held-out).
(B1) donor-metric novelty narrowed to the specific construction. (B2, the weakest defence)
§8.2 val-as-test bound repaired — conceded the classifier's realized recall contribution
is also seq-08-tuned (geom-only R 0.775 → classifier 0.677 → track filter 0.730, #47/#49),
dropped the "least in doubt" overclaim, bounded optimism to that band vs the
classifier-independent upstream ceiling. (B3) §8.1 "both claims survive" now carries the
PARTIALLY HOLDS qualifier. (B5) §8.6 grades the three mechanisms (Chamfer-invalidity +
donor construction transfer by argument; split diagnosis is an inference from two
sequences). (C1) compact hallucination named as a completion defect, #37↔#41 connected.
(C2) single held-out sequence stated. (C3) completion sim-to-real domain gap added (#26
residual flatness; #16/#17/#19 fine-tuning negative). (C4) unquantified-tracking limitation
added (no MOTA/IDF1; tracker feeds the 23% fallback). (C5) detection single-sequence
breadth caveat added, distinct from val-as-test optimism. Reviewer's B4 (a "wording
tightening" point) was referenced in the priority list but its text was absent from the
pasted review — not actioned. Still NOT advisor-reviewed. Uncommitted.

**Chapter 2 (Background & Related Work) + B3 table — DRAFTED 2026-08-26.**
`docs/writing/thesis/sec_2_background.tex` written; builds clean standalone under TeX
Live 2026 (7 pp, zero undefined citations, one 1.09 pt overfull). First literature
chapter, so it introduces classic BibTeX into the thesis tree:
`\bibliographystyle{IEEEtran}` + `\bibliography{../../references}` (relative path to
`docs/references.bib`, same pattern as `docs/report/progress_report_2026_05_12.tex`).
Scope = **related-work-focused** (user decision): positions the field, defers method
mechanics to Ch 3/6. Sections: §2.1 SemanticKITTI/KITTI + the seg/instance/panoptic
tasks + annotation cost; §2.2 deep segmentation (projection/point/voxel + panoptic);
§2.3 classical clustering (RANSAC + DBSCAN/HDBSCAN, the paradigm used); §2.4 object
completion (PCN/PoinTr/SnowflakeNet/SeedFormer + synthetic-to-real gap + Chamfer-
invalidity foreshadowing Ch 7); §2.5 positioning + the **B3 table** (Tab 2.1).
**B3 = real published numbers, conservative non-head-to-head layout** (user decision):
requirements columns lead, each metric cell labelled with its own metric, explicit
non-comparability caveat in caption + prose (defends Q2/Q10). **All Tab 2.1 numbers
web-verified against primary papers** (SemanticKITTI test set): RangeNet++ mIoU 52.2 /
car IoU 91.4, RandLA-Net 53.9 / 94.2, Cylinder3D 67.8 / car IoU 97.1;
RangeNet++△+PointPillars PQ 37.1 / PQ^Th 20.2, Panoptic-PolarNet PQ 54.1 / PQ^Th 53.3,
DS-Net PQ 55.9 / mIoU 61.6, EfficientLPS PQ 57.4 / PQ† 63.2; this work car-F1 0.808 /
R 0.730 / P 0.905 (point-IoU≥0.3, seq 08, Finding #34). Note: the "PQ 63.6" seen in
secondary sources is Panoptic-PolarNet's **RQ**, not PQ — corrected to test PQ 54.1.
**`docs/references.bib` appended** (additive, safe) with 8 new entries:
`zhu2021cylindrical`, `zhou2021panopticpolarnet`, `hong2021dsnet`,
`sirohi2021efficientlps`, `behley2021panoptic`, `gasperini2021panoster`,
`ester1996dbscan`, `campello2013hdbscan`, `mcinnes2017hdbscan`, plus `lim2025longrange`
(Lim & Park, *IEEE Access* 13, 2025, DOI 10.1109/ACCESS.2025.3541267) — the closest
**same-paradigm** prior work (BEV clustering + ML vehicle classifier for long-range
LiDAR detection), extracted from the `.md` conversion of the previously
password-protected PDF and cited in §2.3 (paradigm anchor) + §2.5 (contribution
boundary). PoinTr (`yu2021pointr`) and PCN (`yuan2019pcnpointcompletionnetwork`)
confirmed already cited in §2.4 and correct against their paper `.md` files.
Uncommitted; NOT advisor-reviewed. Remaining chapters (plan order): Ch 1 → Ch 9 →
abstract.

**✅ RESOLVED 2026-08-26 — Ch 2 Tab 2.1 numbers PRIMARY-SOURCE verified; zero errors.**
The parked verification was restarted and completed against each paper's own results
table (ar5iv.labs.arxiv.org rendered them cleanly this pass; arxiv.org was back up).
This is now a *measured* read of the primary tables, not a WebSearch summary. **All
seven published Tab 2.1 rows confirmed, all test-set, no transcription errors:**
- Semantic (RandLA-Net arXiv 1911.11236 Table 3 "online single scan eval track" = test;
  Cylinder3D arXiv 2109.05441 Table I "single-scan test set"): RangeNet++ 52.2 / car
  91.4; RandLA-Net 53.9 / car 94.2 (own paper — Cylinder3D's table lists RandLA-Net as
  50.3 / 94.0; the `.tex` correctly uses each method's own-paper number); Cylinder3D
  67.8 / car 97.1.
- Panoptic (each own paper, test set): RangeNet++ & PointPillars PQ 37.1 / PQ^Th 20.2
  (via Panoptic-PolarNet Table 1); Panoptic-PolarNet PQ 54.1 / PQ^Th 53.3 (RQ 65.0);
  DS-Net PQ 55.9 / mIoU 61.6; EfficientLPS PQ 57.4 / PQ† 63.2.
- **The suspected Cylinder3D 97.1→96.4 fix is REJECTED.** Cylinder3D's own Table I
  prints mIoU 67.8 and car IoU 97.1 *in the same row* — the 96.4 was itself a
  secondary-source artifact. `sec_2_background.tex` line 241 is correct as-is; NO edit.
- **Test-vs-val (external-review concern #1) closed:** the online single-scan eval track
  is the test benchmark (seq 11–21; val cannot be submitted), so the two semantic rows
  are test numbers and the caption's "test-set" claim holds. The earlier extraction's
  "validation" label was the error, not the numbers.
- The earlier "all Tab 2.1 numbers web-verified against primary papers" wording (above)
  was overstated *when written* (it rested on WebSearch summaries, one of which returned
  a wrong 76.1 mIoU); it is now genuinely primary-verified.
- **External-review minors — ALL APPLIED 2026-08-26 (rebuilds clean, 7 pp):**
  RangeNet++-appears-twice footnote added to Tab 2.1 caption; Cylinder3D "among the
  strongest" → factual "reports 67.8 mIoU… among the higher single-scan results"; §2.1
  "22 sequences" clarified (00–10 labelled, 11–21 withheld to the online test server);
  `references.bib` EfficientLPS → `year=2022, volume=38, number=3` (citation-of-record).
- **Reference archive COMPLETE 2026-08-26 (`docs/papers/`, PDFs gitignored/local):**
  all 25 Ch-2 cited papers downloaded + pdfminer-verified readable (RangeNet++ not on
  arXiv → IPB PDF; EfficientLPS/SeedFormer >10 MB → user-fetched; DBSCAN/campello2013
  paywalled → user-fetched; rest via WebFetch, sandbox `curl` cannot reach arXiv/CVF).
- **Already applied this session (independent of the above, compiles clean, exit 0):**
  §2.5 reworded recall to a paradigm claim (no cross-metric implication); new paragraph
  on the 3D-detection omission (why point-IoU not box-AP) + §4.1 forward pointer;
  tracking-as-filter sentence; §4.5 runtime pointer. `references.bib` gained 3 entries:
  `lang2019pointpillars`, `yin2021centerpoint`, `weng2020ab3dmot`.

**Cross-file audit fixes — DONE 2026-08-25.** An external audit (5 thesis drafts
vs. frozen evidence, ~30 numbers) flagged one must-fix + should/optional items;
all six applied (Ch 3/4/6/§7.2). Central fix: **Ch 3 was reporting pre-#34
timings** (`timing_seq08_full_n4068.json`, 921 ms / 1.1 fps) for the promoted
pipeline its own Table 3.1 documents — regenerated every Ch 3 timing from
`timing_seq08_full_n4068_combined.json` (623.5 ms / 1.6 fps; ground 56, HDBSCAN
370 = 59%, classifier 34, geo 36, preproc 126, 45 clusters/frame, completion +13
ms/car), so Ch 3 now agrees with §4.5. Also: "learned stages" accounting
corrected (classifier-only, not classifier+completion/tracker) in Ch 3 + §4.5;
§7.2 n=39 (acceptance run #32) vs n=40 (promoted re-run #37) reconciled with
inline cites; Ch 6 Fig 6.2 box-IoU triples cite the `fig62_box_overlay.py` render;
Ch 3 preprocessing-order framing softened. No frozen artifact changed (the
evidence was correct; Ch 3 had cited the wrong file). Text-only edits, no
recompile run.

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

### Seq 08 (full, 4071 frames) — generalization check (promoted config, updated 2026-07-23)

| Metric | Value |
|--------|-------|
| Precision | 0.905 |
| Recall | 0.730 |
| F1 | 0.808 |
| Mean IoU | 0.912 |

TP=25478 FP=2676 FN=9444. Promoted `PIPELINE_CONFIG` (voxel_before_denoise,
ransac_iterations=300, cluster_voxel_size=0.10; Finding #34). Pre-opt numbers
(0.903/0.699/0.788/0.895, TP=23823 FP=2565 FN=10240) preserved in #34.
Command: `.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000`.
Confirms the seq-00 story at 40× scale: precision-saturated, recall-limited;
per-frame recall anti-correlated with GT-car density, FP flat ~1/frame.
Figures: `output/figures/seq08_{bev_detections,failure_zooms,timeseries}.png`.

## Recall Bottleneck — Characterized; partly lifted by coarse-voxel clustering (#34)

The recall shortfall was root-caused to HDBSCAN **splitting cars into fragments**
(not a classifier problem). Five *post-hoc* repair strategies all failed on
held-out data — **but** coarsening the clustering resolution (cv=0.10, promoted
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

## Project Focus (updated 2026-06-30): Point-Cloud Completion

Direction shifted to deepen completion, prioritizing **thesis narrative
strength**; retraining acceptable. Four directions:

1. **Valid real-data completion metric** — donor-frame occluded-side Chamfer +
   symmetry self-consistency; curated synthetic bench. Foundational: currently
   NO valid real-data metric (pseudo-GT CD invalid, #26/#27).
2. **Improve `complete()` geometry** — centroid (dominant residual error), 90°
   heading flip. Startable now on the valid synthetic metric.
3. **Close train-vs-real partiality gap** (#28 bottleneck) via masked-Chamfer
   fine-tuning on real cars. Retraining OK; contingency (negative-result
   precedent #16/#17/#19).
4. **Downstream utility** — completion improves bbox dims/orientation
   (measurable now via GT boxes) or recovers split cars (#23).

Chosen order (narrative-first): **4a → 1 → 2 → 3.** Scratchpad viz scripts kept
in scratchpad (Group 1 frozen records / Group 2 reusable tools); not promoted.

Full roadmap and active step plan: **`docs/completion/plan.md`**.

Prior completion milestones (DONE): KITTI-like PCN data fix + inference-bug fix
(#26), single-frame completion wired into `main.py`, L-shape input gate
(precision 38%→69%, #27), full seq-08 regenerated (`output/08`, 518 completed),
PoinTr benchmarked → keep PCN (#28; synthetic table corrected 2026-07-06 via
matched eval `scratchpad/matched_eval_pcn_pointr.py` — PoinTr's edge is small
(CD 0.161→0.153 m, F 0.755→0.782), not the "halves CD" originally recorded;
real-data equivalence and keep-PCN decision unchanged). See `docs/findings.md`.

## Direction 4a — COMPLETE (2026-07-05): "Completion adds value" established

Finding #29. All four steps done (`docs/completion/plan.md` has details):
Step 0 amodal GT (40 well-observed static cars, `output/08/amodal_gt.json`);
Steps 1–3 paired box eval (`scratchpad/completion_box_eval*.py`): 2,075 TP
pairs / 1,339 completed on seq 08.

**Headline (per-car medians, n=39, Wilcoxon):** completed box beats raw partial
on BEV IoU 0.707→**0.747** (p=.002), |ΔW| 0.270→**0.170** (p=1.5e-4), |ΔH|
0.255→**0.131** (p=1.6e-10), center err 0.286→**0.234** (p=2.8e-5); L and yaw
neutral. Gains largest on sparse inputs (<100 pts: IoU 0.461→0.599). Figure:
`output/figures/completion_box_overlays_08.png`.

Direction-2 targets logged (not blockers): length under-completion on normal
cars (far end not extended, signed ΔL −0.49→−0.55) and heading errors on
sparse inputs.

## Direction 1 — COMPLETE (2026-07-17): valid real-data completion metric

Finding #32; method/results: `docs/completion/donor_metric.md`. Donor-frame
occluded-side metric: complete from one frame's pipeline cluster, score
coverage of donor points (other frames of the same static car) ≥ 0.15 m from
every input point, + out-of-amodal-GT-box hallucination guard. Seq 08:
2,092 TP pairs / 1,337 gate-passed / 39 cars. **Validation gate: all four
items pass.** Headline (per-car medians, n=39): cov@0.1 raw 0.000 / mirrored
0.043 / **completed 0.304**, med novel-dist 0.518→**0.161 m**, out-of-box
~0; all Wilcoxon p < 1e-6. First real-data evidence PCN reconstructs unseen
surface (7× the symmetry-mirror baseline). Weakest region = far end (cov
0.133) — #29's length under-completion, now measurable. Supporting refactor:
`estimate_canonical_frame()` extracted from `complete()` (behavior-preserving).
Figure: `output/figures/donor_metric_08.png`.

## Direction 2 — improve `complete()` geometry

### Step 1 — far-end under-completion: DONE (2026-08-01, Finding #35)

Longitudinal length prior added to `estimate_canonical_frame()` (extend-only Z
push toward the ego-far end, `COMPLETION_CAR_LENGTH_PRIOR = 4.14 m`, mirrors the
width prior; shipped ON, `length_prior=None` to A/B off). Inference-only, no
retraining. **Paired seq-08 result (donor metric, per-car median, τ=0.15, n=39):
far_end cov 0.123 → 0.324 (2.6×), overall cov 0.307 → 0.428, med-dist 0.162 →
0.117; out_of_box 0.0004 → 0.0014 (≪ 0.0083 guard).** Box metric (#29):
reverses the length regression — signed ΔL −0.44 → −0.32 (completed now beats
raw), |ΔL| 0.44 → 0.35, width flat, |ΔH| +1.8 cm. L_prior=4.5 rejected
(out_of_box 0.0122 breaks the guard — over-extends compacts). Synthetic true-GT
check corroborates (far-quarter cov 0.42 → 0.59). Scripts:
`scratchpad/length_prior_{synth_check,box_recheck}.py`,
`scratchpad/donor_metric_recompute.py`; A/B outputs in
`output/experiments/donor_metric_len_{off,414,450}/`.

**Follow-on:** `output/08` **regenerated under the shipped prior (2026-08-01)** —
1040 tracks / 518 completed / 1558 PLYs; only the 518 completed clouds changed
(detection/tracking + `_partial.ply` inputs byte-identical, verified by md5), GT
artifacts (`amodal_gt.json`, `amodal_gt_check.png`) copied in. No-prior version
preserved at `output/08_noprior_backup/` (reversible; older `output/08_preperf_backup/`
also kept). (Superseded 2026-08-03 — see Step 1c.)

**#29/#32 production-config tables refreshed (2026-08-01, prior OFF vs ON,
promoted `PIPELINE_CONFIG`, `stage_b_scratch` for both — also resolves the #29
classifier drift; n=40 cars / 1508 pairs, paired):** #32 donor far_end cov 0.121
→ 0.346, overall cov 0.301 → 0.403, out_of_box 0.0004 → 0.0020 (guard holds); #29
box |ΔL| 0.476 → 0.354 (reverses the length regression), BEV IoU 0.743 → 0.747.
**Caveat:** the fixed 4.14 m prior over-extends compacts (< 3.6 m: signed ΔL
−0.10 → +0.25) while fixing normal cars (−0.55 → −0.34) — a per-car length
estimate would remove this (Step-1b idea). Tables in Finding #35;
outputs `output/experiments/donor_perf_len{off,on}/`,
`completion_box_eval/step2_metrics_08_len{off,on}.json`.

### Step 1b — compact-overshoot fix: DONE (2026-08-02, Findings #36/#37)

Fixed prior replaced by a **per-car estimate**: `track_length_estimate()` in
`src/completion.py` = q90 of gate-passed `fit_length` over the track + 0.12 m
(`COMPLETION_LENGTH_TRACK_{QUANTILE,OFFSET}`), fallback 4.14 below 5 frames.
Plumbed as an optional `length_estimate` arg through `complete()`; `main.py`
aggregates over the track it already completes. **Shipped ON.**

Estimator chosen offline against amodal GT with no PCN inference
(`length_estimator_probe{,2}.py`), which killed aspect-ratio (corr(GT L, GT W) =
+0.018), height/range/density, a far-end truncation test, and track-max.

**Box metric (band split, n=40):** compact signed ΔL +0.295 → **+0.063**
(p=.016), compact center err 0.322 → 0.192; ALL |ΔL| 0.354 → **0.304**, BEV IoU
0.747 → **0.771**, center err 0.229 → **0.184**. **Donor cost:** pooled cov
0.403 → 0.364, far_end 0.346 → 0.316 (better on compact + long, worse on normal).

**Finding #37 (guard defect):** #32's out-of-box gate is a pooled median and was
blind to the shipped prior hallucinating at **100×** the compact band's mirrored
baseline (0.0433 vs 0.0004) — the statistic #35 used to size the prior could not
see the failure it was meant to catch. Per-band gate `d2` added to
`donor_metric_step3.py` and backfilled into all donor summaries (originals kept
as `*.pre_d2_backup.json`). #35's direction stands; its magnitude was validated
by a blind guard.

**Sparse-fallback OLS (Finding #40): tried, rejected.** Replacing the 4.14 m
constant fallback (tracks < 5 gate-passed frames) with a single-frame OLS was
A/B'd frame-correctly and is a **wash** — it fires on only 2/40 cars, box |ΔL|
0.214 → 0.221. Kept the simpler constant.

**Correctness fixes this session (not Direction-2 geometry):** (#38) production
`complete()` carried `self._rng` across tracks, so a completed cloud depended on
call order — `complete(sample_seed=…)` + per-call reset in `main.py` makes it
order-independent (reproducibility, not quality — seed noise sd 0.0014). (#39)
donor-cache frame trap: `data["raw"]` is **sensor** frame; `world_box` needs
`raw @ Rᵀ + t` (cost four retracted probe scripts).

### Step 1c — second under-extension mechanism: TESTED, NEGATIVE (2026-08-09, T13, Finding #45)

Pre-registered (`docs/completion/t13_step1c_plan.md`) and tested: **D1** decouple
`radius` from the Z length-push + **D2** length-axis fill factor (fill_z=1.074,
calibrated on synthetic true GT, n=300). A/B behind
`PointCloudCompleter(decouple_radius, fill_z)` (default OFF). **Verdict: DO NOT
SHIP.** Fixes the target (seq-08 normal |ΔL| 0.329→0.232 p=7.7e-3, long
0.583→0.363; donor far_end cov up on both seqs) **but over-extends compacts**
(|ΔL| +0.109 seq 08 / +0.116 seq 00) → the pre-registered compact non-regression
guard fails on both sequences. Confirms the compensation is irreducibly
length-dependent (option 3), which stays out of scope. Production unchanged
(flags default OFF, invariant tests green). Original scoping notes below.



#36's over-extension control (q90 **+0.45**) improved **both** metrics on
normal/long cars (cov 0.364→0.483, far_end 0.316→0.509, box |ΔL| 0.304→0.210), so
completions are still genuinely too short even when the length estimate is
unbiased. Cause is inside the completion, not the center estimate: PCN under-fills
its normalized frame, and the center push also drives `radius` (output scale), so
the length prior doubles as a rescale. Required compensation is length-dependent
(≈0.02/0.68/1.02 m by band) — too few cars to fit safely. Plan in
`docs/completion/plan.md` (decouple radius from the center push first; calibrate
fill factor on synthetic true GT).

**`output/08` regenerated (2026-08-03, T7 of the delegate brief) under the
per-car estimate + #38 RNG fix** — 1040 tracks / 518 completed, verified
byte-identical detection/tracking vs. the prior version
(`scratchpad/verify_regen_08_t7.py`, md5). Fallback-frequency measured for the
first time: 119/518 (23.0%) of completed tracks fell back to the 4.14 m
constant (< 5 gate-passed frames) rather than the per-car estimate — well
above the ~5% threshold, logged as a live limitation (Finding #41), no fix
implemented. Old (fixed-prior/pre-#38) version preserved at
`output/08_fixedprior_backup/`.

### Step 2 — remaining Direction-2 target: heading/center on diagonal/sparse views

The other #29/#32 weakness (worst donor figure panels + out-of-box flags): the
completed cloud rotates off the box on diagonal/sparse inputs. Candidate levers:
symmetry-derived center/heading (`next_ideas.md` #2), symmetry-mirror input
(#1). Measure with the same donor metric + #29 boxes. Idea backlog:
`docs/completion/next_ideas.md`. Plan: `docs/completion/plan.md`.

Far-end plan (written up in
`docs/report/results_report_2026_07_17_rev2.docx` §6.1): Step 1
inference geometry (L-shape near-corner anchor + longitudinal length prior +
symmetry-derived center; no retraining) → Step 2 visibility-weighted
asymmetric Chamfer retrain on KITTI-like synthetic → Step 3 (contingency)
masked-Chamfer real fine-tuning. Architecture changes ruled out (#28).

Amodal-GT cross-validation against KITTI raw tracklets: **RESOLVED as
impossible (2026-07-17)** — raw drive 2011_09_30_drive_0028 (= odometry seq
08) has no tracklet annotations (verified: 404 on the official S3 bucket
`avg-kitti/raw_data/2011_09_30_drive_0028/…_tracklets.zip`; control drive
2011_09_26_drive_0001 exists; 2011_09_30_drive_0027 and 2011_10_03_drive_0027
also unannotated). Thesis defense of the amodal GT instead rests on: paired
design tolerates reference noise (#29), construction guards (viewpoint
coverage + zero motion), and dimension sanity check (median L 4.14 / W 1.75 /
H 1.47 m vs published car statistics). Written up in rev2 §6.2.

### DONE: pipeline runtime optimization (Findings #33, #34; 2026-07-23)

`docs/perf/plan.md` + locked plan (Section A resolved). Baseline: full seq 08 =
**934.5 ms/frame** (stride-20 harness; 921 ms full-run). Tier 1 (exact-output,
per-frame TP/FP/FN bit-for-bit, #33) + Tier 2/3 (behavior-changing, promoted,
#34) landed. **Promoted combined config** (now the `PIPELINE_CONFIG` defaults):
`voxel_before_denoise=True`, `ransac_iterations=300`, `cluster_voxel_size=0.10`.

- **Runtime: 934.5 → 650.6 ms/frame (−30%, ≈1.54 fps)** — stride-20 benchmark
  `output/experiments/timing/timing_seq08_stride20_combined.json`.
- **Detection improved on full seq 08** (guard metrics, all up, none regressed):
  P 0.903→0.905, R 0.699→**0.730**, F1 0.788→**0.808**, mIoU 0.895→0.912
  (TP 23823→25478, FP 2565→2676, FN 10240→9444). cv=0.10 closes intra-car
  density gaps (#23) → +1655 TP.
- **≤400 ms target NOT reached, and proven unreachable via the authorized tiers:**
  the post-ground object cloud is intrinsically sparse, so coarse-voxel compresses
  it only ~18% and Python HDBSCAN has a ~185 ms floor on it. User decision
  (2026-07-23): **accept 650 ms** (real-time is out of scope); deferred GPU HDBSCAN
  (cuML/WSL2) and Open3D DBSCAN paths NOT pursued.

**Completion re-check DONE (isolate-config, 2026-07-23):** re-ran #29/#32 against
the promoted config, each keeping its own published classifier (#29 `stage_b_best`,
#32 `stage_b_scratch`) so only the config differs. **Neither finding shifts** —
#29 box quality unchanged (BEV IoU 0.747→0.739, |ΔW/H/L| within ±0.008), #32 donor
coverage barely moved (τ=0.10 0.346→0.338, τ=0.15 0.304→0.302), both still far
above their raw/mirrored baselines. Deltas within noise; n grew (n_pairs 2075→2262,
n_cars 39→40) because detection recall improved, so read population-level, not
paired. Detail + tables in Finding #34. Baselines preserved (`*_perf` outputs).
The #29 classifier drift (fine-tuned vs production scratch, pre-#31) is untouched
by design — flagged as a separate pre-existing item.

**`output/08` regenerated DONE (2026-07-23):** re-ran under the promoted config
(1040 tracks, 518 completed, 1558 PLYs); GT artifacts copied in; old output
preserved at `output/08_preperf_backup/` (reversible).

**Report refresh DONE (2026-07-23):** created
`docs/report/results_overview_2026_07_23.docx` — full refresh of §1–§6 + Summary
to the promoted operating point (one consistent operating point throughout), plus
an **honesty pass**: §3 reframed (dropped the obsolete "~0.74 hard ceiling"; the
cv=0.10 recall lift 0.699→0.730 is now first-class, ceiling "smaller than first
claimed"); §5.4 length prose made to match Table 4's "worse" verdict
(|ΔL| 0.428→0.463 m, p=0.034); [22] track-filter ablation given a provenance
caveat (0.762→0.801 is pre-opt; endpoint now 0.808). Item 3b (#29 fine-tuned vs
production scratch checkpoint) left undisclosed by design — immaterial to the
paired raw-vs-completed delta. All Tier-1/2/3 code edits + findings/state/report
**committed and pushed** (2026-07-24) in `97fb75b` (perf + report refresh) and
`a25455b` (Finding #24 correction); working tree clean, `main` == `origin/main`.

**Finding #24 wording corrected DONE (2026-07-24):** appended a dated
`**Correction**` note (house style, cf. #28) superseding the "~0.74 hard limit"
conclusion — the failed strategies were all *post-hoc*; changing clustering
*resolution* (cv=0.10) recovered part of the loss (recall 0.699→0.730, #34).
Original 2026-06-24 reasoning preserved as historical record.

### Deferred
- Thesis writing (pipeline description, experiment results, recall-ceiling discussion).
  Advisor progress report done 2026-07-05: `docs/report/progress_report_2026_07_05.md`.
- Advisor results reports done 2026-07-17 (`docs/report/`):
  `results_overview_2026_07_17.docx` (condensed, for the advisor) +
  `results_report_2026_07_17_rev2.docx` (extended justifications /
  defense-prep companion). Overview sent 2026-07-17; advisor asked (2026-07-19)
  whether metrics cover many frames/cars or one car — i.e. test scenarios were
  not explicit. Fixed in `results_overview_2026_07_19.docx`: new "Test scenario
  and evaluation protocol" subsection at top of §2 (seq 08 = 4,071 scans / 393
  distinct cars / ≈34k per-frame car instances, IoU≥0.3 greedy 1-to-1 point-level
  matching, micro-averaged TP/FP/FN; seq 00 100 frames = 41 cars; completion
  §5.3–5.4 unit = 39 static cars, per-car medians). Distinct-car counts from
  label scan (sem ∈ {10,252}, inst>0, ≥10 raw pts). Note: this "≥10 raw pts"
  is a distinct-car *count* threshold applied to unfiltered label points
  (label scan), not the eval recall denominator ("GT instances with ≥10
  points surviving preprocessing" — z-filter, voxel, denoise, ground
  removal — used by `evaluate.py`); the two ≥10-point rules measure
  different quantities and are not interchangeable.
- Advisor follow-up (2026-07-19): inference time + completion-metric
  explanation. Timing benchmark (`scratchpad/timing_benchmark.py`, Ryzen 7
  7800X3D + RTX 3070 Ti): headline from **full seq 08, all 4,071 frames**
  (`output/experiments/timing/timing_seq08_full_n4068.json`, 3 warmup
  excluded): **921 ms/frame ≈ 1.1 frames/s** — HDBSCAN 502 ms (54%), RANSAC
  163 ms (18%), preprocessing 136 ms, classifier 74 ms (~43 clusters/frame,
  1.7 ms each), tracker 0.3 ms; PCN completion +19 ms per completed car
  (n=12,322; gate rejection 1.5 ms, n=14,823). Scene-density variability:
  ~450-frame block means range ≈770–1,040 ms. Sampling lesson: the
  first-100-contiguous frames understate the mean by 28% (675 ms — sparse
  opening scene); uniform stride-20 sample (201 frames, 934 ms) agreed with
  the full run to 1.4%, so ~100 frames suffice *if drawn across the whole
  drive*. Learned components are cheap; classical CPU stages dominate (72%).
  Both answers added to `results_overview_2026_07_19.docx`: "Inference time"
  section after §1 and "§5.3 Completion evaluation metrics" (Coverage =
  primary quality metric w/ out-of-box hallucination guard; BEV IoU =
  downstream utility; they cross-check each other). Doc restyled 2026-07-19
  from Q&A tone to report tone (question headings removed; completion
  sections renumbered §5.3–§5.5) so it can serve as a school
  submission / paper draft base. Docx and VN reply ready to send (final figure
  921 ms/frame supersedes the ~934 quoted in the earlier chat draft).
- Pipeline diagram for thesis report.

## Maintenance Pass — Delegate Brief Tier 1–2 (2026-08-02/09)

Executing `docs/plans/delegate_brief_2026_08_02.md` (full-repo review
follow-ups, delegated with defaults + STOP conditions). Tier 1 (T1–T7)
DONE 2026-08-03, one commit per task, all gated on the seq-00
reproduction baseline (P 0.967/R 0.777/F1 0.862/mIoU 0.962, TP=1296/
FP=44/FN=371) matching exactly: T1 `9ba02f2` (scratchpad scripts
tracked), T2 `4df200a` (evaluate.py CLI drift), T3 `5303291`
(resolve_track_class dedup), T4 `f046443` (invariant tests), T5 `ef73981`
(doc consistency), T6 `bcf012f` (fallback-frequency instrumentation),
T7 `a4a65c5` (output/08 regen, Finding #41).

### T8 — held-out sequence selection + amodal GT: DONE (2026-08-09)

Tier 2 begins. Default seq **05 fell to the fallback rule**: only **11
well-observed static cars** survived the guards (< 15 threshold). Dominant
rejection was `center_spread` (107/170 fitted) — seq 05 has few cleanly-static,
well-surrounded parked cars; its survivors also skew short (median
L 3.61/W 1.82/H 1.50). Per the brief, fell back to seq **00** (never 08).

**Seq 00 fallback SUCCEEDS: 46 well-observed static cars** (of 531 fitted / 537
observed). Same guards as seq 08. **Sanity gate passes both halves:** median
well-observed dims **L 3.80 / W 1.81 / H 1.49 m** (W/H within 2–6 cm of the
seq-08 reference L 4.14/W 1.75/H 1.47; L ~0.34 m shorter but inside the sane
range 3.5–5.0 — the well-observed set skews compact), and the visual check
(`output/00/amodal_gt_check.png`, 4 typical + 8 worst outliers) shows tight,
correctly-oriented boxes on all 12; short-L outliers are genuinely short/compact
clouds, not mis-fits. Artifacts (gitignored): `output/00/amodal_gt.json` (chosen
GT), `output/00/amodal_gt_check.png`, `output/05/amodal_gt.json` (fallback-trigger
record). No `src/` change, no existing output overwritten.

**Confound logged (per brief):** seq 00 is in the classifier's Stage-B training
split (train 00–07,09–10; val 08), so detection recall reads optimistic. The
completion claims survive this because T9b compares raw vs completed on the SAME
paired TP inputs, and PCN was trained on synthetic (never seq 00). No labeled
sequence is simultaneously classifier-held-out and completion-held-out (08 is the
only classifier-val seq but completion constants were tuned there; test seqs
11–21 have no labels, so no amodal GT is possible).

### T9a — pre-registration: DONE (2026-08-09, user-approved)

Held-out sequence for T9 fixed as seq **00**. Pre-registration
`docs/plans/preregistration_heldout.md` drafted, **user-approved**, and
committed BEFORE any seq-00 eval (`c524612`, immutable timestamp). Registers:
primary refutation-bearing metrics = **BEV IoU (#29) + donor cov@0.1 (#32)**
(completed beats raw, per-car medians, Wilcoxon p<.05); |ΔW|/|ΔH|/|ΔL|/center-err
secondary/reported; R1 does-not-generalize, R2 08-specific length constants
(per-band d2), R3 escalate uncovered results; outcome taxonomy
HOLDS / PARTIALLY HOLDS / DOES NOT GENERALIZE.

### T9b — frozen-config evals: DONE (2026-08-09)

Ran #29 box + #32 donor (steps 1–3 + d2) on seq 00, production config +
checkpoints, per-car length estimate ON (`track-q90off` = shipped), no tuning.
n_pairs=2588 / 1592 gate-passed; **n_cars=45** (of 46; one all-rejected);
bands compact=17 / normal=28 / **long=0**. Tables committed to
`docs/plans/t9b_results_heldout_seq00.md` (the T9c input). Helper
`scratchpad/t9b_box_all_wilcoxon.py` adds the pooled ALL-cars box Wilcoxon
(length_1b only does per-band). **Headline (numbers only — verdict is T9c):**
donor cov@0.1 (primary τ) raw 0.000 / mirrored 0.050 / **completed 0.413**, all
p≈0, win across every band/tau/region; box BEV IoU ALL raw 0.739 → **0.766**
(p=1.6e-3), |ΔW| 0.203→0.165, |ΔH| 0.278→0.120 (both p<1e-3). Nuances flagged
for T9c: BEV IoU win is normal-band-driven (compact 0.783→0.771 p=0.85 n.s.);
d2 compact pass-bit **fails** (completed 0.009 > mirrored 0.005, ratio ~1.8× vs
seq-08 ~16×); **long band empty** → long-band predictions untestable (R3).

**Next: T9c** = fresh Opus session (executor/judge separation) — reads ONLY the
committed pre-registration + `t9b_results_heldout_seq00.md` and writes the
finding draft applying R1/R2/R3 verbatim. NOT this session.

### T9c — arbitration verdict: DONE (2026-08-09)

Fresh judge session applied R1/R2/R3 verbatim to the T9b tables. **Verdict:
PARTIALLY HOLDS** (Finding #42, full worked verdict
`docs/plans/t9c_verdict_heldout_seq00.md`). Both primaries are decisive wins
(BEV IoU 0.739→0.766 p=1.6e-3; donor cov@0.1 0.000→0.413 p≈0); R1 and R2 both
clear verbatim; R3 triggered by the **empty long band** (0 cars ≥4.6 m →
long-band checks untestable), escalated and resolved via the taxonomy's
"caveat named" clause (user decision) — downgrade is a coverage gap, not a weak
metric. **Tier-3 gate satisfied (T13 permitted).**

### T10 — clustering benchmark: DONE (2026-08-09)

Backlog #5. HDBSCAN vs DBSCAN vs Euclidean at fixed `cluster_voxel_size=0.10`,
identical pipeline, per-frame (track filter OFF — stride/tracker incompatible).
**HDBSCAN wins F1 on seq 00 (0.831 vs 0.767/0.768) and seq 08 (0.735 vs
0.678/0.683), entirely via recall; precision is method-independent.**
Alternatives are 5–7× faster (58–92 ms vs 393–452 ms). eps sweep confirms no
fixed radius reaches HDBSCAN's recall (best: Euclidean eps=0.4, R 0.684/F1
0.800 < HDBSCAN 0.731/0.831). **Adopt nothing** — HDBSCAN stays production
(Finding #43). `dbscan`/`euclidean` kept as `--clustering-method` options.
Baseline re-verified EXACT after the additive `src/` edits.

### T11 — moving-car plausibility: DONE (2026-08-09)

Checked whether production-completed movers (never validated; #29/#32 covered
statics only) form plausible car boxes. On regenerated `output/08` (518
completed car tracks), split by kinematic net-displacement of the track
centroid (static ≤2 m / moving ≥5 m / ambiguous). **Movers complete as
plausibly as statics: 57.9% (11/19) vs 53.7% (225/419)**, near-identical median
dims; every motion bucket 54–58% → motion does not degrade completion
plausibility (completion is single-frame, no motion smear). Recipe caveat: the
#28 axis-aligned `dims` box understates quality for diagonally-oriented cars
(most failures are width-inflation on genuine car shapes), but the comparison is
fair (same recipe both groups). Finding #44; figure
`output/figures/t11_mover_completions_bev.png`. No fixes (observational).

**Remaining delegate-brief tasks:** T13 DONE (Step 1c, negative result — Finding
#45, `t13_step1c_plan.md`; do not ship, production unchanged). **Only T14
(thesis chapters, user-supervised) remains** — all mechanically-delegated tasks
(T1–T13) complete.

## Medium-Term Backlog

4. ~~Ablation: Stage A-only vs Stage B on real data~~ — done (Findings #7, #25)
5. ~~Benchmark HDBSCAN vs Euclidean clustering vs DBSCAN~~ — done (Finding #43, T10)
6. Replace global RANSAC with grid-based ground removal (Patchwork++)
7. Tracker upgrade — IOU-based matching (SORT-style)
8. Recreate `.venv` in place (Python 3.10.11) — fix relocated-venv pip launchers
