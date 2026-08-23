# Thesis Execution Plan — Repository → Defensible Master's Thesis

Date: 2026-08-21. Baseline: thesis-readiness audit verdict **GREEN — START WRITING NOW**
(audit re-verified the seq-00 reproduction baseline exactly: P 0.967 / R 0.777 /
F1 0.862 / mIoU 0.962, TP=1296 FP=44 FN=371).

Repo-verified facts are marked **[V]**; everything else is planning recommendation.

---

## 1. Current state (baseline: GREEN)

- **Complete & established:** detection characterization on full seq 08 with ablations;
  recall bottleneck root-caused (#23) with negative results (#21/#24) and partial fix
  (#34); donor metric built, validated, hardened (#32/#37); completion value
  demonstrated (#29/#32) and replicated held-out under pre-registration (#42); mover
  plausibility (#44); clustering benchmark (#43); timing.
- **Remains:** thesis writing (T14 — the delegate brief's only open task **[V]**). All
  mandatory evidence *experiments* are done (B1/B2/B6 — #47/#48); the remaining
  Section-3 items (B3 literature table, B5 pipeline diagram) are writing/figure work
  bundled into their host chapters.
- **Optional:** distance-stratified recall refresh (B4) — not run; geometric-only
  ablation re-run under promoted config **done** (B6, #47).
- **Frozen from today:** everything in Section 2.

## 2. Research freeze

**Frozen as of today (do not touch):**

| Item | Frozen value | Source **[V]** |
|---|---|---|
| Pipeline config | `PIPELINE_CONFIG` as committed: `voxel_before_denoise=True`, `ransac_iterations=300`, `cluster_voxel_size=0.10`, `clustering_method="hdbscan"` | `src/pipeline.py:10` |
| Checkpoints | `stage_b_scratch_best.pth` (classifier), `pcn_kitti_best.pth` (completion) | `docs/project_state.md` |
| Completion constants | per-car length estimate q90+0.12, fallback 4.14 m, min 5 frames; T13 flags default OFF | `src/completion.py`, Finding #45 |
| Eval protocol | point-level IoU ≥ 0.3 greedy 1-to-1, `supported-vehicles`, track filter ON, micro-averaged | `src/evaluate.py` |
| Authoritative sequences | **seq 08 = headline** (4,071 frames); seq 00 = tuning/replication sequence (state its dual role); seq 05 = fallback-trigger record only | Finding #18, T8/T9 |
| Authoritative completion artifacts | `output/experiments/donor_perf_len{off,on}/`, `completion_box_eval/step2_metrics_08_len*.json`, `docs/plans/t9b_results_heldout_seq00.md`, `output/08` (current regen) + `*_backup/` dirs | verified present |
| Closed experiments | recall repair strategies (#21/#24), Step 1c (#45), OLS fallback (#40), PoinTr (#28), Stage A production use (#31), PCN real fine-tuning direction (#16/#17/#19), merge-centroid fix (#46) | findings log |

**Freeze rule:** no edits to `src/pipeline.py`, `src/completion.py`, `src/classifier.py`,
checkpoints, or `output/08` until after submission. New work goes to `scratchpad/` +
new `output/experiments/` subfolders only (house data-safety rule **[V]**).

**Must run before declaring the freeze complete:** nothing. Tasks B1–B6 below are
read-only analyses against the frozen config — they don't change it, so freeze now
and run them in parallel with writing.

## 3. Remaining experimental tasks

**Status (2026-08-21):** all mandatory *experiments* done — B1 ✅ (#47), B2 ✅ (#48),
B6 ✅ (#47). Remaining: B3 (literature table — writing, Ch 2) and B5 (pipeline diagram
— figure, Ch 3), both bundled into their host chapters; B4 optional, not run.

### B1 — IoU-threshold sensitivity (MANDATORY)
- [x] **Done** 2026-08-21 (#47): F1 0.811/0.808/0.785 at IoU 0.25/0.30/0.50 — stable, PASS. Logs in `output/experiments/iou_sensitivity/`.
- Objective: show the headline is not an artifact of IoU=0.3 (defends examiner Q4).
- Command **[V exists]**: `.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --iou-threshold 0.25`
  and `--iou-threshold 0.5`, stdout redirected to
  `output/experiments/iou_sensitivity/seq08_iou{025,050}.log` (new folder; evaluate.py
  writes only to stdout **[V]**). ~50–60 min each, background. Optionally the same two
  on seq 00 (~4 min each).
- Output → one 3-row table (0.25 / 0.30 / 0.50) in the detection-results chapter.
- Acceptance: F1 at 0.5 within a few points of 0.3 (predicted by matched-IoU 0.96; if
  it *isn't*, that's a finding to report, not tune away).

### B2 — Quantify GT cars excluded by the ≥10-surviving-points rule (MANDATORY)
- [x] **Done** 2026-08-21 (#48): seq 08 stride-20 — 25.5% of all annotated cars (14.6% of ≥10-raw-point cars) excluded by the eligibility rule; implied recall-vs-all-annotated ≈0.54. `scratchpad/gt_eligibility_count.py`, `output/experiments/gt_eligibility/gt_eligibility_08.json`.
- Objective: state recall against *all* annotated cars, not just survivors (defends
  Q9). No existing script does this **[V — checked scratchpad]**.
- Method: new `scratchpad/gt_eligibility_count.py` reusing `get_frame_detections`'s
  preprocessing + `gt_masks` logic (`src/evaluate.py:121-243`) vs. a raw label scan
  (sem ∈ {10,252}, inst>0) per frame. Note the two ≥10-point rules are different
  quantities — project_state already warns about this **[V]**.
- Output: JSON + one sentence ("N% of raw GT car instances per frame are excluded by
  the eligibility rule") → evaluation-protocol section. Seq 08 stride-20 sample is
  sufficient (~10 min run).
- Acceptance: a single defensible percentage with the stride documented.

### B3 — Literature-positioning table (MANDATORY, writing work)
- [ ] **Done**
- Objective: context for "why is F1 0.808 / R 0.73 interesting" (Q2). No experiments —
  a table of published SemanticKITTI panoptic/instance numbers + annotation/compute
  requirements vs. yours, with an explicit "not directly comparable, different
  protocol" caveat.
- Source: literature + `docs/papers/`; lands in Related Work.
- Acceptance: every row cites a paper; no protocol-mismatched number presented as
  head-to-head.

### B4 — Distance-stratified recall refresh (OPTIONAL)
- [ ] **Done**
- Finding #5 is precision-only, seq 00, pre-promoted config (2026-05-14) **[V]**. A
  refreshed table (TP/FP/**FN** by range bin, promoted config, seq 08 sample) supports
  the "effective operating range" issue raised in the proposal §3.4. Inspect
  `src/analyze_fp.py` first — it produced the #4/#5 analyses and likely has the
  distance-binning machinery (internals not verified — flag, not fact).
- Skip if time-pressed: the thesis can cite Finding #5 with its date/config caveat.

### B5 — Pipeline diagram (MANDATORY)
- [ ] **Done**
- Already on the deferred list **[V]**. One block diagram: raw scan → preprocess →
  ground removal → HDBSCAN → geometric filter → classifier → tracker → completion,
  annotated with per-stage ms from
  `output/experiments/timing/timing_seq08_full_n4068.json` **[V exists]**. Lands in
  the pipeline chapter.

### B6 — Geometric-only ablation under promoted config (RECOMMENDED)
- [x] **Done** 2026-08-21 (#47): geom-only P0.149/R0.775/F10.250 (like-for-like); C2 corrected 0.305→0.250. Log in `output/experiments/iou_sensitivity/`.
- Justification: the ablation table's "geometric-only F1 0.305" (#18) was measured
  2026-05-28 under the *pre*-promoted config; the endpoint 0.808 is post-promoted.
  Mixed-config ablation rows are an examiner risk.
- Command **[V flags exist]**:
  `.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000 --no-learned-classifier --no-track-filter > output/experiments/iou_sensitivity/seq08_geomonly.log`
  (~50 min, background). One run makes the whole ablation table like-for-like. If
  skipped, keep the provenance caveat in the caption.

Everything else: **no further experiments.** Each of these six has a named thesis
section and a named examiner question it defends; nothing here authorizes touching
frozen configs.

## 4. Thesis structure

Mapped onto the FPT template (`docs/writing/report-template/main.tex` **[V]**):

| Ch | Sections | Purpose | Evidence | Figures/Tables |
|---|---|---|---|---|
| 1 Introduction | problem, approach, contributions, scope (car-only, offline), thesis outline | Frame + own the proposal drift explicitly | proposal + Section 7 below | pipeline teaser fig |
| 2 Background & Related Work | SemanticKITTI; clustering; PointNet; completion (PCN/PoinTr); positioning table | Context for Q2/Q10 | B3 | Tab: literature positioning |
| 3 Detection Pipeline | preprocessing, ground removal, clustering, geometric filter, classifier (arch + Stage A/B training), tracker | Method | `src/pipeline.py`, `docs/classifier/` | Fig: pipeline diagram (B5); Tab: config |
| 4 Detection Evaluation | **4.1 protocol** (matching, IoU thresh + B1, recall denominator + B2, seq roles/leak history #18), 4.2 seq-08 results, 4.3 ablations (classifier B6, Stage A #25/#30, clustering #43, track filter), 4.4 distance (B4/#5), 4.5 runtime | The honest protocol section is the thesis's load-bearing wall | evaluate.py, #18/#25/#30/#43, timing JSON | Tabs: headline, ablation, clustering, IoU-sensitivity, timing; Figs: seq08_bev_detections, seq08_timeseries |
| 5 The Recall Bottleneck | split analysis #23, filter ablation #22, five negative strategies #21/#24, resolution fix #34, corrected-ceiling narrative | Negative results as contribution | findings #21–24, #34 | Fig: seq08_failure_zooms; Tab: strategy outcomes |
| 6 Completion Method | PCN, domain-gap history #15–19, KITTI-like partials + inference-bug fix #26, L-shape gate #27, canonical frame + length estimate #35/#36, closed Step 1c #45 | Method + the debugging story | `docs/pcn/`, completion findings | Fig: KITTI-like partial illustration; Fig: before/after de-blob (verify_pcn_step2) |
| 7 Completion Evaluation | 7.1 why pseudo-GT CD is invalid #26, **7.2 donor metric** (design, validation gate, #37 guard lesson), 7.3 seq-08 results #29/#32, 7.4 pre-registered held-out replication #42, 7.5 movers #44 | The scientific core | donor_metric.md, t9b/t9c docs | Figs: donor_metric_08, completion_box_overlays_08, t11_movers; Tabs: donor headline, box metric, T9b replication |
| 8 Discussion & Limitations | val-as-test structure, statics-only accuracy, compact d2, empty long band, 23% fallback #41, label-quality assumption, what generalizes | Preempt the committee | audit list | — |
| 9 Conclusion & Future Work | answers to the research questions; future-work list | — | — | — |

## 5. Writing order (dependency-driven)

1. **§4.1 Evaluation protocol** — every other chapter cites it; writing it first
   forces the honest phrasings (mean-IoU-of-matched, denominator, seq roles) to be
   settled once. Needs B1/B2 numbers → start those runs *first*, write around them.
2. **§7.2 Donor metric** — the second protocol anchor; `docs/completion/donor_metric.md`
   is nearly a draft already **[V]**.
3. **Ch 3 + Ch 6 (methods)** — no result dependencies; liftable from docs.
4. **Ch 4 results + Ch 5** — after B1/B2/B6 land.
5. **Ch 7 results + replication** — tables exist; mostly transcription with
   pre-registered wording.
6. **Ch 8 Discussion** — after all results are in prose.
7. **Ch 2 Related work** — parallel-anytime, but final positioning table after Ch 4/7
   numbers are frozen in text.
8. **Ch 1 Introduction** — after the contribution story is stable (post Ch 8).
9. **Ch 9 + Abstract** — last.

## 6. Claims → Evidence map

| # | Claim | Evidence / Experiment | Data / Metric | Fig/Tab | Section | Status & required wording |
|---|---|---|---|---|---|---|
| C1 | Hybrid pipeline reaches P .905/R .730/F1 .808 | full seq-08 eval | seq 08, 4071 fr | Tab 4.1 | 4.2 | **Strong**, but state seq 08 = classifier model-selection set |
| C2 | Classifier is the precision mechanism (geom-only .250→.808 F1, P .149→.905) | #18 ablation + B6 rerun (#47) | seq 08 | Tab 4.2 | 4.3 | Strong; like-for-like under promoted config (B6, #47); #18's .305 is pre-promo, kept as historical. State geom-only recall (.775) > full (.730) |
| C3 | Recall limited by HDBSCAN splitting, not classification | #23 (31–37% split; ceiling matches recall) | seq 00+08 | Tab 5.x | 5 | **Strong** |
| C4 | Post-hoc repairs fail; resolution partially works | #21/#24 negatives; #34 (+.031 R) | seq 08 | Tab 5.y | 5 | Strong; use the *corrected* #24 wording (no "hard limit") |
| C5 | Synthetic pretraining redundant; sim-to-real gap total | #25/#30 | Stage-B val | Tab 4.3 | 4.3 | Strong **negative** — this *replaces* the proposal's two-stage-training contribution |
| C6 | Pseudo-GT Chamfer invalid for real completion eval | #26 (raw partial wins CD on every example) | seq 08 statics | — | 7.1 | **Strong** |
| C7 | Donor metric is a valid real-data completion metric | #32 4-item gate; #37 hardening | seq 08, cov/med-dist/d2 | Fig donor_metric_08 | 7.2 | Strong; present #37 as design lesson, not erratum |
| C8 | Completion adds unseen surface & improves boxes | #29/#32: IoU .707→.747, cov 0→.304 | seq 08, n=39-40 statics | Tabs 7.1-7.2 | 7.3 | Strong, **scoped**: static, gate-passed, compact+normal |
| C9 | The value generalizes held-out | #42 PARTIALLY HOLDS (IoU .739→.766 p=1.6e-3; cov 0→.413) | seq 00, n=45 | Tab 7.3 | 7.4 | **Must use the T9c caveat wording verbatim** (coverage gap; empty long band; compact d2 fail named) |
| C10 | Per-car length estimate fixes compact overshoot | #36 (ΔL +.295→+.063) | seq 08 | Tab 7.2 | 6/7.3 | Careful: donor cost on normals + 23% fallback (#41) stated |
| C11 | Movers complete as plausibly as statics | #44 (57.9% vs 53.7%) | output/08 | Fig t11 | 7.5 | **Plausibility only — never "accuracy"** |
| — | "Two-stage training reduces annotation needs" | refuted by #25/#30 | — | — | — | **REMOVE from claims**; reframe as C5 finding |
| — | Bus/motorcycle support; CD+EMD loss (proposal) | dropped (#20; CD-only training) | — | — | 1 | Reframe in Ch 1 scope: deliberate narrowing, each backed by a finding |

## 7. Final contribution statement

**Primary scientific:**
1. The donor-frame occluded-side coverage metric — a validated, reference-leakage-free
   way to measure real-data completion, including the demonstration that standard
   accumulated-pseudo-GT Chamfer is invalid (#26/#32/#37).
2. Pre-registered, held-out-replicated evidence that PCN completion adds genuinely
   unseen surface and improves amodal boxes on real LiDAR (#29/#42).

**Secondary findings:** recall bottleneck = clustering fragmentation, with
recoverable-ceiling quantification (#23/#34); sim-to-real classifier gap is total and
symmetric, making synthetic pretraining redundant at 420k real clusters (#25/#30).

**Engineering:** the modular pipeline itself, its runtime characterization, amodal-GT
builder, reproducibility infrastructure.

**Negative results (present as findings):** #16/#17/#19 (PCN domain adaptation),
#21/#24 (recall repairs), #28 (PoinTr), #40 (OLS fallback), #45 (Step 1c).

**Limitations:** val-as-test structure; statics-only accuracy validation; compact-band
d2; empty long band; 23% length-estimate fallback; SemanticKITTI label quality assumed.

## 8. Results package

| ID | Title | Source | Ch | Exists? |
|---|---|---|---|---|
| Tab 4.1 | Detection headline (seq 08 full; seq 00 for reference, labeled) | evaluate.py logs / project_state | 4 | ✅ numbers exist |
| Tab 4.2 | Stage ablation: geometric → +classifier → +track filter | #18 (+B6) | 4 | ✅ numbers exist (B6 done, #47) |
| Tab 4.3 | Stage-A ablation + cross-domain matrix | #25/#30, `output/experiments/cross_domain_classifier/` | 4 | ✅ |
| Tab 4.4 | Clustering benchmark + eps sweep | #43, `output/experiments/t10_clustering/` | 4 | ✅ |
| Tab 4.5 | IoU-threshold sensitivity | **B1** | 4 | ✅ numbers exist (#47); drafted in §4.1 |
| Tab 4.6 | Runtime breakdown | timing JSONs | 4 | ✅ |
| Tab 5.1 | Recall-strategy outcomes (5 negatives + #34) | findings | 5 | ✅ (compile) |
| Tab 7.1 | Donor metric headline (raw/mirrored/completed) | #32 / donor_perf_lenon JSON | 7 | ✅ |
| Tab 7.2 | Box metric raw vs completed (+ band split) | #29/#36 JSONs | 7 | ✅ |
| Tab 7.3 | Held-out replication | `docs/plans/t9b_results_heldout_seq00.md` | 7 | ✅ |
| Fig 3.1 | Pipeline diagram (+ per-stage ms) | **B5** | 3 | ❌ generate |
| Fig 4.1/4.2 | seq08_bev_detections / seq08_timeseries | output/figures **[V]** | 4 | ✅ |
| Fig 5.1 | seq08_failure_zooms (split cars) | output/figures | 5 | ✅ |
| Fig 6.1 | KITTI-like partial vs naive render | `output/experiments/fig61_partials/` (fig61_partial_compare.py) | 6 | ✅ rendered 2026-08-23 |
| Fig 6.2 | De-blob before/after (old PCA+partial-r blob vs corrected canonical frame, same ckpt) | `output/experiments/fig62_deblob/` (fig62_deblob.py) | 6 | ✅ rendered 2026-08-23 |
| Fig 7.1 | donor_metric_08 (best→worst panels) | output/figures | 7 | ✅ |
| Fig 7.2 | completion_box_overlays_08 | output/figures | 7 | ✅ |
| Fig 7.3 | t11_mover_completions_bev | output/figures | 7 | ✅ |
| Fig 7.4 | amodal_gt_check (GT construction sanity) | output/08 + output/00 | 7 | ✅ |

~10 tables + ~9 figures — a complete evidence chain. Figure status: Fig 3.1
(B5 pipeline diagram, inline TikZ in Ch 3), Fig 6.1, and Fig 6.2 all produced
2026-08-23. All Section-8 figures now exist as artifacts; remaining figure work is
cosmetic (restyle/caption) during the results-phase figure pass. Tab 4.5 numbers
exist (#47), drafted in §4.1.

## 9. Examiner-defense plan

| Q | Short answer | Evidence | Ref | Don't claim |
|---|---|---|---|---|
| 1. Val-as-test? | Seqs 11–21 unlabeled; structural. Disclosed; completion side gets a true pre-registered replication | #18, prereg | §4.1, §8 | "held-out test set" for detection |
| 2. Why 0.73 recall vs SOTA? | Different problem: interpretable, ~annotation-free, modest hardware; bottleneck root-caused, not mysterious | B3 table, #23 | §2, §5 | competitiveness with end-to-end detectors |
| 3. Amodal GT valid? | Tracklet cross-check proven impossible (404 verified); construction guards + dims sanity + paired design tolerates reference noise | rev2 §6.2 | §7.2 | GT is error-free |
| 4. Why IoU 0.3? | Matched-IoU 0.96; B1 shows stability | B1 table | §4.1 | — |
| 5. Donor coverage the right metric? | Only leakage-free real signal; 4-gate validation; mirrored baseline; hallucination guard | #32 | §7.2 | absolute surface recall |
| 6. d2 fails on compacts — why ship? | 7× hallucination reduction vs mirrored incumbent; band bar effectively unclearable (mirrored ~0.0004); ratios reported | #37, agenda P2 **[V]** | §8 | guard "passes" |
| 7. Priors fitted on eval cars? | Leakage controls (#36) + pre-registered replication whose outcome was reportable either way | #42 | §7.4 | — |
| 8. Movers? | Plausibility parity only (57.9 vs 53.7%); donor metric requires statics by construction | #44 | §7.5 | mover *accuracy* |
| 9. Recall vs ALL cars? | B2 number + both ≥10-rules explained | B2 | §4.1 | — |
| 10. What's yours vs. reimplementation? | Donor metric, split-car root cause, negative-result chain, validation methodology | Section 7 above | §1.3 | pipeline novelty |

**Five to know cold (no thesis in hand):** Q1, Q5, Q6, Q7, Q10 — exactly the ones
`docs/plans/personal_agenda_2026_08_02.md` earmarks for `/quiz-me` **[V]**; that
file's P2 section is the defense syllabus.

## 10. Execution schedule

- [x] **Phase 1 — Research freeze (~30 min).** Declare Section 2 in `project_state.md`
  (one commit). Done when: freeze list committed. — done 2026-08-21
- [x] **Phase 2 — Final verification (~half day active, runs in background).** B1 (2
  runs), B6 (1 run), B2 (script + run), B4 optional. Done when: logs in
  `output/experiments/iou_sensitivity/`, one exclusion percentage, findings-log
  entries via `/note-finding`. — done 2026-08-21: all three mandatory items landed
  (#47 B1/B6, #48 B2); B4 optional, skipped.
- [ ] **Phase 3 — Methodology (multiple days).** §4.1 → §7.2 → Ch 3 → Ch 6, in that
  order; B5 diagram alongside. Done when: advisor-reviewable drafts with every
  phrasing-rule item from `personal_agenda` P1 checked. — **IN PROGRESS:** §4.1 + §7.2
  first drafts done 2026-08-21 (`docs/writing/thesis/`, build clean, committed `e1fabd3`,
  not yet advisor-reviewed); Ch 3, Ch 6, and the B5 diagram remaining.
- [ ] **Phase 4 — Results (multiple days).** Ch 4 + Ch 5 + Ch 7 results sections; tables
  transcribed with per-chapter spot-check against `docs/findings.md` (agenda P1 rule
  **[V]**). Done when: every number traces to a finding/artifact.
- [ ] **Phase 5 — Discussion (~1 day).** Ch 8 from the limitation list. Done when: every
  audit-identified weakness appears *by name*.
- [ ] **Phase 6 — Intro / related work (~2 days).** B3 table, Ch 2, then Ch 1 with
  contribution statement + proposal-drift reconciliation. Done when: Ch 1 claims
  match Section 6's map exactly.
- [ ] **Phase 7 — Finalization (~2 days).** Ch 9, abstract, reference pass, template
  formatting, `/humanizer` pass, full read-through. Done when: PDF builds clean;
  checklist §12 green.
- [ ] **Phase 8 — Defense prep (multiple days, after full draft).** `/quiz-me` per block
  (donor metric, length chain, replication, Step 1c); rehearse the 5 cold questions;
  slide deck. Done when: the pre-registration argument and the d2 defense can be
  reproduced unprompted (agenda P2 standard).

## 11. Parallelism

**Parallel:** B1/B6 background runs while writing §4.1; B2 script while runs execute;
Ch 2 literature reading during any writing phase; B5 diagram anytime; figure
polishing during Phases 4–6; advisor review of chapter N while drafting N+1 (the
agenda's rhythm **[V]**).

**Sequential:** freeze → B-runs → results tables that cite them; §4.1 before any
results chapter; Ch 8 after all results; Ch 1 after Ch 8; abstract last; defense
prep after full draft.

## 12. Definition of "done"

- [ ] **Research:** T1–T13 closed **[V]**; B1/B2 landed; no open experiment.
- [ ] **Evaluation:** every reported number traces to a command + artifact; protocol
  section states matching, thresholds, denominator, sequence roles.
- [ ] **Code/reproducibility:** working tree clean, `main`==`origin/main`; evidence
  scripts tracked (T1 **[V]**); invariant tests green (`src/test_invariants.py`);
  reproduction baseline matches (re-verified 2026-08-21).
- [ ] **Figures/Tables:** all Section-8 items present, captioned with source artifact +
  config provenance.
- [ ] **Literature:** positioning table with citations; PCN/PoinTr/HDBSCAN/SemanticKITTI
  properly cited.
- [ ] **Methodology/Results/Discussion:** phrasing checklist from `personal_agenda` P1
  applied per chapter (that checklist *is* the done-criterion **[V]**).
- [ ] **Limitations:** all six named items present.
- [ ] **Conclusion/Abstract:** claims ⊆ Section-6 map; nothing beyond PARTIALLY-HOLDS
  scope.
- [ ] **References/Formatting:** builds on the FPT template; bibliography complete.
- [ ] **Defense:** 5 cold questions rehearsed; quiz-me passes; deck done.

## 13. What NOT to do

Confirmed against the repo, all with negative-result precedent or closed status **[V]**:

- **No more recall work** — five strategies negative (#21/#24), Step 1c negative
  (#45), T10 says adopt nothing (#43).
- **No length-compensation retuning** — #45 proved it irreducibly length-dependent;
  out of scope by the pre-registration.
- **No PCN real-data fine-tuning / architecture swaps** — #16/#17/#19/#28.
- **No runtime work** — 650 ms accepted by explicit user decision 2026-07-23;
  real-time out of scope per CLAUDE.md.
- **No hunt for a labeled test sequence** — seqs 11–21 have no labels; proven.
- **No refactors** — the #46 lesson: even a *real* inconsistency was negative to fix.
  Also skip backlog items 6–8 (Patchwork++, SORT, venv rebuild) — future work.
- **No third-sequence amodal GT for the long band** — the caveat is named and
  user-adjudicated; days of work for one table cell. Not needed for a defensible
  thesis.
- **Don't optimize metrics further** — F1 0.808 and BEV IoU +0.027 are defended by
  explanation, not by another point of improvement.

## 14. THE PLAN — next 10 actions in order

1. [x] Commit a "research freeze" note into `docs/project_state.md` listing Section 2's
   frozen items (~30 min). — done 2026-08-21
2. [x] Launch B1 in background:
   `evaluate.py --seq 08 --frames 5000 --iou-threshold 0.25` then
   `--iou-threshold 0.5`, stdout → `output/experiments/iou_sensitivity/`
   (~2 h wall, unattended). — done 2026-08-21 (#47).
3. [x] Launch B6 in background: same eval with
   `--no-learned-classifier --no-track-filter` → same folder. — done 2026-08-21 (#47).
4. [x] While those run, write `scratchpad/gt_eligibility_count.py` (B2), run on seq 08
   stride-20, record the exclusion percentage via `/note-finding`. — done 2026-08-21 (#48): 25.5% vs all annotated / 14.6% vs ≥10-raw.
5. [x] Record B1/B6 results as a finding (#47); C2 corrected 0.305→0.250. — done 2026-08-21.
   Table *numbers* locked in #47; LaTeX Tab 4.5 / corrected Tab 4.2 transcribed when Ch 4 is drafted.
6. [x] Draft **§4.1 Evaluation protocol** in the FPT template
   (`docs/writing/report-template/main.tex` copy), applying the mandated phrasings;
   cite B1/B2. — done 2026-08-21 (first draft, committed `e1fabd3`): `docs/writing/thesis/sec_4_1_evaluation_protocol.tex`, builds clean; cites B1(#47)/B2(#48). Not yet advisor-reviewed.
7. [x] Draft **§7.2 Donor metric** from `docs/completion/donor_metric.md` + #37. — done 2026-08-21 (first draft, committed `e1fabd3`): `docs/writing/thesis/sec_7_2_donor_metric.tex`, builds clean; design + 4-item gate + #37 per-band guard lesson. Not yet advisor-reviewed.
8. [x] Draft Ch 3 (pipeline) and Ch 6 (completion method); produce the B5 pipeline
   diagram alongside. — done 2026-08-23 (first drafts, uncommitted):
   `docs/writing/thesis/sec_3_detection_pipeline.tex` (Ch 3; B5 diagram is the
   inline TikZ Fig 3.1, per-stage ms from `timing_seq08_full_n4068.json`; frozen-
   config table; production classifier written as the from-scratch checkpoint #31
   with Stage A framed as Ch-4 ablation) and `sec_6_completion_method.tex` (Ch 6,
   debugging-story structure #15-19→#26→#27→#35/#36→#45; eval deferred to Ch 7).
   Both build clean under TeX Live 2026. **Fig 6.1** (naive vs KITTI-like partial)
   and **Fig 6.2** (de-blob before/after) RENDERED 2026-08-23 and embedded (no
   placeholders left): `scratchpad/fig61_partial_compare.py` →
   `output/experiments/fig61_partials/`; `scratchpad/fig62_deblob.py` →
   `output/experiments/fig62_deblob/` (reconstructs the retired PCA+partial-radius
   normalization for the before/after — freeze-safe, touches nothing frozen; blob
   reproduction verified). **Honest reframing (user critique 2026-08-23):** on
   real sparse clusters the corrected completion is a *diffuse* cloud, NOT a crisp
   car — "blob→clean car" was overclaiming. Fig 6.2 + §6.5 claim only POSE + SCALE
   recovery (old PCA path tilts the car off-level and mis-scales; corrected
   gravity-aligned path sits level at the right extent), crisp completion deferred
   to the in-distribution synthetic result, quantitative real-data win to Ch 7.
   **Fig 6.2 is now a BEV BOX OVERLAY** (`scratchpad/fig62_box_overlay.py` →
   `output/experiments/fig62_deblob/box_overlay_deblob_08.png`), matching the
   `completion_box_overlays_08.png` style the user recalled — GT amodal box (black)
   vs old-completion box (red, mis-scaled/rotated off GT) vs corrected-completion box
   (green, aligned), quantified by BEV IoU vs GT at three densities: **0.32→0.73
   (sparse), 0.25→0.74 (mid), 0.47→0.85 (dense)**. Reuses the box-eval helpers
   (`world_box`/`box_corners_xz`/`bev_iou`) + reconstructed old path; falls through
   step1_records candidates per band (records predate the promoted config). Detection
   uses `stage_b_best` (checkpoint the amodal-GT records were built with; old-vs-
   corrected comparison is classifier-independent). Diffuse-cloud caveat kept in §6.5
   so the box figure isn't read as a crisp-surface claim. **Superseded the earlier
   3D-scatter overlay** (`fig62_deblob.py`, kept; other-sequences check used it — seq
   00 mined = 31 static cars vs seq 08's 15, corroborates, pickle-cached). Ch 6
   rebuilds clean (7 pp). Not yet advisor-reviewed.
9. [ ] Draft Ch 4/5/7 results chapters, transcribing the Section-8 tables, spot-checking
   one number per chapter against `docs/findings.md`.
10. [ ] Draft Ch 8 limitations → then Ch 2 (+B3 table) → Ch 1 → Ch 9 → abstract →
    `/humanizer` pass → Phase 8 defense prep with `/quiz-me`.

### Current status
GREEN; research complete; writing underway. Phases 1–2 done; Phase 3 (methodology)
substantially drafted. All mandatory evidence experiments landed (B1/B2/B6 —
#47/#48). Protocol milestone reached 2026-08-21 (§4.1 + §7.2, committed `e1fabd3`).
**Methods milestone reached 2026-08-23:** Ch 3 (`sec_3_detection_pipeline.tex`) +
Ch 6 (`sec_6_completion_method.tex`) first-drafted and the B5 pipeline diagram
produced (inline TikZ Fig 3.1 in Ch 3). All four method/protocol sections build
clean under TeX Live 2026; §4.1/§7.2 committed, Ch 3/Ch 6 uncommitted, none yet
advisor-reviewed. Ch-6 figures (Fig 6.1/6.2) rendered and embedded 2026-08-23.
Remaining Section-3 item: B3 table (Ch 2); B4 optional.

### Immediate next action
Action 9: draft the results chapters (Ch 4 detection results/ablations, Ch 5 recall
bottleneck, Ch 7 completion results/held-out replication/movers), transcribing the
Section-8 tables and spot-checking one number per chapter against `docs/findings.md`.
Ch 4/7 cite the already-drafted protocol anchors (§4.1, §7.2). Action 8 fully
closed (Fig 6.1/6.2 rendered and embedded 2026-08-23).

### Next milestone
"Methods milestone" — Ch 3 + Ch 6 drafted and the B5 pipeline diagram produced —
**REACHED 2026-08-23** (first drafts, uncommitted, pending advisor review; Fig
6.1/6.2 rendered and embedded). Next: "Results milestone" — Ch 4 + Ch 5 + Ch 7
results sections drafted, every number traced to a finding/artifact.

### Thesis-ready
All Section-12 boxes green; every claim within the Section-6 scoping; PDF builds;
advisor has reviewed all chapters.

### Defense-ready
Thesis-ready + the 5 cold questions answerable unprompted + `/quiz-me` passed per
major block + slides done.

---

**Closing caution:** the freeze is the plan's keystone. Every past incident in this
repo came from touching semantics mid-stream (#26, #38, #39); the corollary in
writing mode is that any "quick improvement" now invalidates committed tables. From
today, the repository is evidence, not a lab.
