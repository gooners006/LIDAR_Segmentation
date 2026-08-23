# Delegate brief — review follow-up tasks (2026-08-02, rev 2: max delegation)

Audience: the executing Claude model (not the user). Source: full-repo review
2026-08-02 (research idea / correctness / direction). Read
`docs/project_state.md` first per CLAUDE.md.

**Delegation mode:** the user supervises ONLY thesis-chapter review and defense
prep. Everything below is delegated, including decisions that were previously
user checkpoints — each now has a stated default and a STOP condition instead.
The two remaining user contact points: (1) approving the held-out
pre-registration (T9a, ~5 min — hard gate unless explicitly waived), (2)
responding to STOP escalations.

**Why these guardrails:** this repo's past failures were convention traps,
unverified claims, and one motivated-reasoning flip (Findings #26, #37, #38,
#39; session log 2026-08-02). Capability is not the bottleneck; discipline is.
General rules for all tasks:

- Reproduction baseline (measured 2026-08-02, committed code, default config):
  `.venv\Scripts\python.exe src/evaluate.py` → **P 0.967 / R 0.777 / F1 0.862 /
  mIoU 0.962, TP=1296 FP=44 FN=371**. Any task touching `src/` must re-run
  this and match EXACTLY before claiming done.
- Frame conventions: sensor→global is `s @ Rᵀ + t`; global→sensor is
  `(g − t) @ R`; donor-cache `data["raw"]` is SENSOR frame (Finding #39). Never
  write a new transform without citing which rule it instantiates.
- Data safety: never overwrite `output/*`; use `--out-tag`, verify, swap by
  rename, keep the old dir as `*_backup/` (house convention, see #35 follow-on).
- Deviations: conservative option, log under a **Deviations** heading, continue
  (CLAUDE.md experiment protocol).
- No task authorizes tuning `PIPELINE_CONFIG` or completion constants except
  T13, under its own pre-registered criteria.

## Tier 1 — hygiene (model: **Sonnet 5**; one session, ~1–2 h active)

### T1. Track the evidence scripts (highest priority)
`scratchpad/` is gitignored (41 files, 0 tracked); every script cited in
Findings #29/#32/#35/#36/#37/#40 is unversioned, including the d2 guard fix.
**Decision delegated, default chosen:** edit `.gitignore` to `scratchpad/*` +
`!scratchpad/*.py`; commit all scripts as-is in one "freeze evidence scripts"
commit. Do not move or edit the scripts. Outputs stay ignored.

### T2. `evaluate.py` CLI drift
- `--target` help says "all-things (default)"; actual default is
  `supported-vehicles`. Fix the text, not the default.
- `--ransac-iterations` help says "default 1000"; actual default is 300.
- `--voxel-before-denoise` is `store_true` but the config default is already
  `True` (no-op, no off-switch). Add an explicit `--no-voxel-before-denoise`
  so pre-#34 configs are reachable.
Gate: reproduction baseline exact.

### T3. Deduplicate `resolve_track_class`
Identical function in `src/main.py:27` and `src/evaluate.py:51`. Move one copy
to a shared location, import in both. Behavior-preserving only. Gate:
reproduction baseline exact.

### T4. Invariant tests (`src/test_invariants.py` or `tests/`)
~50–80 lines of pytest locking down the historically violated invariants:
- sensor↔global round-trip: random orthonormal R, t, points; assert
  `((s @ R.T + t) - t) @ R ≈ s`.
- `estimate_canonical_frame` is pure (same input → same output) and the length
  push is extend-only (displacement sign = `sign(center[2])`, magnitude ≥ 0).
- `track_length_estimate`: empty → fallback; < min_frames → fallback;
  fallback=None → None; ≥ min_frames → percentile+offset.
- `complete()` order-independence with `sample_seed` set (skipif when
  `checkpoints/pcn_kitti_best.pth` absent): warmed vs fresh completer, same
  cluster → identical output.
Gate: pytest green; no production-code changes for the tests' sake.

### T5. Doc consistency (markdown only — do NOT edit .docx files)
- `docs/project_state.md`: fill the "TP/FP/FN not separately recorded" gap
  with TP=1296 FP=44 FN=371 (measured 2026-08-02).
- Unify GT-eligibility phrasing in `.md` docs: "≥10 points surviving
  preprocessing (z-filter, voxel, denoise, ground removal)"; note that
  `project_state.md:383`'s "≥10 raw pts" is the label-scan car count, a
  different quantity. Flag, don't rewrite, anything ambiguous.

### T6. Fallback-frequency instrumentation (Finding #40 loose end)
In `src/main.py`, where `length_est = track_length_estimate(fit_lengths)` is
computed, add to the track's `tracks.json` entry:
`"length_estimate_source": "track_q90" | "fallback"` (fallback ⇔
`len(fit_lengths) < COMPLETION_LENGTH_MIN_FRAMES`) and
`"n_gate_passed_frames": len(fit_lengths)`. No behavior change. Gate:
reproduction baseline exact; grep-verify `evaluate.py` still never imports
completion.

### T7. Regenerate `output/08` (AFTER T6, so metadata is captured)
Current `output/08` predates the per-car length estimate and the #38 RNG fix.
- `.venv\Scripts\python.exe src/main.py --seq 08 --frames 5000 --no-gui
  --save-output --out-tag _regen` (**~60–90 min**, basis 650 ms/frame × 4071 +
  completion; run in background).
- Verify vs current `output/08`: track set and `_partial.ply` inputs identical
  (md5, as in #35 follow-on); only completed clouds + tracks.json metadata may
  differ.
- **Swap authorized** (decision delegated): rename old to
  `output/08_fixedprior_backup/`, promote `_regen`, copy GT artifacts in.
- Report the fallback-frequency count (answers #40 caveat (b)). **Decision
  rule (delegated):** if > ~5% of completed tracks used the 4.14 fallback,
  log it as a live limitation in findings — do NOT implement a fix.
STOP if md5 shows detection/tracking artifacts changed — that is a regression,
not RNG/prior noise. Report, do not swap.

## Tier 2 — held-out replication (model: **Opus**)

### T8. Sequence selection + amodal GT (decision delegated)
- **Default: seq 05** (fully fresh in the completion context); fallback rule:
  if < 15 well-observed static cars survive the guards, use **seq 00**; if
  that also fails, STOP and escalate. Constraint: never 08 (all completion
  constants tuned there). Log the confound either way: every non-08 sequence
  is in the classifier's Stage-B training split, so detection reads
  optimistic; the completion claims survive because they are paired
  (raw vs completed on the same TP inputs).
- Run `scratchpad/amodal_gt.py` + `amodal_gt_viz.py` on the chosen sequence
  with the SAME guards as seq 08 (moving-ID rejection, face support, overhang
  flag). **~30–60 min.**
- Sanity gate: median dims near published car stats (seq-08 reference:
  L 4.14 / W 1.75 / H 1.47 m); visual check renders sane boxes.

### T9a. Pre-registration draft — THE one hard user gate
Before any T9 eval runs: draft ~5 lines into
`docs/plans/preregistration_heldout.md` — expected outcomes (completed beats
raw on BEV IoU and |ΔW|/|ΔH|, per-car medians, Wilcoxon p < .05; donor
cov@0.1 completed ≫ mirrored ≫ raw; magnitudes may shrink vs 08) and explicit
refutation criteria (completed fails to beat raw on BEV IoU OR donor coverage
→ "completion adds value" does not generalize, thesis scopes it to seq 08;
d2 band violations worse than seq-08 levels → length constants are
08-specific, report as limitation, do not retune). **Send to the user for
approval and WAIT.** Rationale (do not skip): the pre-registration defines
what the thesis will claim; the executor must not author its own grading key
unreviewed. If the user has explicitly waived this checkpoint, commit the
dated draft first (immutable timestamp), then proceed.

### T9b. Frozen-config evals
Run the #29 box eval and #32 donor metric (steps 1–3 + d2) against the new
amodal GT: production config, production checkpoints, shipped completion
(per-car length estimate ON). **No tuning of any kind.** **~1–3 h** (PCN
inference over all TP pairs; scale from seq-08 runs). Deliverables: the
per-band tables of Findings #36/#37 (raw vs completed; compact/normal/long;
d2 ratios), n_cars/n_pairs.

### T9c. Arbitration — fresh session, not the executor
A **new session** (Opus; not the T9b session — separation of executor and
judge, given the documented yes-man incident) reads ONLY the committed
pre-registration + T9b tables and writes the finding draft: which
pre-registered outcome obtained, verbatim criteria applied. Any result the
pre-registration does not cover → STOP, escalate to the user with the tables.
No constant changes in any branch.

### T10. Clustering benchmark (backlog #5; model: **Sonnet 5**; independent)
HDBSCAN (production) vs DBSCAN (`open3d cluster_dbscan`) vs Euclidean, seq 00
100 frames + seq 08 stride sample, standard `evaluate.py` metrics + per-frame
runtime. Hold `cluster_voxel_size=0.10` fixed across methods. Thesis table;
adopt nothing.

### T11. Moving-car plausibility check (model: **Sonnet 5**; after T7)
Validation covered statics only; production completes movers too. On the
regenerated `output/08`: identify completed moving-car tracks, compute the
plausible-car-box rate (dims L∈[3.3,4.9], W∈[1.5,2.1], H∈[1.1,1.7] — the #27
recipe in `docs/completion/next_ideas.md`), render a small BEV figure
(remember output world frame is Y-down, #26). Report mover rate vs static
rate. Numbers + figure only; no fixes.

## Tier 3 — research increments (model: **Opus**; only after T9c lands and
## only if its outcome is "holds" or "partially holds")

### T13. Step 1c — residual under-extension (highest-risk delegation)
Full plan already scoped in `docs/completion/plan.md` (Step 1c). Rules:
- Start with `/tweakable-plan`; commit the plan (dated) before implementing.
- Pre-register metrics before any run: band-split #29 box metric + #32 donor
  + per-band d2, on BOTH seq 08 and the T8 held-out set.
- Order as plan.md states: (1) decouple `radius` from the center push,
  (2) calibrate fill factor on synthetic true GT (large n) and apply — not
  refit — on real. A length-dependent target (option 3) stays out of scope.
- Ship criteria: improves the band-split box metric with no per-band donor or
  d2 regression, on both sequences. Anything else → record as a negative
  result (house precedent #16/#17/#19/#40) and stop.

### T14. Thesis chapter drafts (model: **Opus or Fable**; `/latex-report` +
### `/humanizer`; chapters delivered one at a time for user review)
The user supervises this tier personally — deliver drafts, never send
anything external. Skeleton (liftable from `results_overview_2026_07_23.docx`,
`results_report_2026_07_17_rev2.docx`, findings):
1. Pipeline + detection results (promoted config; timing section exists).
2. Recall ceiling: root cause (#23), negative results (#21/#24), the
   resolution-artifact correction (#34) — negative results as contribution.
3. Donor-frame occluded-side metric (#32) as the methodology core; include
   #37 as a metric-design lesson, not an erratum.
4. Completion geometry (#26/#27/#35/#36) + held-out replication (after T9c).
5. Limitations: statics-only validation vs movers (T11 result), in-sample
   calibration (mitigated by T9), ~0.73 recall accepted.
Mandatory phrasing fixes from the review: "mean IoU of matched detections";
recall denominator "GT instances with ≥10 points surviving preprocessing";
completion claims scoped per the T9c outcome. Chapters 1–3 can draft
immediately; 4–5 wait for T9c/T11.

## Sequencing
T1–T6 (one Sonnet session) → T7 background → T8 → T9a (**user gate**) →
T9b background → T9c fresh session → T10/T11 anytime after their deps →
T14 ch. 1–3 in parallel from the start → T13 last, conditional.

## Model selection rationale
- **Sonnet 5** (T1–T7, T10, T11): well-specified, mechanically verifiable,
  exact-match gates. Haiku not recommended for anything touching `src/` —
  the repo's failure history is convention traps; a silent semantics slip
  costs more than the model-cost savings.
- **Opus** (T8, T9a/b, T13): long multi-step runs through
  frame-convention-hazardous territory; T13 additionally needs research-design
  judgment under pre-registered constraints.
- **Fresh Opus session** for T9c: executor/judge separation.
- **Opus or Fable** (T14): long-form writing quality; user reviews every
  chapter.
