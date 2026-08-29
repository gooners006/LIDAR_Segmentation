# Defense Deck — Execution Plan (for a future session)

**Status:** planned, not started. Written 2026-08-30.
**Skill:** organized by `/tweakable-plan` (judgment calls first, mechanical last).
**Adaptation:** this is a presentation, not an experiment, so the usual
"hypothesis / metric expected to improve" header is replaced by a **goal +
success criteria**. Everything else follows the tweakable-plan structure.

## Goal

Produce a thesis-defense slide deck that mirrors the peer template
`docs/LVTN.pptx` (same advisor, Dr. Doan Nhat Quang) and faithfully presents the
thesis's three contributions and its honest hedges, so it can be sent to the
advisor **alongside** the finished manuscript (`docs/writing/thesis/main.pdf`,
79 pp). See memory `defense-deck-baseline`.

## Success criteria (the "metric" for this task)

- Complete deck, title slide → "thanks" slide, in the four-act template structure.
- Every quantitative claim on a slide is traceable to a thesis section or a
  `docs/findings.md` finding (no new numbers invented for the talk).
- The load-bearing hedges survive onto the slides: recall stated against its
  car-only ≥10-surviving-point denominator (Finding #48), completion held-out
  result labelled **PARTIALLY HOLDS** (Finding #42), synthetic-pretraining result
  presented as the negative it is (#25/#30), moving cars = plausibility only (#44).
- Contribution list on the deck matches the manuscript's locked three (donor
  metric / recall root-cause / synthetic redundancy) — the same set in §1.4 / §2.5
  / §9.3. Do **not** reintroduce dropped proposal promises (multi-class, EMD,
  two-stage-as-contribution).

## Key inputs the next session should load first

- `docs/project_state.md` — frozen headline numbers, per-chapter status, the review
  history. **Numbers come from here / findings.md, never re-derived.**
- `THESIS_PLAN.md` — claims map C1–C11 (every slide claim must be ⊆ this).
- `docs/writing/thesis/main.tex` + the `sec_*.tex` chapter files — the prose the
  slides compress. Figures referenced there are the deck's figure pool.
- `docs/LVTN.pptx` — the structure/visual template (peer deck; untracked, local).
- This file.

---

## Template skeleton (extracted from `docs/LVTN.pptx`, 2026-08-30)

63 slides, 16:9 (20 × 11.25 in). Four acts + dividers, with a running section
footer and slide numbers, numbered subsections, roman-numeral divider slides, and
concrete "execution walkthrough" (baselines vs ours) + "result analysis"
(strengths/weaknesses) slides. Macro-structure:

1. **Title** (thesis title, supervisor, presenter + student ID)
2. **Table of contents** (I–IV)
3. **I. Introduction** — Background · Current Approach · Literature Review ·
   Research Gap · Objectives & Contributions
4. **II. Proposed Methodology** — System Architecture · detailed structure
   examples · input processing · the core method · the full pipeline
5. **III. Experiment Evaluation** — Data Preparation · Experiment Setup
   (baselines / metrics / split) · Results (walkthroughs + analysis) · Runtime ·
   Ablation Study (incl. parameter sensitivity)
6. **IV. Conclusion & Future Work** — Research Summary · Limitations · Discussion
   & Future Works
7. **"The end / thanks."**

Our deck follows this spine but maps our own content onto it (below). The peer's 56
content slides are on the high side; we compress (see decision A2).

---

## Section A — Decisions you'll probably want to tweak

Ordered by blast radius. **The next session should confirm A1–A3 before generating
anything** — they change the whole build.

### A1. Build tool / format  — **LOCKED 2026-08-30: (a) python-pptx on a cleaned template copy** (fallback: fresh clean 16:9 theme, then manual)

**This decision hinges on one question: how much of the deck do you want built _for_
you vs. hand-assembled yourself?** An agent can't drive the PowerPoint GUI, so the
more manual the path, the more the actual slide-making falls on you (my role shrinks
to writing the storyboard). Weigh that against engineering risk. **Whatever the base,
start from a COPY and purge every trace of the peer's content** (name, title,
results, any transcript text — see the `defense-deck-baseline` fix-list) so their
deck cannot leak into ours.

- **(a) `python-pptx`, generating onto a cleaned _copy_ of the peer template.**
  *(recommended, with a fallback)* Open `LVTN.pptx`, strip all peer content, and
  bulk-substitute our text + drop in `output/figures/*.png` programmatically —
  inheriting the theme/master/footer for free. Strongest where a deck is mostly
  text-on-standard-layouts + dropped figures; keeps it regenerable and lets me
  produce a real `.pptx` you then polish. *Two real risks, both partly flagged by the
  external review and confirmed here:* (i) python-pptx has **no clean
  slide-duplication API** — cloning styled layouts is an XML hack that can corrupt the
  file; (ii) it is genuinely fiddly for pixel-precise figure/overlay placement, so
  custom-diagram slides still need manual nudging. *Evidence viable:* `python-pptx
  1.0.2` confirmed in `.venv` (2026-08-30).
- **(b) Markdown storyboard → you build every slide manually in PowerPoint.**
  Lowest engineering risk (the external review's pick), but discards agent leverage on
  the build: I produce only the outline; all slide-making is yours. Reasonable if
  you'd rather own the visual assembly end-to-end.
- **(c) Beamer / LaTeX.** Reuses thesis figures directly and is git-tracked, but
  format-mismatches the advisor's `.pptx` ecosystem and is heavier to style. Only if
  the advisor wants LaTeX.

**My read (not flipped on the review's say-so):** its "scripting consumes more time
than it saves" assumes _you_ do everything manually anyway, which isn't the case when
I'm building — so I'm not demoting (a) to fully-manual. But its caution is fair: (a)
is only worth it if the slide-clone step behaves. **Recommendation: attempt (a) on a
cleaned template copy; if the first generated `.pptx` is corrupt or ugly, fall back to
(a)-on-a-fresh-clean-16:9-theme, and only then to (b).**

**Blast radius:** whether the next session writes a Python build script vs. only a
markdown storyboard; how figures get embedded; whether the deck is regenerable.

### A2. Slide budget / talk length  — **LOCKED 2026-08-30: 25–30 min → ~44–48 slides**

User confirmed a **25–30 minute** slot. Target **~44–48 slides total** (~7 non-content:
title, TOC, four dividers, thanks → ~37–41 content), paced ~1.5 min per substantive
slide and faster on dividers/summary. The extra ~5–10 min over a 20-min talk is spent
where the thesis is strongest, not spread thin: (i) the **donor-metric methodology**
gets room to explain _why_ pseudo-GT Chamfer is invalid (a subtle point worth its own
build), and (ii) **completion results** get a baselines-vs-ours walkthrough slide in the
peer template's style. Detection methodology stays brisk (it is the platform, per A3).
Do not pad toward the peer's 56 — backup/appendix slides after "thanks" absorb overflow
(full tables, failure-mode deep-dives). **Authoritative slide list: `docs/defense/storyboard.md`.**
**Blast radius:** number of results/ablation/walkthrough slides per contribution.

### A3. Narrative spine — detection vs completion emphasis  *(which contribution leads)*

The thesis has three contributions; the **title itself is completion-centric**
("A Donor-Frame Coverage Metric …").

- **(a) Completion-forward.** *(recommended)* Donor metric is the headline; the
  detection pipeline is framed as the platform that produces completion inputs, and
  the recall root-cause is the second contribution. *Evidence:* §8.1 and
  `project_state.md` explicitly call the donor metric + completion value the
  **strongest** evidence and detection "an engineering result, not SOTA." The title
  commits to this.
- **(b) Balanced two-act** — detection results then completion results, equal weight
  (mirrors chapter order). Safer/more conventional but buries the strongest claim.
- **(c) Detection-forward** — lead with pipeline + recall. Contradicts the title
  and §8.1; not recommended.

**Blast radius:** ordering of Part III (Results); which contribution gets the
"execution walkthrough" slide(s); what the Objectives & Contributions slide leads with.

### A4. How prominently to show limitations  *(defense posture)*

- **Own it** *(recommended)* — a compact "Limitations" slide in Part IV mirroring
  Ch 8, plus inline hedges on the result slides (recall denominator; PARTIALLY
  HOLDS badge on the held-out slide). *Evidence:* the whole thesis (Ch 8,
  project_state review rounds) is built on preempting committee critiques; the deck
  should match that posture, and it's what the external-review-pushback discipline
  expects.
- Fold hedges only inline, no dedicated slide — leaner but weaker against "what are
  the limitations?" as an opener question.

**Blast radius:** 1–2 slides + caption wording on result slides.

### A5. Figure sourcing

- **Reuse existing thesis figures; export the one vector figure once.**
  *(recommended)* Most figures are already raster renders in `output/figures/*.png`
  (gitignored, regenerable). The pipeline diagram (Fig 3.1) is inline **TikZ** —
  export it to PNG once for the deck. Regenerate a figure only if it's unreadable at
  20-in slide width.
- Regenerate all figures at slide aspect/resolution — cleaner, more work; do only
  for any figure that looks bad scaled up.

**Figure pool (from the chapters):** Fig 3.1 pipeline (TikZ → export); Fig 5.1
`seq08_failure_zooms.png` (fragmentation root-cause); Fig 6.1–6.3 completion
(partial-vs-completed, box overlay); Fig 7.1–7.3 completion results;
`output/figures/seq08_bev_detections.png` / `_timeseries.png` for detection.
**Blast radius:** whether the next session re-runs any `scratchpad/*.py` figure scripts.

---

## Section B — Build sequence (once A1–A3 are locked)

1. **Storyboard first, in markdown** (`docs/defense/storyboard.md`): the full
   slide-by-slide list with, per slide, the one-line message, the source
   (§/Finding), and the figure. Get user sign-off on the storyboard **before**
   generating the pptx — cheap to change here, expensive later. A proposed starting
   storyboard is in the appendix below.
2. **Sub-decision at storyboard time:** clone the peer deck's visual theme
   (colors/master/footer) or start a clean 16:9 theme. Cloning gives instant
   advisor-familiar styling; clean avoids inheriting the peer's content artifacts.
3. **Assemble figures** — export the TikZ pipeline diagram to PNG; collect the
   `output/figures/*.png` pool; note any missing render to regenerate (A5).
4. **Generate the deck** via the A1 tool; write **speaker notes** on every content
   slide (the notes are also the user's script for anticipating committee Qs).
5. **Verification pass** (the success-criteria check): opens in PowerPoint; slide
   count matches the budget; every figure embedded; footer + numbering correct; no
   text overflow; and a claims audit — each numeric claim traced to §/Finding, each
   required hedge present (recall denominator, PARTIALLY HOLDS, negatives-as-findings).
6. **Export a PDF** of the deck for emailing to the advisor with the manuscript.

## Section C — Mechanical work (skippable — do not spend judgment here)

- Title-page metadata fill: title = "A Donor-Frame Coverage Metric for
  Reference-Free Evaluation of Occluded-Vehicle Completion in Automotive LiDAR";
  presenter = Ngo Vi Viet Anh, ID MSE13205; advisor = Dr. Doan Nhat Quang.
- Section-divider slides (I–IV) + running section footer + slide numbers.
- Color theme / font selection; TOC slide list.
- PDF export step; file naming for the advisor email.

---

## Appendix — Proposed starting storyboard (subject to A2/A3)

Assumes A3 = completion-forward, A2 ≈ 40 slides. Adjust once locked.

- **1** Title. **2** Table of contents.
- **I. Introduction (~6):** Background (annotation cost of end-to-end LiDAR seg) ·
  Approach (modular near-annotation-free hybrid; tracker = filter, not a
  contribution) · the completion measurement problem (no clean real GT) · Research
  gap · Objectives + the **three research questions** · **Contributions** (lead with
  the donor metric).
- **II. Proposed Methodology (~9):** Pipeline overview (Fig 3.1) · ground removal +
  HDBSCAN clustering · geometric filter + binary classifier (two-stage training as
  ablation, not a claim) · tracker as temporal filter · completion: the KITTI-like
  partial + inference-normalization fix (the #26 debugging story, briefly) · the
  **donor-frame coverage metric** (definition + why pseudo-GT Chamfer is invalid,
  #26) — this is the methodological centerpiece, give it room.
- **III. Experiment Evaluation (~16):** Setup (dataset = SemanticKITTI, seq 08
  operating point; eval protocol / point-IoU 0.3 / recall denominator) · **Detection
  results** (headline P/R/F1 table; distance breakdown) · **LOSO cross-validation —
  DEDICATED, emphasized slide** (the validation-as-test defense, hit it before the
  committee can: leakage-free fold-08 recall 0.737 vs shipped 0.730, pooled 0.719
  across all eleven sequences, per-sequence spread; Finding #51) · **Recall
  bottleneck** (fragmentation root-cause Fig 5.1; the chain of negative repairs — one
  slide) · **Completion results** (coverage 0.000→0.304 above mirror; amodal-box gain
  on static gate-passed cars; a completion walkthrough figure) · **Held-out
  replication** (seq 00, PARTIALLY HOLDS badge) · Runtime (623.5 ms / 1.6 fps, offline
  scope) · Ablations (synthetic-pretraining redundancy #25/#30; geometric-only #47).
- **IV. Conclusion & Future Work (~5):** Research summary · answers to the three RQs
  (one slide, tight) · **Limitations** (own it — lead with the three most likely to be
  probed: (1) the 23% length-fallback to the fixed 4.14 m prior, which reintroduces
  compact-car hallucination, #37/#41; (2) the architectural recall ceiling — density
  clustering has no objectness, #34/#43; (3) operational-domain brittleness on sparse
  highway scenes — LOSO seq-01 recall 0.336, #51. Ch 8 names **six** in total; keep
  the other three — amodal-GT unvalidatable #48, movers plausibility-only #44,
  completion sim-to-real gap #26 — in speaker notes for deeper Q&A) · Future work
  (real-time variant; completion fine-tuning; long-band held-out GT) · Contributions
  restated.
- **Final:** Thanks / Q&A.

## Related

- Memory: `defense-deck-baseline` (template pointer), `thesis-plan-authority`,
  `external-review-pushback`.
- After the deck: manuscript + deck go to the advisor together (Phase 7 review),
  then Phase 8 defense rehearsal.
