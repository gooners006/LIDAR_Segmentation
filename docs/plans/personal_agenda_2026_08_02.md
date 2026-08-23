# Personal agenda — thesis supervision & defense prep (2026-08-02, rev 2)

Companion to `delegate_brief_2026_08_02.md`. Per your decision, everything
else is delegated; this file is only what stays with you. One consequence to
keep in view: **delegation moves your burden to the defense, it doesn't remove
it** — every delegated decision (held-out sequence, pre-registration criteria,
Step 1c design, ship/no-ship calls) is one you must defend as your own. That
is what P2 is for.

## P0. Two irreducible micro-checkpoints (not thesis work, but only you can)

1. **Approve the pre-registration** (T9a; ~5 min, one message). The delegate
   drafts it; you read and approve — or explicitly waive, in which case it
   proceeds from a committed dated draft. This is the line between "held-out
   evidence" and "the executor grading its own homework".
2. **Respond to STOP escalations.** The brief instructs delegates to stop and
   report rather than improvise (regressions in T7's md5 check, < 15 cars in
   T8, results outside the pre-registration in T9c). These arrive rarely; when
   they do, they're genuinely yours.

## P1. Thesis writing — supervision loop (chapters drafted by delegate, T14)

Per-chapter review checklist (the delegate has the phrasing rules; you check
they were actually applied, and that the voice is yours):

- [ ] Claims match the evidence tier: measured vs inferred vs speculative
      (your global Evidence rule — now also a thesis rule).
- [ ] "Mean IoU" always qualified as *of matched detections*.
- [ ] Recall denominator stated: GT instances with ≥10 points surviving
      preprocessing.
- [ ] Completion claims scoped per the T9c outcome (generalizes / scoped to
      seq 08), and to static cars unless T11's mover rate supports more.
- [ ] Negative results (#16/#17/#19/#21/#24/#40) presented as findings, not
      apologies; #37 presented as a metric-design lesson.
- [ ] Every number traceable to a finding/output file — spot-check one per
      chapter against `docs/findings.md`.
- [ ] Nothing you couldn't re-derive at the whiteboard survives the read.

Rhythm suggestion: chapters 1–3 can arrive while Tier 2 runs; review one
chapter per sitting, return margin comments to the delegate, don't wordsmith
inline (that's its job).

## P2. Defense prep (after chapter drafts exist)

Run `/quiz-me` per major delegated block — this is now load-bearing, because
the committee will not distinguish delegated design from yours:

- Donor metric (#32) + guard defect (#37): why pooled medians hide
  band-localized failure; why the compact d2 bar is effectively unclearable
  (mirrored baseline ~0.0004) and you report ratios instead.
- Length-prior chain (#35 → #36 → #40): why q90 not max, why the OLS fallback
  was rejected, why the leakage control matters.
- Held-out replication: why THAT sequence, what was pre-registered, what the
  outcome licenses you to claim (read `preregistration_heldout.md` + the T9c
  finding until you can reproduce the argument unprompted).
- Step 1c (if T13 ran): the radius/center-push coupling, why fill factor is
  calibrated on synthetic and applied on real.

Two committee questions to have crisp (answers already exist in the record):

1. *"Your d2 guard fails on the compact band — why did you ship?"* →
   7× hallucination reduction vs the incumbent, box metric improves, the band
   bar is effectively unclearable, and you report the ratio (#37).
2. *"Your priors were fitted on the cars you evaluate on."* → LOO for the
   offset, the leakage-free OLS control (#36), and the held-out replication —
   whose outcome you can state either way because it was pre-registered.

## Suggested calendar shape

Week 1: P0.1 (approve pre-registration) → review T14 ch. 1–2.
Week 2: review ch. 3; T9c lands → skim its finding (10 min, it's mechanical
against your approved criteria); ch. 4–5 get drafted.
Week 3+: review ch. 4–5, then P2. T13, if it happens, only feeds a
subsection of ch. 4 — do not let it block writing.
