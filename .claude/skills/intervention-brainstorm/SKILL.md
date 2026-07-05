---
name: intervention-brainstorm
description: >
  Brainstorm ~10 codebase-grounded interventions ranked cheapest to most ambitious
  when a metric has stalled, a direction is blocked, or the user is choosing what to
  try next. Use when the user says a metric "plateaued", "is stuck", "won't improve",
  asks "what should we try next", "what are our options", or wants to open a new
  research direction.
---

# Intervention Brainstorm Skill

A metric has stalled or a direction needs options. Produce ~10 candidate
interventions, **ranked cheapest to most ambitious**, each grounded in what actually
exists in this codebase — not hypothetical solutions.

**Precedent:** the recall-ceiling investigation (Findings #21–24) did this serially
over two sessions. The point of this skill is to front-load the enumeration so the
user picks from the full option space before any single bet is made.

## Steps

1. **Restate the problem quantitatively.** Which metric, current value, on which
   sequence/split, and what the relevant findings already ruled out. Check
   `docs/findings.md` and `docs/project_state.md` first — do not propose
   interventions that prior findings already killed (e.g., clustering-level recall
   fixes, Findings #21–24) unless you explicitly say why the situation changed.
2. **Search the codebase** for existing infrastructure each idea could reuse:
   disabled flags, CLI toggles, alternative implementations already in
   `src/pipeline.py`, scratchpad tools, existing checkpoints.
3. **Produce ~10 interventions**, cheapest → most ambitious. For each:
   - **Name** and one-sentence mechanism.
   - **Evidence from the repo:** file/flag/checkpoint that makes it cheap, or the
     gap that makes it expensive.
   - **Expected effect:** which metric moves, rough magnitude if estimable, and the
     tradeoff (e.g., recall up / precision down = possible over-segmentation).
   - **Cost:** hours vs. days vs. retraining-required.
   - **Risk / prior art:** related finding numbers, known failure modes.
4. **End with a recommendation:** the 1–2 you would try first and why, framed as a
   testable hypothesis per the Experiment Protocol.

## Rules

- Ground every item in a real code location or dataset fact — cite paths.
- Include at least one "flip an existing switch" item and at least one ambitious
  item, so the cost spectrum is real.
- Do not start implementing. The deliverable is the option map; the user picks.
- If the user picks one, run it through the Experiment Protocol (hypothesis,
  baseline, single change, evaluate, record) and consider `/tweakable-plan` if it
  has open design decisions.
