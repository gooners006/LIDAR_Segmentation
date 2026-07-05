---
name: tweakable-plan
description: >
  Write an experiment or feature plan organized by likelihood-of-tweaking instead of
  execution order: judgment calls with alternatives first, mechanical work last. Use
  this skill when the user asks to "plan", "design", or "spec" an experiment, metric,
  or feature that has open design decisions — e.g., a new evaluation metric, a
  training-data change, a pipeline stage redesign.
---

# Tweakable Plan Skill

You are planning an experiment or feature for this thesis project. Do NOT organize
the plan by execution sequence. Organize it by **how likely the user is to want to
change each item**: high-judgment decisions at the top, mechanical plumbing collapsed
at the bottom. The user's attention should land exactly where their input changes the
outcome.

## Structure

### Section A — Decisions you'll probably want to tweak

For each judgment call:

- **The decision** in one sentence.
- **Options** (2–3), each with a one-line tradeoff. Mark your recommended option and
  why — grounded in this codebase and prior findings (`docs/findings.md`), not
  generic reasoning.
- **What changes downstream** if the user picks differently (blast radius).

Order these by blast radius: decisions that would change the architecture or
invalidate results first.

Typical high-judgment areas in this project: metric definitions (what counts as a
match, Chamfer direction, thresholds), train/eval data splits, normalization and
coordinate choices, GT construction rules, filtering criteria.

### Section B — Build sequence

The execution order once Section A is locked. Short, numbered, with file paths.
Include the evaluation command that will judge the result (Experiment Protocol:
hypothesis, expected metric, baseline before, compare after).

### Section C — Mechanical work (safe to ignore)

Plumbing, I/O, config wiring, plotting boilerplate. One bullet each. Explicitly label
this section as skippable.

## Rules

- State the **hypothesis** and the **metric expected to improve** at the top of the
  plan (Experiment Protocol requirement).
- Every option recommendation must cite evidence: a finding number, a prior result,
  or a code location — not taste.
- Keep the plan in the conversation for small experiments; save to
  `docs/<topic>/plan.md` (or extend the existing plan file) for multi-session
  directions.
- After the user resolves Section A, restate the locked decisions in one short list
  before starting implementation.
