# CLAUDE.md

## Project Context

This repository is for a master's thesis research project on LiDAR point-cloud
processing using KITTI-style sequences: object segmentation, tracking/output
extraction, point-cloud completion, and evaluation against ground-truth labels.
The current focus is tracked in `docs/project_state.md`.

This is research code. Prioritize clarity, reproducibility, and easy experimentation over heavy abstraction.

## Start of Session

At the start of each session:

1. Read `docs/project_state.md` for the current context and immediate next steps.
2. Do not read `docs/session_history.md` unless explicitly requested.
3. Check relevant files in `docs/` before asking the user for background context.

Important docs:

- `docs/project_state.md` — current architecture, blockers, next steps (read this first)
- `docs/session_history.md` — chronological session diary (read only if asked)
- `docs/datasets.md` — dataset layout and assumptions
- `docs/findings.md` — experiment observations and conclusions
- `docs/pcn/` — PCN-related notes and references
- `docs/completion/plan.md` — active completion roadmap and step plan

## Common Commands

**Always use `.venv` to run any Python script.** Either activate the environment
first, or invoke the venv interpreter directly (e.g.
`.venv\Scripts\python.exe src/main.py` on Windows,
`.venv/bin/python src/main.py` on Linux/macOS). Do not run scripts with a bare
system `python`/`python3`.

Common evaluation commands:

```bash
# Headline baseline: seq 00, 100 frames (defaults; deterministic)
.venv\Scripts\python.exe src/evaluate.py
# Generalization check: seq 08, full sequence
.venv\Scripts\python.exe src/evaluate.py --seq 08 --frames 5000
# Run pipeline headless with saved outputs
.venv\Scripts\python.exe src/main.py --seq 00 --no-gui --save-output
```

## Training Strategy

Classifier uses two-stage training:
- **Stage A:** Train on synthetic ShapeNet partial renders
- **Stage B:** Fine-tune on real mined LiDAR clusters from SemanticKITTI

The classifier is binary (car / not-car). Current checkpoints and metrics: see
`docs/project_state.md`.

Checkpoints are saved in `checkpoints/`. Training logs are CSV files alongside checkpoints.

## Experiment Protocol

When testing any change, such as a new parameter, algorithm, or pipeline step:

1. State the hypothesis.
2. State which metric is expected to improve.
3. Run `evaluate.py` before the change to record baseline results.
4. Make only the intended change. If an edge case forces a departure from the
   stated change mid-run, take the conservative option, log it under a
   **Deviations** heading (what the plan said vs. what the code/data revealed),
   and continue.
5. Run `evaluate.py` after the change.
6. Compare:
   - Precision
   - Recall
   - F1
   - meanIoU
7. Record:
   - what changed
   - command used
   - before/after metrics
   - whether the change helped
   - any visible failure cases
   - any deviations from the intended change (from step 4)

Do not treat a change as successful without evaluation results.
When metrics conflict, explain the tradeoff (e.g., higher recall with much
lower precision may indicate over-segmentation).

## Integrating External Methods or Code

Before porting, adapting, or integrating any external method, library, or data
format (e.g., Patchwork++, SORT-style tracking, KITTI raw tracklets, a paper's
reference implementation), use the `/semantics-map` skill: produce a reviewable
semantics map (coordinate conventions, normalization, units, what is preserved
vs. changed vs. dropped) and get explicit user sign-off **before writing any
integration code**. Finding #26 (PCN inference-normalization bug) is the
motivating failure: a silent train/inference semantics mismatch cost multiple
sessions.

## Workflow Skills

Situational skills — invoke them when the situation matches, without waiting to
be asked:

- `/semantics-map` — mandatory before external integrations (see above).
- `/tweakable-plan` — when planning an experiment or feature with open design
  decisions: lead with the judgment calls, bury mechanical work.
- `/intervention-brainstorm` — when a metric stalls or a research direction
  needs options: 10 codebase-grounded interventions, cheapest to most ambitious.
- `/quiz-me` — after substantial Claude-written changes land: comprehension
  quiz so the user can defend the design at their thesis defense.

## Coding Conventions

- Keep all pipeline parameters in `PIPELINE_CONFIG` in `src/pipeline.py`.
- Global transforms must use:

```python
poses[i] @ Tr
```

Do not use raw poses directly for global transformations.

- Open3D visualization must stay behind the `--no-gui` flag.
- Do not create Open3D windows at module import time.
- Saved object outputs should follow this structure:

```text
output/<seq>/objects/<track_id>.ply
output/<seq>/tracks.json
```

- Prefer readable, explicit code over premature abstraction; avoid large
  refactors; keep changes small and easy to compare experimentally.

## Data and Output Safety

- Do not overwrite or delete existing outputs, results, or notebooks without
  permission; preserve previous results when comparing experiments.
- Use distinct output folders when testing new configurations (`main.py --out-tag`).

## Known Failure Modes

- Small or distant objects may be under-segmented.
- Adjacent vehicles may be merged into one cluster.
- Sparse LiDAR points can reduce IoU even when the visual result looks acceptable.
- Parameter changes may improve one sequence but hurt another.

## Agent Behavior

Before editing code:

1. Inspect the relevant files.
2. Explain the intended change and its reasoning for exploratory/risky changes.
3. Keep the diff minimal.
4. Update `docs/project_state.md` if the change affects project direction or results.

When asking for approval to run a long/background task, include a time estimate.

## Out of Scope

The following are out of scope unless explicitly requested:

- real-time execution
- SLAM or map optimization
- camera-LiDAR fusion
- large architecture rewrites
- production-grade packaging
