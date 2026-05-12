# CLAUDE.md

## Project Context

This repository is for a master's thesis research project on LiDAR point-cloud processing using KITTI-style sequences. The current focus is object segmentation, tracking/output extraction, and evaluation against ground-truth labels.

This is research code. Prioritize clarity, reproducibility, and easy experimentation over heavy abstraction.

## Start of Session

At the start of each session:

1. Read `docs/session_summary.md`.
2. Identify:
   - current research goal
   - latest experiment results
   - open issues
   - next planned step
3. Check relevant files in `docs/` before asking the user for background context.

Important docs:

- `docs/session_summary.md` — latest progress, open issues, next steps
- `docs/datasets.md` — dataset layout and assumptions
- `docs/findings.md` — experiment observations and conclusions
- `docs/pipeline_feedback.md` — notes on pipeline behavior and known problems
- `docs/pcn/` — PCN-related notes and references

## Common Commands

Activate the environment:

```bash
source .venv/bin/activate
```

Visualize the pipeline with Open3D:

```bash
python src/main.py
```

Run headless and save PLY + `tracks.json` output:

```bash
python src/main.py --no-gui --save-output --seq 00 --frames 100
```

Evaluate segmentation against ground-truth labels:

```bash
python src/evaluate.py --seq 00 --frames 100 --iou-threshold 0.3
```

Open the exploratory notebook:

```bash
jupyter notebook notebooks/data_exploratory.ipynb
```

## Experiment Protocol

When testing any change, such as a new parameter, algorithm, or pipeline step:

1. State the hypothesis.
2. State which metric is expected to improve.
3. Run `evaluate.py` before the change to record baseline results.
4. Make only the intended change.
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

Do not treat a change as successful without evaluation results.

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

- Prefer readable, explicit code over premature abstraction.
- Avoid large refactors unless specifically requested.
- Keep changes small and easy to compare experimentally.

## Data and Output Safety

- Do not overwrite existing experiment outputs unless explicitly requested.
- Preserve previous evaluation results when comparing experiments.
- Use clear output folders or filenames when testing new configurations.
- Do not delete generated outputs, notebooks, or result files without permission.

## Evaluation Priority

Primary segmentation metrics:

- F1
- meanIoU
- Precision
- Recall

When metrics conflict, explain the tradeoff. For example, higher recall with much lower precision may indicate over-segmentation.

## Known Failure Modes

- Small or distant objects may be under-segmented.
- Adjacent vehicles may be merged into one cluster.
- Sparse LiDAR points can reduce IoU even when the visual result looks acceptable.
- Parameter changes may improve one sequence but hurt another.

## Agent Behavior

Before editing code:

1. Inspect the relevant files.
2. Explain the intended change for exploratory/risky changes.
3. Keep the diff minimal.
4. Run or suggest the correct evaluation command.
5. Update `docs/session_summary.md` if the change affects project direction or results.

## Out of Scope

The following are out of scope unless explicitly requested:

- real-time execution
- SLAM or map optimization
- camera-LiDAR fusion
- large architecture rewrites
- production-grade packaging
