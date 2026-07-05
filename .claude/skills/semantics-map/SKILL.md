---
name: semantics-map
description: >
  Produce a reviewable semantics map before porting, adapting, or integrating any
  external method, library, reference implementation, or data format. Use this skill
  whenever the task involves "integrate", "port", "adapt", "reimplement", "add support
  for", or wiring in third-party code or formats — e.g., Patchwork++ ground removal,
  SORT-style tracking, KITTI raw tracklets, a paper's reference code, a new dataset
  layout. MANDATORY before writing any integration code (see CLAUDE.md).
---

# Semantics Map Skill

You are about to integrate external code, a published method, or an unfamiliar data
format into this research pipeline. Before writing any integration code, produce a
**semantics map** — an explicit, reviewable statement of what the reference actually
does — and get user sign-off.

**Motivating failure:** Finding #26. PCN inference applied PCA alignment and
partial-centroid normalization that training never used. The mismatch was silent,
produced plausible-looking garbage ("blobs"), and cost multiple sessions (#15–19) to
diagnose. A semantics map would have caught it in minutes.

## Steps

1. **Read the reference first.** Actual source code, format spec, or paper — not your
   memory of it. Quote the load-bearing lines.
2. **Write the semantics map** covering, as applicable:
   - **Coordinate conventions:** axis meanings, handedness, sensor vs. world frame,
     which pose/calibration transforms apply (this repo: global transforms are
     `poses[i] @ Tr`).
   - **Units and scales:** meters vs. normalized, scale factors, voxel sizes.
   - **Normalization pipeline:** exactly what is centered/rotated/scaled, in what
     order, at train time vs. inference time. Flag any asymmetry.
   - **Data assumptions:** density, viewpoint, preprocessing the reference expects
     vs. what our pipeline actually feeds it (post-voxelization 0.05 m,
     ground-removed, single-viewpoint partials).
   - **Behavior matrix:** what will be *preserved*, *deliberately changed*, and
     *dropped* — with a one-line reason for each change/drop.
   - **Edge cases:** empty/sparse clusters, degenerate geometry, boundary frames.
3. **Flag the gotchas** — anything non-obvious that could silently diverge
   (RNG/determinism, integer truncation, implicit sorting, frame off-by-one).
4. **Stop and get sign-off.** Present the map and wait for explicit user confirmation
   before writing integration code. Do not "just start" while the user reads.

## Output format

Present the map in the conversation (a table for the behavior matrix works well). If
the integration is substantial, also save it to
`docs/<topic>/semantics_map_<reference>.md` so it survives the session and can be
cited in the thesis.

## After sign-off

- Implement against the confirmed map; if implementation reveals the map was wrong,
  update the map and re-flag rather than silently diverging (log under Deviations
  per the Experiment Protocol).
