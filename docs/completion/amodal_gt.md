# Amodal GT Boxes for Static Cars (Direction 4a, Step 0)

Created: 2026-07-02. Status: done, validated on seq 08.

## Purpose

Direction 4a asks whether PCN completion yields a bbox closer to the true car
than the raw single-frame partial. That needs a per-car reference box covering
the **full (amodal) extent** — which no single frame provides. This step builds
those reference boxes by accumulating each static car's GT-labeled points
across all frames of a sequence, fitting an oriented box, and keeping only
instances whose accumulated evidence actually supports the fitted extents.

Output: `output/08/amodal_gt.json` (+ visual check `output/08/amodal_gt_check.png`).
Script: `scratchpad/amodal_gt.py`; visual checker: `scratchpad/amodal_gt_viz.py`.

## Method

1. **Accumulate.** For every frame: points with sem=10 (static car) and
   instance ID > 0, transformed to the global frame via `poses[i] @ Tr`
   (cam0 frame: X right, Y down, Z forward; ground plane X–Z, up = −Y),
   pooled per instance ID, voxel-downsampled at 3 cm. Observations with
   < 10 points in a frame are skipped.
2. **Fit.** Per instance (≥ 100 accumulated points): L-shape angle search
   (Zhang et al. closeness criterion, 0.5° steps over 0–90°) in X–Z gives
   yaw; extents are 0.5/99.5 percentiles along the fitted axes (robust to
   label-bleed noise); height is the robust Y extent. Yaw convention:
   length-axis angle from +X toward +Z, in [0, 180).
3. **Filter.** `well_observed` requires ALL of:
   - ≥ 5 frames and ≥ 1500 accumulated points;
   - **azimuth coverage**: ego viewed the car from ≥ 3 of 8 45°-bins, plus
     width evidence (some view within 30° of the length axis, or views from
     both sides);
   - **not `also_moving`**: the instance ID never appears as moving-car
     (sem=252) anywhere in the sequence;
   - **face support**: ≥ 40 points within 0.35 m inside each of the four box
     faces (both length ends, both width sides);
   - **not `extent_uncertain`**: no more than max(15, 0.1% · n) points beyond
     any face + 0.15 m;
   - per-frame bbox-center spread ≤ 2 m in X–Z (static sanity).

All criteria values are stored per instance in the JSON, so downstream steps
can re-filter without re-running the accumulation.

## Why the extra guards (beyond the planned azimuth filter)

Visual inspection of the first build exposed three failure modes that azimuth
coverage does not catch:

| Failure | Mechanism | Guard |
|---------|-----------|-------|
| Smeared accumulation (e.g. W ≈ 2.8 m) | Stop-and-go car: SemanticKITTI labels it 10 while stopped, 252 while moving, **same instance ID** — the sem=10 subset accumulates at multiple positions | `also_moving` rejection |
| Unseen far end despite wide azimuth span | Car end permanently occluded by an adjacent parked car | face-support counts |
| Truncated length (L ≈ 2.7–3.1 m for normal cars) | Sparsely-observed real tail holds < 0.5 % of points, so the percentile trim cuts it | overhang flag (points beyond a face ⇒ extent untrustworthy) |

A support-histogram extent estimator (occupancy bins with an n-scaled
threshold) was tried as a fix for the truncation and **reverted**: it truncated
sparse ends harder (one car L 2.75 → 2.10) and inflated W/H by ~0.1 m through
bin-edge quantization. Lesson: percentile extents were fine on well-covered
cars; the fix belonged in the *filter* (flag truncated fits), not the
*estimator*. The overhang threshold is size-relative — max(15, 0.1% · n) —
because a fixed cutoff rejected clean 30–40k-point cars over ~20 stray
label-bleed points while a 7k-point cloud needs the low floor.

## Results (seq 08, all 4071 frames, ~30 s runtime)

| Stage | Count |
|-------|-------|
| Instances observed (≥ 10 pts in ≥ 1 frame) | 393 |
| Box fitted (≥ 100 acc. points) | 388 |
| **Well-observed** | **40** |

Well-observed dims (m): median L 4.14 / W 1.75 / H 1.47; in sanity range
(L 3.5–5, W 1.6–2.0, H 1.4–1.6): 38/40, 29/40, 29/40. Out-of-range cases are
genuine compacts (L 3.0–3.5 m), confirmed in the BEV check figure — a fleet
property of the Karlsruhe streets, not fit error. 31 instance IDs were
excluded as `also_moving`; the truncation flag fires on 140 fitted instances.

## Verification

```bash
# Reproduce the report (writes to a scratch path; the real JSON is guarded)
.venv\Scripts\python.exe scratchpad\amodal_gt.py --seq 08 --out scratchpad\amodal_gt_repro.json

# Render BEV panels: accumulated points + fitted box + ego path
.venv\Scripts\python.exe scratchpad\amodal_gt_viz.py --seq 08
.venv\Scripts\python.exe scratchpad\amodal_gt_viz.py --seq 08 --insts 96 140 45
```

Expected: 393 / 388 / 40 with the dims table above; boxes hugging the point
footprints in the PNG.

## Limitations

- Accumulated dims are **lower bounds** — LiDAR never sees through occlusion.
  The strict filter trades sample size (40 of 388) for trustworthy extents.
- Weakest accepted case: inst 45 (23 frames, 85° span); raise
  `end_support_min` in Step 1 if it matters.
- W remains the least-constrained dim (as anticipated in the plan): prefer
  L/H/yaw as headline signals; treat W explicitly under the filter.
- Instance IDs are assumed temporally consistent within a sequence
  (SemanticKITTI panoptic property).

## Reuse

- `fit_oriented_box_xz()` is the shared box fitter for Step 1: GT, raw-partial
  and completed boxes all go through the same fitter (`extent="minmax"` for
  small single-frame clusters), so fitter bias cancels in the paired
  comparison.
- The accumulation + viewpoint-coverage machinery is the same infrastructure
  Direction 1a (donor-frame occluded-side Chamfer) needs.

## JSON schema (per instance)

`inst_id`, `n_frames`, `first_frame`, `last_frame`, `n_points_accumulated`,
`also_moving`, `frames` [[frame_idx, n_raw_pts], …], `center_spread_xz_m`,
`center_world` [x, y, z], `yaw_deg`, `dims_lwh` [L, W, H],
`length_end_support` / `width_side_support` [+end, −end],
`ends_seen`, `sides_seen`, `face_overhang` [4 faces], `extent_uncertain`,
`azimuth_bins_occupied`, `azimuth_span_deg`,
`min_view_angle_to_length_axis_deg`, `min_view_angle_to_width_axis_deg`,
`both_sides_seen`, `width_evidence`, `well_observed`.
Top level: `seq`, `created`, `frame_convention`, `config` (all thresholds).
