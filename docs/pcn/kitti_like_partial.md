# KITTI-like Single-View Partial Generation

`ShapeNetCompletionDataset._render_kitti_like` (in `src/train_pcn.py`) generates
synthetic completion *partials* that match the **real single-frame LiDAR clusters**
the completer sees at inference. Enabled with `--kitti-like` (sets
`config["kitti_like_partial"]`).

## Motivation

Findings #15–19: every prior PCN attempt trained on partials that were structurally
out-of-distribution from real data, so the model produced blobs on real clusters.

The scratchpad derisk (`scratchpad/derisk_synth_vs_real.py`,
`finalize_generator.py`, `normalized_check.py`) compared candidate generators
against real seq-08 clusters mined with `mine_completion_pairs.py` and established:

1. **Accumulated tracks are the wrong target.** For moving cars they are motion
   smears (up to ~89k points, ~8 m long) — not completable, not matchable.
   Completion should run on a **single-frame** observation, and training partials
   should be single-view.
2. **The real completion input is post-pipeline**, not a raw scan: it has been
   voxel-downsampled (0.05 m), ground-removed, and is observed from one viewpoint.
   The old `_render_lidar_partial` skipped voxelization and ground removal — the
   key omission.
3. **Range noise must be small/absolute** (~0.015 m). The old `0.005 × range`
   model added up to 0.15 m of noise at 30 m, washing out scan-ring banding.

## What it does

Per ShapeNet car mesh (centred at origin, scaled to `category_scale_m`):

1. **Orient to gravity.** ShapeNet cars are not Z-up; the height axis is the
   smallest-extent axis. Permute so the up-axis → Z (and permute hit points back
   afterwards so the partial stays aligned with the GT sample).
2. **One Velodyne viewpoint.** Sample azimuth and range `kitti_distance_range`
   (8–30 m). Sensor at `kitti_sensor_height` (1.73 m) above the ground plane
   (mesh min along up-axis). Cast HDL-64E beams (64 elevations −24.9°…+2.0°,
   `kitti_h_res_deg` = 0.09° horizontal) spanning the object.
3. **Range noise** — additive Gaussian, `kitti_noise_sigma` = 0.015 m.
4. **Voxel downsample** at `kitti_voxel_size` = 0.05 m (must match
   `PIPELINE_CONFIG["voxel_size"]`).
5. **Ground removal** — drop points within `kitti_ground_cut` (0.30 m) of the
   ground plane.

The GT target is unchanged: a dense uniform surface sample of the *complete* mesh.
`__getitem__` then centres both on the GT centroid, applies the existing random
sparsification (k~U[32,256]) + unit-sphere normalization, so the model trains
across a range of densities and is robust to real clusters' 40–1371 point range.

## Validation (scratchpad derisk)

Distribution match vs real seq-08 single-frame car clusters (medians):

| metric | synthetic single-view | real single-frame |
|---|---|---|
| points (raw) | ~270–600 | 283 |
| NN spacing (m) | ~0.06 | 0.058 |
| length / width / height (m) | ~2.4 / 2.0 / 1.5 | 3.3 / 1.6 / 1.1 |

Coverage (one-sided), voxel density, scale, and scan banding match closely.
Residual gaps: synthetic sees a slightly cleaner/fuller side and ShapeNet cars
run wider/taller than KITTI sedans — second-order, and the random sparsification
plus per-cloud normalization absorb most of it. The true test is downstream
completion quality on real clusters (visual + Chamfer), per the experiment protocol.

## Config keys (`TRAIN_CONFIG`)

`kitti_like_partial`, `kitti_distance_range`, `kitti_sensor_height`,
`kitti_ground_cut`, `kitti_noise_sigma`, `kitti_h_res_deg`, `kitti_voxel_size`.

## Reproduce

```bash
# Train PCN on KITTI-like partials (checkpoints -> pcn_kitti_*.pth)
python src/train_pcn.py --kitti-like --epochs 80
# or with an explicit tag
python src/train_pcn.py --kitti-like --tag pcn_kitti --epochs 80
```

## Open follow-up

At inference the completer must run on a **single representative frame** per track,
not the accumulated `all_pts` (a smear). That `main.py` change is tracked separately
and is required before pipeline-level completion evaluation is meaningful.
