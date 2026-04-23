# CLAUDE.md

## Thesis Context

Master thesis research: LiDAR point cloud segmentation & object completion on SemanticKITTI. The project is in the **research & experimentation phase** — the focus is on iterating on ideas, running experiments, and measuring results, not production code.

The core research question: can we build a classical (non-DL) pipeline for LiDAR object segmentation, then use those extracted objects as training data for a DL-based point cloud completion network that bridges the synthetic-to-real domain gap?

`pipeline.html` is the canonical diagram of the full architecture.

## Research Status

**Working (baseline established):**
- 6-stage classical pipeline: noise removal → downsample → RANSAC ground removal → HDBSCAN clustering → geometric filtering → heuristic classification
- Multi-frame tracking with centroid-based Hungarian assignment
- Evaluation harness comparing detections against SemanticKITTI GT (IoU-based P/R/F1)

**Active research / next experiments:**
- `completion.py` — DL-based point cloud completion (scaffold only, model not yet integrated). Has sim-to-real augmentation (`simulate_lidar_noise`) and `KITTIObjectDataset` loader ready. Needs: pick a completion architecture, integrate weights, implement `fine_tune()`.
- Improving segmentation quality (current eval metrics are the baseline to beat)
- Bridging ShapeNet (synthetic) → KITTI (real) domain gap for completion training

**Not started:** completion model selection, training loop, quantitative completion evaluation.

## Running Experiments

```bash
source .venv/bin/activate

# Visualize pipeline (opens Open3D window)
python src/main.py

# Headless run with PLY + tracks.json output
python src/main.py --no-gui --save-output --seq 00 --frames 100

# Evaluate segmentation against GT labels (prints per-frame + aggregate P/R/F1)
python src/evaluate.py --seq 00 --frames 100 --iou-threshold 0.3

# Data exploration
jupyter notebook notebooks/data_exploratory.ipynb
```

## How to Run an Experiment

When testing a change (new parameter, new algorithm, new pipeline step):
1. State the hypothesis and what metric should improve
2. Run `evaluate.py` before the change to get baseline numbers
3. Make the change
4. Run `evaluate.py` after — compare P/R/F1/meanIoU
5. Record what changed and whether it helped

## Architecture

```
src/
  pipeline.py    — pipeline functions + PIPELINE_CONFIG (all tunable parameters live here)
  main.py        — full loop: load → steps 1-6 → tracking → visualization
  tracker.py     — CentroidTracker: Hungarian assignment, persistent IDs
  classifier.py  — heuristic classification from bbox dimensions
  completion.py  — (scaffold) DL completion: sim-to-real augmentation, dataset loader, metrics
  evaluate.py    — detection evaluation against SemanticKITTI GT (IoU matching → P/R/F1)
notebooks/
  data_exploratory.ipynb — dataset exploration (bin, label, poses, calib)
docs/
  references.bib — bibliography
```

## Data Flow (per frame)

1. Load `.bin` → `np.fromfile(..., dtype=np.float32).reshape(-1, 4)`
2. **Step 1** — Z-filter (`z > -2.0`) + `remove_statistical_outlier(k=20, σ=2)`
3. **Step 2** — `voxel_down_sample(0.05)`
4. **Step 3** — RANSAC `segment_plane` + normal check (`|c|/norm > 0.5` rejects walls)
5. **Step 4** — HDBSCAN clustering (density-adaptive, `min_cluster_size=10`)
6. **Step 5** — Oriented bounding boxes + geometric filter (height relative to ground plane)
7. **Step 6** — Heuristic classification → pedestrian / car / truck / traffic_sign / vegetation
8. Transform to global frame: `T_total = poses[i] @ Tr` (Tr from `calib.txt`)
9. CentroidTracker assigns persistent IDs across frames
10. *(planned)* Step 7 — DL point cloud completion on tracked objects

## Coordinate Transform

Poses from `poses.txt` are in the **camera frame**. `Tr` from `calib.txt` transforms LiDAR → camera. Global transform: `T_total = poses[frame_idx] @ Tr`.

## Key Parameters (PIPELINE_CONFIG in `src/pipeline.py`)

| Parameter | Value | Purpose |
|---|---|---|
| `voxel_size` | 0.05 | Downsampling resolution (m) |
| `ransac_distance_threshold` | 0.2 | Ground plane tolerance (m) |
| `ransac_min_normal_z` | 0.5 | Normal check: reject non-horizontal planes |
| `hdbscan_min_cluster_size` | 10 | Min points per cluster |
| `hdbscan_min_samples` | 5 | Core point neighbourhood density |
| `min_volume` | 0.5 | Min bbox volume (m³) |
| `max_center_height_above_ground` | 3.0 | Max bbox center height above ground (m) |
| `tracker_max_distance` | 2.0 | Max centroid match distance (m) |
| `tracker_max_disappeared` | 5 | Frames before dropping a track |

All parameters are tunable. When experimenting, change values in `PIPELINE_CONFIG` and re-run `evaluate.py` to measure the effect.

## Evaluation Metrics

`evaluate.py` computes per-frame and aggregate:
- **Precision** — fraction of detected clusters that match a GT instance (IoU ≥ threshold)
- **Recall** — fraction of GT instances matched by a detection
- **F1** — harmonic mean
- **Mean IoU** — average IoU of matched pairs

GT uses SemanticKITTI "thing" classes (cars, pedestrians, cyclists, etc.). IoU threshold default: 0.3.

For completion (future): Chamfer distance and F-Score (`completion.py` already has these implemented).

## Dataset (SemanticKITTI)

```
dataset/sequences/<seq>/velodyne/*.bin   # point clouds
dataset/sequences/<seq>/labels/*.label   # semantic + instance labels (seqs 00-10)
dataset/sequences/<seq>/poses.txt        # ego-motion (camera frame)
dataset/sequences/<seq>/calib.txt        # sensor calibration (includes Tr)
```

Only sequences 00 and 01 have velodyne data locally. Full dataset is on the Windows desktop.

## Coding Conventions

- Point cloud loading: `np.fromfile(path, dtype=np.float32).reshape(-1, 4)`
- All pipeline parameters live in `PIPELINE_CONFIG` in `src/pipeline.py`
- Global transforms always use `poses[i] @ Tr`, never raw poses
- Open3D visualization behind `--no-gui` flag; never at module import time
- Output: `output/<seq>/objects/<track_id>.ply` and `output/<seq>/tracks.json`
- This is research code — prioritize clarity and easy experimentation over abstraction

## Out of Scope

Real-time execution, SLAM/map optimization, camera-LiDAR fusion.
