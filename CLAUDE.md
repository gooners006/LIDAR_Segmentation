# CLAUDE.md

## Purpose

LiDAR point cloud processing pipeline for SemanticKITTI. Processes consecutive frames through 6 stages: noise removal, downsampling, ground segmentation, HDBSCAN clustering, geometric filtering, and heuristic classification. Tracks objects across frames using centroid-based Hungarian assignment.

`pipeline.html` is the canonical diagram of the full architecture.

## Running the Pipeline

```bash
source .venv/bin/activate

# Live visualizer (opens Open3D window)
python src/main.py

# Headless with output
python src/main.py --no-gui --save-output --seq 00 --frames 100

# Evaluate detection against GT labels
python src/evaluate.py

# Data exploration notebook
jupyter notebook notebooks/data_exploratory.ipynb
```

## Architecture

```
src/
  pipeline.py    — shared pipeline functions + PIPELINE_CONFIG (canonical parameters)
  main.py        — full pipeline loop: load → steps 1-6 → tracking → visualization
  tracker.py     — CentroidTracker: Hungarian assignment, persistent IDs
  classifier.py  — Step 6: heuristic classification from bbox dimensions
  completion.py  — (dormant) DL completion scaffold + Chamfer/F-Score metrics
  evaluate.py    — detection quality evaluation against SemanticKITTI GT
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
5. **Step 4** — HDBSCAN clustering (density-adaptive, `min_cluster_size=10`, `cluster_selection_epsilon=0.5`)
6. **Step 5** — Oriented bounding boxes + geometric filter (height relative to ground plane)
7. **Step 6** — Heuristic classification → pedestrian / car / truck / traffic_sign / vegetation
8. Transform to global frame: `T_total = poses[i] @ Tr` (Tr from `calib.txt`)
9. CentroidTracker assigns persistent IDs across frames

## Coordinate Transform

Poses from `poses.txt` are in the **camera frame**. The `Tr` matrix from `calib.txt` transforms LiDAR points into the camera frame. Global transform chain:

```python
T_total = poses[frame_idx] @ Tr   # Tr = calib["Tr"] from calib.txt
pcd.transform(T_total)
```

## Key Parameters (from `src/pipeline.py` PIPELINE_CONFIG)

| Parameter | Value | Purpose |
|---|---|---|
| `voxel_size` | 0.05 | Downsampling resolution (m) |
| `ransac_distance_threshold` | 0.2 | Ground plane tolerance (m) |
| `ransac_min_normal_z` | 0.5 | Normal check: reject non-horizontal planes |
| `hdbscan_min_cluster_size` | 10 | Min points per cluster |
| `hdbscan_min_samples` | 5 | Core point neighbourhood density |
| `min_volume` | 0.5 | Min bbox volume (m³) |
| `max_center_height_above_ground` | 3.0 | Max bbox center height above ground plane (m) |
| `tracker_max_distance` | 2.0 | Max centroid match distance (m) |
| `tracker_max_disappeared` | 5 | Frames before dropping a track |

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

## Out of Scope

Real-time execution, SLAM/map optimization, camera-LiDAR fusion.
