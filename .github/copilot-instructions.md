# Copilot / AI Agent Instructions for LIDAR_Segmentation

## Big Picture

LiDAR point cloud processing pipeline for SemanticKITTI. Seven stages: noise removal, voxel downsampling, RANSAC ground removal, DBSCAN clustering, geometric filtering, heuristic classification, and DL-based point cloud completion (scaffolded). Objects are tracked across frames using centroid-based Hungarian assignment.

`pipeline.html` is the canonical architecture diagram.

## Key Files

```
src/pipeline.py      — shared pipeline functions + PIPELINE_CONFIG (single source of truth)
src/main.py          — full pipeline loop + Open3D visualization + output writing
src/tracker.py       — CentroidTracker (Hungarian assignment, persistent IDs)
src/classifier.py    — Step 6: heuristic classification from bbox dimensions
src/completion.py    — Step 7: DL completion scaffold + Chamfer/F-Score metrics
src/evaluate.py      — detection evaluation against SemanticKITTI GT labels
notebooks/data_exploratory.ipynb — dataset exploration notebook
dataset/sequences/<seq>/velodyne/*.bin, labels/*.label, poses.txt, calib.txt
```

## Environment

```bash
source .venv/bin/activate        # macOS/Linux
python src/main.py               # live visualizer
python src/main.py --no-gui --save-output  # headless with output
python src/evaluate.py           # evaluate detection quality
```

Dependencies: `numpy`, `open3d`, `matplotlib`, `scikit-learn`, `scipy`

## Canonical Parameters (from `src/pipeline.py` PIPELINE_CONFIG)

| Parameter | Value |
|---|---|
| `voxel_size` | 0.05 |
| `ransac_distance_threshold` | 0.2 |
| `ransac_iterations` | 1000 |
| `ransac_min_normal_z` | 0.5 |
| `dbscan_eps` | 0.5 |
| `dbscan_min_points` | 10 |
| `min_volume` | 0.5 |
| `max_center_z` | 0.5 |
| `tracker_max_distance` | 2.0 |
| `tracker_max_disappeared` | 5 |

All pipeline parameters are defined in `PIPELINE_CONFIG` in `src/pipeline.py`. Do not duplicate them elsewhere.

## Coordinate Transform

Poses are in the **camera frame**. Always apply the velo-to-cam calibration:

```python
T_total = poses[frame_idx] @ Tr   # Tr from calib.txt
pcd.transform(T_total)
```

## Coding Conventions

- Point cloud loading: `np.fromfile(path, dtype=np.float32).reshape(-1, 4)`
- Use shared functions from `src/pipeline.py` (preprocess_frame, remove_ground, cluster_objects, filter_clusters)
- Open3D visualization opens GUI windows — wrap behind `--no-gui` flag, never at module import time
- Output artifacts: `output/<seq>/objects/<track_id>.ply`, `output/<seq>/tracks.json`
- Batch scripts: `tools/run_sequence.py` with args `--seq`, `--start`, `--end`, `--save-output`, `--no-gui`

## Out of Scope

Real-time execution, SLAM/map optimization, camera-LiDAR fusion.
