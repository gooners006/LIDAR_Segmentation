# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Classical (non-ML) LiDAR point cloud processing pipeline for the SemanticKITTI dataset. Processes consecutive frames, segments objects via DBSCAN clustering, tracks them across time using centroid-based association (Hungarian algorithm), and accumulates object-centric point clouds.

## Running the Pipeline

Activate the virtual environment first:
```bash
source .venv/bin/activate        # macOS/Linux
# or .\.venv\Scripts\Activate.ps1  (Windows PowerShell)
```

Run the live visualizer (requires GUI, opens an Open3D window):
```bash
python src/main.py
```

Open the step-by-step notebook:
```bash
jupyter notebook notebooks/index.ipynb
```

## Architecture

**Project structure:**
```
src/           — source code (main.py, tracker.py)
notebooks/     — Jupyter notebooks (index.ipynb)
docs/          — references and documentation (references.bib)
dataset/       — SemanticKITTI data
```

- `src/main.py` — full pipeline loop: loads frames → filters/denoises → ground removal → DBSCAN clustering → bounding box filtering → tracking → Open3D visualization
- `src/tracker.py` — `CentroidTracker` class: maintains object ID dict, uses Hungarian assignment (`scipy.optimize.linear_sum_assignment`) for frame-to-frame matching, drops lost tracks after `max_disappeared` frames

**Data flow per frame:**
1. Load `.bin` → `np.fromfile(..., dtype=np.float32).reshape(-1, 4)` (columns: x, y, z, intensity)
2. Z-threshold filter (`z > -2.0`) + `remove_statistical_outlier` + `voxel_down_sample`
3. RANSAC `segment_plane` → separate `ground_pcd` / `objects_pcd`
4. `cluster_dbscan` → oriented bounding boxes with volume/dimension/height filters
5. `CentroidTracker.update(bbox_objects)` → persistent object IDs

**Dataset layout** (SemanticKITTI format):
```
dataset/sequences/<seq>/velodyne/*.bin       # point clouds — only seqs 00 and 01 present locally
dataset/sequences/<seq>/labels/*.label       # semantic labels — seqs 00–10 only
dataset/sequences/<seq>/poses.txt            # ego-motion (use for point accumulation)
```

**Local dataset note:** Only sequences 00 and 01 have `velodyne/` data in this checkout (Mac, limited storage). Sequences 02–10 have labels/poses but no point clouds. Sequences 11–21 are the test split (no labels). The full 22-sequence dataset is on the Windows desktop.

## Key Parameters (defaults in `src/main.py`)

| Parameter | Value | Purpose |
|---|---|---|
| `voxel_size` | 0.05 | Downsampling resolution (m) |
| `distance_threshold` | 0.2 | RANSAC ground plane tolerance (m) |
| `eps` | 0.5 | DBSCAN neighborhood radius (m) |
| `min_points` | 10 | DBSCAN min cluster size |
| `max_distance` | 2.0 | Tracker max centroid match distance (m) |
| `max_disappeared` | 5 | Frames before dropping a track |

The notebook `notebooks/index.ipynb` is the canonical reference for parameter choices and pipeline order.

## Coding Conventions

- Point cloud loading: always `np.fromfile(path, dtype=np.float32).reshape(-1, 4)`
- Open3D visualization (`draw_geometries`, `Visualizer`) opens GUI windows — not suitable for headless/CI runs. Wrap behind `--no-gui` flag in scripts.
- If adding a batch script, place it at `tools/run_sequence.py` with args `--seq`, `--start`, `--end`, `--voxel_size`, `--ransac_thresh`, `--dbscan_eps`, `--dbscan_min`, `--max_match_distance`, `--missed_frame_tolerance`, `--save-output`, `--no-gui`.
- Output artifacts: per-object stacked clouds → `output/<seq>/objects/<track_id>.ply`; track metadata → `tracks.json`.

## Out of Scope

Deep learning detection/tracking, real-time execution, SLAM/map optimization, camera–LiDAR fusion.
