# Copilot / AI Agent Instructions for LIDAR_Segmentation

This file gives concise, repository-specific guidance so AI coding agents can be immediately productive.

## Big Picture

**Purpose**  
This project focuses on dataset extraction, visualization, and multi-frame LiDAR processing for SemanticKITTI-style sequences stored under `dataset/sequences/`. The goal is to process consecutive LiDAR frames, segment objects, track them across time, and accumulate (stack) object-centric point clouds in a consistent reference frame.

**Primary Artifact**  
The primary artifact is an exploratory Jupyter notebook (`index.ipynb`) that demonstrates end-to-end multi-frame LiDAR processing on a sequence. The notebook emphasizes temporal consistency, object tracking, and point accumulation rather than single-frame visualization.

Implementation note: This pipeline processes consecutive LiDAR frames, associates object clusters across time (nearest-centroid or assignment/Hungarian), and accumulates object-centric point clouds. When available, per-frame poses from `poses.txt` should be used to compensate ego-motion before stacking; stacking may be done in a global reference or per-object local frame.

**Core Capabilities**  
The notebook implements and demonstrates:
- Iterating over consecutive LiDAR frames within a sequence
- Loading raw LiDAR scans (`.bin`) into NumPy arrays
- Preprocessing point clouds (denoising, voxel downsampling)
- Ground removal using RANSAC plane fitting
- Object segmentation using DBSCAN clustering
- Per-frame object representation via clusters and bounding boxes
- Frame-to-frame object association (tracking) using geometric proximity
- Temporal stacking of object point clouds across frames
- Visualization of tracked objects and accumulated points

**Data Flow**


dataset/sequences/<sequence_id>/velodyne/*.bin
- NumPy arrays of shape (-1, 4) [x, y, z, intensity]
- Open3D PointCloud
- Denoising + voxel downsampling
- Ground plane removal (RANSAC)
- DBSCAN clustering (per-frame objects)
- Object tracking across frames
- Object-centric point accumulation (stacking)
- Visualization in local or world coordinates


**Temporal Processing Model**
- Frames are processed sequentially.
- Objects are represented by cluster-level features (centroid, extent, bounding box).
- Object identities are propagated across frames using nearest-neighbor or assignment-based matching.
- Points belonging to the same tracked object are accumulated over time to form denser, object-centric point clouds.
- Vehicle ego-motion (poses) may optionally be used to stabilize accumulated point clouds in a global or local reference frame.

**Tracking & Association**
- Association methods: nearest-centroid matching (fast) or Hungarian assignment (optimal for many objects).
- Key parameters: `max_match_distance` (meters), `missed_frame_tolerance` (frames before dropping a track), and `merge_iou_threshold` (when merging split clusters).
- ID handling: persist IDs across occlusions up to `missed_frame_tolerance`, allow track spawn/terminate and simple track merging heuristics based on overlap or centroid proximity.

When available, use `poses.txt` to transform per-frame object points into a consistent reference before accumulating object-centric clouds. If poses are not available, stacking in a local (object) frame with relative motion priors is an alternative.

**Design Intent**
- Treat LiDAR as the primary sensor (no camera fusion).
- Favor geometric and classical methods (clustering, tracking) over learning-based models.
- Maintain clear separation between per-frame processing and temporal association.
- Keep all steps explicit and inspectable for debugging and analysis.
- Enable easy extension toward trajectory estimation, object motion analysis, or evaluation with SemanticKITTI labels.

**Out of Scope**
- Deep learning-based detection or tracking
- Real-time or embedded execution
- Full SLAM or map optimization
- Camera–LiDAR fusion



## Key files and layout
- `dataset/sequences/<seq>/velodyne/*.bin` : input frames, binary float32, reshape(-1,4).
- `dataset/sequences/<seq>/labels/*.label` : per-frame semantic labels (SemanticKITTI format).
- `dataset/sequences/<seq>/calib.txt`, `poses.txt`, `times.txt` : calibration & pose metadata used by downstream evaluation.
- [index.ipynb](index.ipynb) : canonical example pipeline and parameter choices — use it as the primary reference for behavior and defaults.
- `README` : dataset provenance and license notes.

## Environment & common commands
- This repo expects a Python venv. A local `.venv` is present in developer environments; activate before running the notebook.
  - Bash / WSL: `source .venv/Scripts/activate`
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
- Typical dependencies (used in the notebook): `numpy`, `matplotlib`, `open3d`. If no `requirements.txt` exists, create one matching the notebook imports.

## Coding and modification conventions (discoverable patterns)
- Point cloud loading: use the exact pattern from the notebook: `points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)`.
- Color mapping: the notebook normalizes Z to color by height (`norm_z = (z - z.min())/(z.max()-z.min())`) and uses matplotlib colormaps (e.g., `viridis`) then drops alpha channel for Open3D.
- Open3D pipeline steps used (follow this order in code changes):
  1. Build `pcd = o3d.geometry.PointCloud()` with `pcd.points = Vector3dVector(xyz)`
 2. Optional low-Z mask for removing scan noise (example: `z_threshold = -2.0`)
 3. `remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)` for denoising
 4. `voxel_down_sample(voxel_size=0.1)` for downsampling
 5. `segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=200)` for ground extraction
 6. `cluster_dbscan(eps=0.5, min_points=10)` for object clustering
- Reuse parameter values in the notebook as sensible defaults; if you change them, update `index.ipynb` so reviewers see the new defaults.

**Defaults and parameters**
- `voxel_size=0.1`
- RANSAC `distance_threshold=0.2`
- DBSCAN `eps=0.5`, `min_points=10`
- Denoise `nb_neighbors=20`, `std_ratio=2.0`
- Tracking `max_match_distance=2.0`, `missed_frame_tolerance=3`

## Integration points & external dependencies
- Notebook depends on Open3D visualization calls (e.g., `o3d.visualization.draw_geometries`) — these open GUI windows and are not suited for headless CI. For automated scripts, use Open3D geometry export or headless renderers.
- No build system or tests detected; changes to core processing should be validated using the notebook or a small runnable script that reads `dataset/sequences/00/velodyne/000000.bin`.

**Output artifacts**
- Per-object stacked point clouds: save as `.ply` or `.pcd` under `output/<seq>/objects/<track_id>.ply`.
- `tracks.json`: list of tracks with `track_id`, `frames`, `centroids`, `timestamps`, and optional bounding boxes.
- Per-frame diagnostics (cluster counts, unmatched clusters) as CSV/JSON for debugging.

## Examples & copyable snippets (use these to implement or refactor)
- Load frame and form point cloud:

```py
points = np.fromfile("dataset/sequences/00/velodyne/000000.bin", dtype=np.float32).reshape(-1,4)
xyz = points[:, :3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
```

- Basic denoise and downsample (matching notebook defaults):

```py
pcd_denoised, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_down = pcd_denoised.voxel_down_sample(voxel_size=0.1)
```

- RANSAC ground plane segmentation (notebook defaults):

```py
plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=200)
ground = pcd_down.select_by_index(inliers)
objects = pcd_down.select_by_index(inliers, invert=True)
```

## When editing the notebook or adding scripts
- Keep the notebook as the human-readable canonical demonstration of preprocessing order and parameter choices.
- If you add a script for reproducible runs, name it `tools/run_sequence.py` and accept arguments:
  - `--seq` (sequence id)
  - `--start`, `--end` (frame range)
  - `--voxel_size`, `--ransac_thresh`, `--dbscan_eps`, `--dbscan_min`
  - `--max_match_distance`, `--missed_frame_tolerance`
  - `--save-output`, `--no-gui`
- Avoid calling interactive `draw_geometries` inside CI; wrap visualization behind `if __name__ == '__main__'` and expose a `--no-gui` flag.
  Wrap interactive visualization behind `--no-gui` and provide an export path for saved visualizations so CI runs remain headless.


## Dependencies & Quick Run
- Recommended minimal `requirements.txt`: `numpy`, `open3d`, `matplotlib`, `scikit-learn`
- Example one-sequence run (after activating `.venv`):

```bash
python tools/run_sequence.py --seq 00 --start 0 --end 100 --voxel_size 0.1 --dbscan_eps 0.5 --max_match_distance 2.0 --save-output
```

---
If anything here is unclear or you'd like additional examples (unit test scaffolding, a non-interactive runner, or a minimal `requirements.txt`), tell me which piece to add or adjust.
