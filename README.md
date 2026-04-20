# LiDAR Object Segmentation & Tracking

Point cloud processing pipeline for [SemanticKITTI](http://www.semantic-kitti.org). Processes consecutive LiDAR frames through six stages:

1. **Noise removal** — Z-filter + statistical outlier removal
2. **Voxel downsampling** — uniform spatial density (0.05 m)
3. **RANSAC ground removal** — plane fitting with normal validation
4. **HDBSCAN clustering** — density-adaptive object segmentation
5. **Geometric filtering** — oriented bounding box volume/dimension checks
6. **Heuristic classification** — physics-based rules from bbox dimensions

Objects are tracked across frames using centroid-based Hungarian assignment. See `pipeline.html` for the full architecture diagram.

## Dataset

This project uses the **SemanticKITTI** dataset. Each sequence contains:

```
dataset/sequences/<seq>/
├── velodyne/    # Point cloud frames (.bin)
├── labels/      # Semantic + instance labels (.label) — seqs 00–10
├── poses.txt    # Ego-motion poses (camera frame)
├── calib.txt    # Sensor calibration (velo-to-cam transform)
└── times.txt    # Per-frame timestamps
```

Only sequences 00 and 01 have velodyne data in this local checkout. The full dataset (~80 GB) is stored separately.

## How to Run

```bash
source .venv/bin/activate

# Live visualizer (opens Open3D window)
python src/main.py

# Headless with output writing
python src/main.py --no-gui --save-output --seq 00 --frames 100

# Evaluate detection quality against GT labels
python src/evaluate.py

# Data exploration notebook
jupyter notebook notebooks/data_exploratory.ipynb
```

## Output

When run with `--save-output`, the pipeline writes:
- `output/<seq>/objects/<track_id>.ply` — per-object accumulated point clouds
- `output/<seq>/tracks.json` — track metadata (ID, class, frame range, centroids)
