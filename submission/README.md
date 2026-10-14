# LiDAR Point-Cloud Segmentation, Tracking, and Completion

Source code accompanying the master's thesis on object-level processing of
KITTI-style LiDAR sequences. The pipeline segments objects from raw point
clouds, tracks them across frames, completes partial car observations, and
evaluates the detections against SemanticKITTI ground-truth labels.

**Author:** Ngo Vi Viet Anh

## Overview

Each LiDAR frame passes through a six-stage per-frame pipeline, followed by
cross-frame tracking and optional shape completion:

1. **Preprocessing** — voxel downsampling and statistical outlier removal.
2. **Ground removal** — RANSAC plane fitting, keeping only planes whose normal
   is close to vertical.
3. **Clustering** — HDBSCAN on a coarse voxel grid, with labels propagated back
   to the full-resolution points by nearest neighbour. DBSCAN and Euclidean
   clustering are included for benchmarking.
4. **Geometric filtering** — clusters are pruned by volume, dimensions, height
   above the ground plane, and aspect ratio.
5. **Classification** — a PointNet classifier labels each cluster as car or
   not-car. It is trained in two stages: first on synthetic ShapeNet renders,
   then fine-tuned on real clusters mined from SemanticKITTI. A geometric
   heuristic serves as a fallback when no checkpoint is available.
6. **Tracking** — a centroid tracker associates detections across frames to form
   object tracks. Track-level voting then discards short or inconsistent tracks.

Confirmed car tracks are optionally passed to a point-cloud completion model
(PCN, with an alternative PoinTr implementation) that reconstructs a full car
shape from a single partial observation.

## Repository layout

```
submission/
├── src/                 Python source
│   ├── pipeline.py      Per-frame pipeline; all tunable parameters (PIPELINE_CONFIG)
│   ├── main.py          Run the full pipeline on a sequence (visualise or save output)
│   ├── evaluate.py      Detection metrics against SemanticKITTI ground truth
│   ├── classifier.py    PointNet car / not-car classifier
│   ├── tracker.py       Centroid-based multi-object tracker
│   ├── completion.py    Shape-completion wrapper (canonical-frame normalisation)
│   ├── pcn.py, pointr.py         Completion network architectures
│   ├── train_classifier.py       Two-stage classifier training
│   ├── train_pcn.py, train_pointr.py   Completion-network training
│   ├── mine_*.py        Build training data from SemanticKITTI
│   ├── analyze_*.py, visualize_*.py, show_completion.py   Analysis and visual inspection
│   └── test_*.py        Unit tests (geometry invariants, completion determinism)
├── scripts/             Batch experiments (leave-one-sequence-out, gate ablation, mining)
└── requirements.txt     Python dependencies
```

## Requirements

- Python 3.10 or newer.
- An NVIDIA GPU with CUDA 12.4. The pinned PyTorch build (`torch==2.6.0+cu124`)
  is installed from the PyTorch CUDA index declared in `requirements.txt`;
  without that index, pip installs a CPU-only build.
- Core libraries: PyTorch, Open3D, scikit-learn, HDBSCAN, NumPy, SciPy.

Install into a fresh virtual environment:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## Data and checkpoints

This code snapshot ships without the dataset or trained weights. To reproduce
the results, place them under the submission root using the layout the code
expects:

```
submission/
├── dataset/sequences/<seq>/
│   ├── velodyne/*.bin       raw point clouds
│   ├── labels/*.label       SemanticKITTI annotations (for evaluation)
│   ├── poses.txt            ground-truth poses
│   └── calib.txt            sensor calibration
└── checkpoints/
    ├── stage_b_scratch_best.pth    trained classifier
    └── pcn_kitti_best.pth          trained completion network
```

`<seq>` is a two-digit SemanticKITTI sequence identifier (for example `00` or
`08`). Sequences `00`–`10` provide ground-truth labels.

## Running the pipeline

Run the full pipeline on a sequence and save per-track outputs (per-object
`.ply` files plus a `tracks.json` manifest under `output/<seq>/`):

```bash
python src/main.py --seq 00 --no-gui --save-output
```

Omit `--no-gui` to watch the tracking stream in an Open3D window. Add
`--no-completion` to skip shape completion.

## Evaluation

`evaluate.py` matches predicted objects to ground-truth instances by greedy
Intersection-over-Union (default threshold 0.3) and reports precision, recall,
F1, and mean IoU. By default it evaluates against car instances
(`--target supported-vehicles`).

*Note: To avoid train/test data leakage, ensure you evaluate on a sequence that was not included in the classifier's training set (e.g., Sequence 08 is the standard KITTI validation set).*

Evaluate over a full sequence — small samples are noisy and can invert the
ranking of two configurations, so always cover the whole sequence:

```bash
python src/evaluate.py --seq 08 --frames 5000   # sequence 08 (4071 frames)
```

*(Expected results on Sequence 08: F1 ~0.80, depending on the active pipeline configuration.)*

### Cross-Validation

For rigorous academic evaluation, the repository includes a Leave-One-Sequence-Out (LOSO) cross-validation script. This script sequentially holds out each sequence, trains a classifier on the remaining sequences, and evaluates the held-out sequence to guarantee zero data leakage:

```powershell
# Run LOSO cross-validation (e.g., on sequences 00, 01, and 02)
.\scripts\run_loso.ps1 -Seqs 00,01,02
```

## Training

To train the Stage B classifier from scratch using real LiDAR clusters mined from SemanticKITTI:

```bash
# 1. Mine real car/not-car clusters from your training sequences
python src/mine_stage_b.py --seq 00 01 02 03 04 05 06 07 09 10 --frames 5000 --split train

# 2. Train the classifier (Stage B) without ShapeNet pretraining
python src/train_classifier.py --stage-b --no-pretrain --epochs 15
```

## Tests

```bash
python -m pytest src/test_invariants.py -v
```

These cover coordinate-transform round trips, the purity and length behaviour of
the completion canonical-frame estimator, and completion determinism. Tests that
require a trained checkpoint are skipped automatically when the weights are
absent.

## Configuration

Every pipeline parameter — voxel sizes, RANSAC thresholds, clustering settings,
geometric filter bounds, tracker distances, and completion gates — lives in the
`PIPELINE_CONFIG` dictionary at the top of `src/pipeline.py`. Comments there
record which settings are production defaults and which are provided only for
benchmarking.
