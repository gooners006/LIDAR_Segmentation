# Dataset Organization

This project uses the **[SemanticKITTI](http://www.semantic-kitti.org)** dataset format. Download the dataset and extract it before running the code.

## Expected Directory Structure

The full dataset contains 22 sequences (00–21). Each sequence follows this layout:

```text
dataset/sequences/<seq>/
├── velodyne/          # Point cloud frames (.bin) — required to run the pipeline
│   ├── 000000.bin
│   ├── 000001.bin
│   └── ...
├── labels/            # Semantic + instance labels (.label) — sequences 00–10 only
│   ├── 000000.label
│   └── ...
├── poses.txt          # Ego-motion poses (one 3×4 matrix per line)
├── calib.txt          # Sensor calibration
└── times.txt          # Per-frame timestamps
```

### Local vs. full dataset

The full dataset (~80 GB) is not required to run the pipeline. Only sequences with `velodyne/` data present can be processed:

| Sequences | velodyne | labels | Notes |
|-----------|----------|--------|-------|
| 00, 01 | yes | yes | Available in this local checkout |
| 02–10 | no | yes | Full dataset only |
| 11–21 | no | no | Test split — no labels |

> The complete dataset is stored separately (Windows desktop). To run on sequences beyond 00/01, copy the relevant `velodyne/` folder(s) into `dataset/sequences/<seq>/`.

## How to Run

1. Open `index.ipynb` for a step-by-step walkthrough of the implementation.
2. Run the full pipeline from the project root:

```bash
python main.py
```
