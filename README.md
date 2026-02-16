# 📂 Dataset Organization

This project uses the **[SemanticKITTI](http://www.semantic-kitti.org)** dataset format. Download the dataset and extract it before running the code.

## Expected Directory Structure

```text
Project_Root/
│
├── main.py                # Main entry point
├── tracker.py             # CentroidTracker implementation
├── index.ipynb            # Step-by-step notebook
├── README.md              # Documentation
│
└── dataset/
    └── sequences/
        └── 00/
            └── velodyne/
                ├── 000000.bin
                ├── 000001.bin
                ├── 000002.bin
                └── ...
```

## ▶️ How to Run

1. Open `index.ipynb` for a step-by-step walkthrough of the implementation.
2. Run the full pipeline from the project root:

```bash
python main.py
```
