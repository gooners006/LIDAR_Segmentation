## 📂 Dataset Organization

This project is designed to work with the **[Semantic KITTI ](http://www.semantic-kitti.org)** format. Download the dataset and extract the files.

**Your file tree should look like this:**

```text
Project_Root/
│
├── main.py                # The main execution script
├── tracker.py             # The CentroidTracker class file
├── README.md              # Project documentation
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