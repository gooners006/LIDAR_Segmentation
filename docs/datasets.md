# Datasets

## SemanticKITTI

LiDAR point cloud dataset with per-point semantic and instance labels, derived from the KITTI Vision Benchmark. Captured with a Velodyne HDL-64E rotating LiDAR mounted on a car driving through Karlsruhe, Germany.

**Source:** http://www.semantic-kitti.org/

### Directory layout

```
dataset/sequences/
├── 00/                         # sequence directory
│   ├── velodyne/
│   │   ├── 000000.bin          # point cloud frame
│   │   ├── 000001.bin
│   │   └── ...
│   ├── labels/
│   │   ├── 000000.label        # per-point semantic + instance label
│   │   ├── 000001.label
│   │   └── ...
│   ├── poses.txt               # per-frame 3×4 pose matrices (camera frame)
│   └── calib.txt               # sensor calibration (P0–P3 projections, Tr LiDAR→camera)
├── 01/
├── ...
└── 21/
```

### Sequences available locally

| Seq | Frames | Labels | Poses | Calib | Split |
|-----|--------|--------|-------|-------|-------|
| 00  | 4,541  | 4,541  | yes   | yes   | train |
| 01  | 1,101  | 1,101  | yes   | yes   | train |
| 02  | 4,661  | 4,661  | yes   | yes   | train |
| 03  | 801    | 801    | yes   | yes   | train |
| 04  | 271    | 271    | yes   | yes   | train |
| 05  | 2,761  | 2,761  | yes   | yes   | train |
| 06  | 1,101  | 1,101  | yes   | yes   | train |
| 07  | 1,101  | 1,101  | yes   | yes   | train |
| 08  | 4,071  | 4,071  | yes   | yes   | train |
| 09  | 1,591  | 1,591  | yes   | yes   | train |
| 10  | 1,201  | 1,201  | yes   | yes   | train |
| 11  | 921    | —      | yes   | yes   | test  |
| 12  | 1,061  | —      | yes   | yes   | test  |
| 13  | 3,281  | —      | yes   | yes   | test  |
| 14  | 631    | —      | yes   | yes   | test  |
| 15  | 1,901  | —      | yes   | yes   | test  |
| 16  | 1,731  | —      | yes   | yes   | test  |
| 17  | 491    | —      | yes   | yes   | test  |
| 18  | 1,801  | —      | yes   | yes   | test  |
| 19  | 4,981  | —      | yes   | yes   | test  |
| 20  | 831    | —      | yes   | yes   | test  |
| 21  | 2,721  | —      | yes   | yes   | test  |

Total: 41,624 frames, 23,201 with labels (seqs 00–10). Total size: ~89.6 GB.

### File formats

**Point clouds (`.bin`)**
Binary file of float32 values, 4 per point: `[x, y, z, remittance]`.

```python
points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
```

A typical frame contains ~120k points. Example: frame `00/000000.bin` has 124,668 points (1,948 KB).

**Labels (`.label`)**
Binary file of uint32 values, one per point. The lower 16 bits encode the semantic class; the upper 16 bits encode the instance ID.

```python
labels = np.fromfile(path, dtype=np.uint32)
semantic_id = labels & 0xFFFF
instance_id = labels >> 16
```

**Thing classes** (objects with instance labels, used for evaluation):

| ID  | Class        | ID  | Class          |
|-----|--------------|-----|----------------|
| 10  | car          | 30  | person         |
| 11  | bicycle      | 31  | bicyclist      |
| 13  | bus          | 32  | motorcyclist   |
| 15  | motorcycle   | 252 | moving-car     |
| 16  | truck        | 253 | moving-bicyclist |
| 18  | other-vehicle| 254 | moving-person  |
| 20  | other-ground | 255–259 | other moving |

**Poses (`poses.txt`)**
One line per frame. Each line has 12 floats — a 3×4 row-major transformation matrix in the **camera frame**.

```
r00 r01 r02 t0  r10 r11 r12 t1  r20 r21 r22 t2
```

**Calibration (`calib.txt`)**
Contains projection matrices `P0`–`P3` (3×4) and the LiDAR-to-camera transform `Tr` (3×4). To get a point cloud in the global frame:

```python
T_total = poses[frame_idx] @ Tr   # Tr is extended to 4×4 with [0,0,0,1]
```

### Coordinate system

- **LiDAR frame:** x = forward, y = left, z = up
- **Camera frame:** x = right, y = down, z = forward
- Poses are in the camera frame; always apply `Tr` before using poses

---

## ShapeNetCore v2

Large-scale dataset of 3D CAD models organized by WordNet synset categories. Used as the source of **complete** 3D shapes for training the point cloud completion network.

**Source:** https://huggingface.co/datasets/ShapeNet/ShapeNetCore (gated, requires HuggingFace auth)

### Directory layout

```
dataset/shapenet_data/
├── 02958343/                              # car (WordNet synset ID)
│   ├── 100715345ee54d7ae38b52b4ee9d36a3/  # model ID
│   │   ├── models/
│   │   │   ├── model_normalized.obj       # normalized mesh
│   │   │   ├── model_normalized.mtl       # material
│   │   │   ├── model_normalized.json      # metadata (bbox, centroid, vertex count)
│   │   │   ├── model_normalized.solid.binvox    # solid voxelization
│   │   │   └── model_normalized.surface.binvox  # surface voxelization
│   │   └── images/
│   │       ├── texture0.jpg
│   │       └── texture1.jpg
│   ├── 100c3076c74ee1874eb766e5a46fceab/
│   └── ...
├── 02924116/                              # bus
└── 03790512/                              # motorcycle
```

### Categories available locally

| Synset ID  | Category   | Models | Disk size |
|------------|------------|--------|-----------|
| 02958343   | car        | 3,533  | 26.7 GB   |
| 02924116   | bus        | 939    | 3.3 GB    |
| 03790512   | motorcycle | 337    | 2.8 GB    |
| **Total**  |            | **4,809** | **32.7 GB** |

Synset IDs are WordNet 3.0 noun synset offsets. Resolve with:
```python
from nltk.corpus import wordnet
synset = wordnet.synset_from_pos_and_offset('n', 2958343)  # → car.n.01
```

### File formats

**Mesh (`model_normalized.obj`)**
Standard Wavefront OBJ with vertex positions, normals, texture coordinates, and face definitions. References a `.mtl` material file and texture images.

Vertex counts vary by model (7k–66k typical). Example: car model `100715...` has 52,081 vertices, 11 MB file.

**Metadata (`model_normalized.json`)**
```json
{
  "id": "100715345ee54d7ae38b52b4ee9d36a3",
  "numVertices": 52081,
  "min": [-8.03, 0.0, 0.01],
  "max": [0.52, 5.26, 18.30],
  "centroid": [-3.81, 1.77, 8.09]
}
```

**Voxelized (`*.binvox`)**
Pre-computed voxel grids in binvox format. Two variants: `.solid.binvox` (filled interior) and `.surface.binvox` (surface only).

### Alignment and scale

- Models are **pre-aligned**: +Y up, -Z front
- Coordinates are **normalized** (not real-world units)
- For completion training, meshes must be **rescaled** to match KITTI object dimensions (e.g., a car is ~4.5m long, ~1.8m wide, ~1.5m tall in KITTI)

### Usage for completion training

1. Load mesh with `trimesh.load(obj_path)`
2. Sample dense point cloud: `trimesh.sample.sample_surface(mesh, count=16384)` → ground truth
3. Apply `simulate_lidar_noise()` from `completion.py` → partial input
4. Train completion network on (partial, complete) pairs
