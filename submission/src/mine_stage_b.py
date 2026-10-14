"""Mine real LiDAR clusters from SemanticKITTI for Stage B classifier fine-tuning.

Runs pipeline stages 1-5, matches each cluster to GT semantic labels via KD-tree,
and saves centroid-centered .npy files organized by classifier class.

Usage:
    python src/mine_stage_b.py --seq 00 --frames 100
    python src/mine_stage_b.py --seq 00 --frames 4541 --purity 0.8
    python src/mine_stage_b.py --seq 00 01 02 --frames 500 --split train
"""

import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

from pipeline import PIPELINE_CONFIG, cluster_objects, filter_clusters, remove_ground

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SemanticKITTI semantic label -> classifier class (binary: car / not-car)
SEMKITTI_TO_CLASS = {
    10: "car", 252: "car",
}

VALID_CLASSES = {"car", "not-car"}

# SemanticKITTI label names for metadata
SEMKITTI_NAMES = {
    0: "unlabeled", 1: "outlier",
    10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
    16: "on-rails", 18: "truck", 20: "other-vehicle",
    30: "person", 31: "bicyclist", 32: "motorcyclist",
    40: "road", 44: "parking", 48: "sidewalk",
    49: "other-ground", 50: "building", 51: "fence",
    52: "other-structure", 60: "lane-marking", 70: "vegetation",
    71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
    99: "other-object",
    252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
    255: "moving-motorcyclist", 256: "moving-on-rails",
    257: "moving-bus", 258: "moving-truck", 259: "moving-other-vehicle",
}


def classify_cluster(sem_labels: np.ndarray, purity_threshold: float
                     ) -> tuple[str | None, float, float, dict]:
    """Determine classifier class from a cluster's GT semantic labels.

    Maps each SemanticKITTI label to one of 2 classifier classes
    (car, not-car) and votes by point count.  The cluster is accepted
    if the majority class exceeds the purity threshold.

    Args:
        sem_labels: (N,) array of SemanticKITTI semantic label IDs for
            the cluster's points.
        purity_threshold: Minimum fraction of points that must belong to
            the majority classifier class to accept this cluster.

    Returns:
        Tuple of (class_name, purity, raw_sem_purity, histogram).
        *class_name* is ``None`` if purity is below threshold (cluster
        too mixed for training).  *raw_sem_purity* is the fraction of
        points with the single most common semantic label.  *histogram*
        maps human-readable label names to point counts.
    """
    unique, counts = np.unique(sem_labels, return_counts=True)
    total = counts.sum()
    histogram = {SEMKITTI_NAMES.get(int(u), f"label_{u}"): int(c)
                 for u, c in zip(unique, counts)}

    raw_sem_purity = float(counts.max()) / total

    # Map each point's semantic label to a classifier class
    class_votes = {}
    for label, count in zip(unique, counts):
        cls = SEMKITTI_TO_CLASS.get(int(label), "not-car")
        class_votes[cls] = class_votes.get(cls, 0) + int(count)

    majority_class = max(class_votes, key=class_votes.get)
    purity = class_votes[majority_class] / total

    if purity < purity_threshold:
        return None, purity, raw_sem_purity, histogram

    return majority_class, purity, raw_sem_purity, histogram


def mine_frame(bin_path: str, label_path: str, purity_threshold: float):
    """Run the pipeline on one frame and extract labeled clusters.

    Replicates stages 1–5 (z-filter, denoise, downsample, ground
    removal, clustering, geometric filtering) while propagating GT
    semantic labels via KD-tree.  Each filtered cluster is classified
    by label purity and centroid-centered for saving.

    Args:
        bin_path: Path to the Velodyne ``.bin`` point cloud file.
        label_path: Path to the SemanticKITTI ``.label`` file.
        purity_threshold: Minimum purity to accept a cluster.

    Returns:
        Tuple of (results, n_discarded).  *results* is a list of
        ``(points_centered, class_name, metadata)`` tuples for accepted
        clusters.  *n_discarded* counts clusters rejected for low purity.
    """
    cfg = PIPELINE_CONFIG

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    assert len(points) == len(raw_labels), (
        f"bin/label length mismatch: {len(points)} vs {len(raw_labels)} in {bin_path}")
    sem = raw_labels & 0xFFFF

    # Stage 1: Z-filter
    z_mask = xyz[:, 2] > cfg["z_threshold"]
    xyz_filtered = xyz[z_mask]
    sem_filtered = sem[z_mask]

    if len(xyz_filtered) == 0:
        return [], 0

    # Stage 2: Denoise
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
    pcd_denoised, ind_denoise = pcd.remove_statistical_outlier(
        nb_neighbors=cfg["denoise_nb_neighbors"],
        std_ratio=cfg["denoise_std_ratio"],
    )
    sem_denoised = sem_filtered[ind_denoise]
    xyz_denoised = np.asarray(pcd_denoised.points)

    if len(xyz_denoised) == 0:
        return [], 0

    # Stage 3: Downsample
    pcd_down = pcd_denoised.voxel_down_sample(voxel_size=cfg["voxel_size"])
    xyz_down = np.asarray(pcd_down.points)

    if len(xyz_down) == 0:
        return [], 0

    # Propagate labels via KD-tree (nearest denoised point)
    tree = cKDTree(xyz_denoised)
    _, nn_idx = tree.query(xyz_down)
    sem_down = sem_denoised[nn_idx]

    # Stages 4-5: Ground removal, clustering, filtering
    ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)

    object_mask = np.ones(len(xyz_down), dtype=bool)
    object_mask[ground_inliers] = False
    xyz_obj = xyz_down[object_mask]
    sem_obj = sem_down[object_mask]

    if len(xyz_obj) == 0:
        return [], 0

    cluster_labels = cluster_objects(objects_pcd)
    assert len(cluster_labels) == len(xyz_obj), (
        f"cluster/object length mismatch: {len(cluster_labels)} vs {len(xyz_obj)}")
    clusters = filter_clusters(objects_pcd, cluster_labels, ground_plane)

    results = []
    n_discarded = 0
    for pcd_cluster, cl in clusters:
        cluster_mask = cluster_labels == cl
        cluster_points = xyz_obj[cluster_mask]
        cluster_sem = sem_obj[cluster_mask]

        cls_name, purity, raw_sem_purity, histogram = classify_cluster(
            cluster_sem, purity_threshold)

        if cls_name is None:
            n_discarded += 1
            continue

        centroid = cluster_points.mean(axis=0)
        points_centered = (cluster_points - centroid).astype(np.float32)

        metadata = {
            "class": cls_name,
            "purity": round(float(purity), 3),
            "raw_sem_purity": round(float(raw_sem_purity), 3),
            "n_points": len(cluster_points),
            "centroid": centroid.tolist(),
            "histogram": histogram,
        }
        results.append((points_centered, cls_name, metadata))

    return results, n_discarded


def main():
    """Mine real LiDAR clusters from SemanticKITTI for Stage B training.

    Iterates over frames in the specified sequences, extracts clusters
    that pass the purity threshold, and saves centroid-centered point
    clouds as ``.npy`` files organized by class.  Writes a comprehensive
    metadata JSON for reproducibility.
    """
    parser = argparse.ArgumentParser(
        description="Mine real LiDAR clusters for Stage B training")
    parser.add_argument("--seq", nargs="+", default=["00"],
                        help="Sequence IDs")
    parser.add_argument("--frames", type=int, default=100,
                        help="Max frames per sequence")
    parser.add_argument("--purity", type=float, default=0.75,
                        help="Minimum label purity to accept a cluster")
    parser.add_argument("--split", default="train",
                        choices=["train", "val"],
                        help="Output split directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output root (default: dataset/stage_b)")
    args = parser.parse_args()

    output_root = args.output_dir or os.path.join(PROJECT_ROOT, "dataset", "stage_b")

    class_counts = {cls: 0 for cls in VALID_CLASSES}
    all_metadata = []
    total_clusters_seen = 0
    total_discarded = 0

    for seq in args.seq:
        seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
        bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
        label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))

        n_frames = min(args.frames, len(bin_paths), len(label_paths))
        bin_paths = bin_paths[:n_frames]
        label_paths = label_paths[:n_frames]

        print(f"\nSequence {seq}: {n_frames} frames")

        for frame_idx in tqdm(range(n_frames), desc=f"seq {seq}"):
            results, n_discarded = mine_frame(
                bin_paths[frame_idx], label_paths[frame_idx], args.purity)
            total_discarded += n_discarded
            total_clusters_seen += len(results) + n_discarded

            for points_centered, cls_name, meta in results:
                cls_dir = os.path.join(output_root, args.split, cls_name)
                os.makedirs(cls_dir, exist_ok=True)

                filename = f"{seq}_{frame_idx:06d}_{class_counts[cls_name]}.npy"
                np.save(os.path.join(cls_dir, filename), points_centered)

                meta["file"] = os.path.join(args.split, cls_name, filename)
                meta["seq"] = seq
                meta["frame"] = frame_idx
                all_metadata.append(meta)
                class_counts[cls_name] += 1


    # Save metadata
    os.makedirs(output_root, exist_ok=True)
    meta_path = os.path.join(output_root, f"metadata_{args.split}.json")
    with open(meta_path, "w") as f:
        json.dump({
            "config": {
                "sequences": args.seq,
                "frames_per_seq": args.frames,
                "purity_threshold": args.purity,
                "split": args.split,
                "pipeline_config": {k: v for k, v in PIPELINE_CONFIG.items()},
            },
            "class_counts": class_counts,
            "total_clusters_seen": total_clusters_seen,
            "total_discarded": total_discarded,
            "clusters": all_metadata,
        }, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Stage B mining complete")
    print(f"  Output: {output_root}/{args.split}/")
    print(f"  Purity threshold: {args.purity}")
    print(f"  Clusters seen: {total_clusters_seen}")
    print(f"  Discarded (below purity): {total_discarded}")
    print(f"  Class counts:")
    total = 0
    for cls in VALID_CLASSES:
        print(f"    {cls:12s}: {class_counts[cls]:5d}")
        total += class_counts[cls]
    print(f"    {'total':12s}: {total:5d}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
