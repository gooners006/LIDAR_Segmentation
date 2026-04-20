import numpy as np
import open3d as o3d
import hdbscan
from typing import List, Tuple

PIPELINE_CONFIG = {
    "z_threshold": -2.0,
    "voxel_size": 0.05,
    "denoise_nb_neighbors": 20,
    "denoise_std_ratio": 2.0,
    "ransac_distance_threshold": 0.2,
    "ransac_n": 3,
    "ransac_iterations": 1000,
    "ransac_min_normal_z": 0.5,
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 5,
    "min_points_in_cluster": 15,
    "min_volume": 0.5,
    "max_volume": 100.0,
    "max_dim_length": 8.0,
    "min_max_dim": 0.5,
    "min_med_dim": 0.2,
    "max_center_height_above_ground": 3.0,
    "tracker_max_distance": 2.0,
    "tracker_max_disappeared": 5,
}


def load_calib(path: str) -> dict:
    data = {}
    with open(path, "r") as f:
        for line in f:
            key, _, value = line.partition(":")
            if value.strip():
                vals = np.array([float(x) for x in value.strip().split()])
                if len(vals) == 12:
                    M = np.eye(4)
                    M[:3, :] = vals.reshape(3, 4)
                    data[key.strip()] = M
    return data


def load_poses(path: str) -> list:
    poses = []
    with open(path, "r") as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split()])
            T = np.eye(4)
            T[:3, :] = values.reshape(3, 4)
            poses.append(T)
    return poses


def preprocess_frame(
    xyz: np.ndarray, config: dict = PIPELINE_CONFIG
) -> o3d.geometry.PointCloud:
    """Steps 1-2: Z-filter, statistical outlier removal, voxel downsample."""
    mask = xyz[:, 2] > config["z_threshold"]
    xyz_filtered = xyz[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
    pcd_denoised, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config["denoise_nb_neighbors"],
        std_ratio=config["denoise_std_ratio"],
    )
    return pcd_denoised.voxel_down_sample(voxel_size=config["voxel_size"])


def remove_ground(
    pcd: o3d.geometry.PointCloud, config: dict = PIPELINE_CONFIG
) -> tuple:
    """Step 3: RANSAC ground removal with normal check.

    Returns:
        (ground_pcd, objects_pcd, plane_model, inlier_indices) where plane_model
        is (a, b, c, d) for ax+by+cz+d=0 (or None if rejected), and
        inlier_indices is the list of ground point indices into the input cloud.
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=config["ransac_distance_threshold"],
        ransac_n=config["ransac_n"],
        num_iterations=config["ransac_iterations"],
    )
    a, b, c, d = plane_model
    normal_len = np.sqrt(a**2 + b**2 + c**2)

    if abs(c) / normal_len < config["ransac_min_normal_z"]:
        return o3d.geometry.PointCloud(), pcd, None, []

    ground_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    return ground_pcd, objects_pcd, tuple(plane_model), inliers


def cluster_objects(
    objects_pcd: o3d.geometry.PointCloud, config: dict = PIPELINE_CONFIG
) -> np.ndarray:
    """Step 4: HDBSCAN clustering (density-adaptive). Returns label array (-1 = noise)."""
    points = np.asarray(objects_pcd.points)
    if len(points) == 0:
        return np.array([], dtype=np.int32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["hdbscan_min_cluster_size"],
        min_samples=config["hdbscan_min_samples"],
    )
    return clusterer.fit_predict(points)


def filter_clusters(
    objects_pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    ground_plane=None,
    config: dict = PIPELINE_CONFIG,
) -> List[Tuple[o3d.geometry.OrientedBoundingBox, int]]:
    """Step 5: OBB extraction + geometric filtering. Returns [(bbox, cluster_label), ...]."""
    if ground_plane is not None:
        gp_a, gp_b, gp_c, gp_d = ground_plane
        gp_norm = np.sqrt(gp_a**2 + gp_b**2 + gp_c**2)

    results = []
    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_indices = np.asarray(labels == label).nonzero()[0]
        cluster_pcd = objects_pcd.select_by_index(cluster_indices)

        if len(cluster_pcd.points) < config["min_points_in_cluster"]:
            continue

        bbox = cluster_pcd.get_oriented_bounding_box()

        volume = bbox.volume()
        if volume < config["min_volume"] or volume > config["max_volume"]:
            continue

        sorted_ext = np.sort(bbox.extent)
        min_dim, med_dim, max_dim = sorted_ext[0], sorted_ext[1], sorted_ext[2]

        if max_dim > config["max_dim_length"]:
            continue
        if max_dim < config["min_max_dim"] or med_dim < config["min_med_dim"]:
            continue

        center = bbox.get_center()
        if ground_plane is not None:
            height = (gp_a * center[0] + gp_b * center[1] + gp_c * center[2] + gp_d) / gp_norm
        else:
            height = center[2]

        if height > config["max_center_height_above_ground"]:
            continue

        results.append((bbox, int(label)))

    return results
