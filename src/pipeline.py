import numpy as np
import open3d as o3d
import hdbscan
from scipy.ndimage import binary_dilation, binary_erosion, label as ndimage_label
from typing import List, Tuple

PIPELINE_CONFIG = {
    # preprocessing
    "z_threshold": -2.0,
    "voxel_size": 0.05,
    # denoising
    "denoise_nb_neighbors": 20,
    "denoise_std_ratio": 2.0,
    # ransac ground fitting
    "ransac_distance_threshold": 0.2,
    "ransac_n": 3,
    "ransac_iterations": 1000,
    "ransac_min_normal_z": 0.5,
    # clustering
    "clustering_method": "hdbscan",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 5,
    "bev_resolution": 0.2,
    "bev_morph_kernel": 3,
    # post-clustering fragment merge
    "merge_fragments": False,
    "merge_max_centroid_dist": 2.0,
    "merge_max_z_gap": 1.0,
    "merge_small_threshold": 50,
    # distance-adaptive HDBSCAN
    "adaptive_hdbscan": False,
    "adaptive_ranges": [0, 15, 30, 80],
    "adaptive_min_cluster_sizes": [30, 15, 10],
    # filtering
    "min_points_in_cluster": 15,
    "min_volume": 0.5,
    "max_volume": 50.0,
    "max_dim_length": 6.0,
    "min_max_dim": 0.5,
    "min_med_dim": 0.4,
    "max_center_height_above_ground": 1.5,
    "max_height_span": 1.8,
    "max_aspect_max_min": 6.0,
    # tracking
    "tracker_max_distance": 2.0,
    "tracker_max_disappeared": 5,
    # track-level filtering
    "min_track_length": 2,
    "track_class_vote": True,
    "min_track_known_votes": 2,
    "min_track_known_ratio": 0.5,
    # PCN completion
    "pcn_min_points": 64,
    "pcn_completion_classes": ["car"],
    "pcn_sample_seed": 0,
}


def load_calib(path: str) -> dict:
    """Parse a KITTI calibration file into 4x4 homogeneous matrices.

    Args:
        path: Path to the calibration file (e.g. ``calib.txt``).

    Returns:
        Dict mapping key names (e.g. ``"Tr"``) to 4x4 numpy arrays.
        Only entries with exactly 12 values (3x4 matrix) are included.
    """
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
    """Load per-frame camera poses from a KITTI pose file.

    Args:
        path: Path to the pose file (one 3x4 matrix per line, 12 floats).

    Returns:
        List of 4x4 numpy arrays, one per frame, representing the
        camera-to-world transformation.
    """
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
    """Preprocess a raw LiDAR frame (Steps 1--2).

    Applies a minimum-height z-filter, removes statistical outliers, and
    voxel-downsamples the result.

    Args:
        xyz: Raw point cloud as an (N, 3) array.
        config: Pipeline parameters dict.

    Returns:
        Cleaned and downsampled Open3D point cloud.
    """
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
    """Separate ground points from objects via RANSAC plane fitting (Step 3).

    Fits a plane using RANSAC and checks that its normal is approximately
    vertical (z-component above ``ransac_min_normal_z``). If the check
    fails, the plane is rejected and all points are returned as objects.

    Args:
        pcd: Preprocessed point cloud.
        config: Pipeline parameters dict.

    Returns:
        Tuple of (ground_pcd, objects_pcd, plane_model, inlier_indices).
        ``plane_model`` is ``(a, b, c, d)`` for the equation
        ``ax + by + cz + d = 0``, or ``None`` if the fitted plane was
        rejected.  ``inlier_indices`` lists ground-point indices into
        the input cloud.
    """
    points = np.asarray(pcd.points)
    n_pts = len(points)
    if n_pts < config["ransac_n"]:
        return o3d.geometry.PointCloud(), pcd, None, []

    rng = np.random.default_rng(42)
    dist_thresh = config["ransac_distance_threshold"]
    n_sample = config["ransac_n"]
    n_iter = config["ransac_iterations"]
    min_normal_z = config["ransac_min_normal_z"]

    best_inliers = []
    best_plane = None
    for _ in range(n_iter):
        idx = rng.choice(n_pts, size=n_sample, replace=False)
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -normal.dot(p0)
        dists = np.abs(points @ normal + d)
        inlier_mask = dists < dist_thresh
        n_inliers = inlier_mask.sum()
        if n_inliers > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0].tolist()
            best_plane = (normal[0], normal[1], normal[2], d)

    if best_plane is None:
        return o3d.geometry.PointCloud(), pcd, None, []

    a, b, c, d = best_plane
    normal_len = np.sqrt(a**2 + b**2 + c**2)
    if abs(c) / normal_len < min_normal_z:
        return o3d.geometry.PointCloud(), pcd, None, []

    ground_pcd = pcd.select_by_index(best_inliers)
    objects_pcd = pcd.select_by_index(best_inliers, invert=True)
    return ground_pcd, objects_pcd, best_plane, best_inliers


def _cluster_hdbscan(points: np.ndarray, config: dict) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["hdbscan_min_cluster_size"],
        min_samples=config["hdbscan_min_samples"],
    )
    return clusterer.fit_predict(points)


def _cluster_adaptive_hdbscan(points: np.ndarray, config: dict) -> np.ndarray:
    """Run HDBSCAN with different min_cluster_size per distance ring."""
    ranges = config["adaptive_ranges"]
    mcs_list = config["adaptive_min_cluster_sizes"]
    min_samples = config["hdbscan_min_samples"]

    xy_range = np.linalg.norm(points[:, :2], axis=1)
    labels = np.full(len(points), -1, dtype=np.int32)
    next_label = 0

    for i in range(len(ranges) - 1):
        ring_mask = (xy_range >= ranges[i]) & (xy_range < ranges[i + 1])
        ring_pts = points[ring_mask]
        if len(ring_pts) < mcs_list[i]:
            continue

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs_list[i],
            min_samples=min_samples,
        )
        ring_labels = clusterer.fit_predict(ring_pts)

        valid = ring_labels >= 0
        ring_labels[valid] += next_label
        labels[ring_mask] = ring_labels
        if valid.any():
            next_label = ring_labels[valid].max() + 1

    return labels


def _merge_nearby_clusters(points: np.ndarray, labels: np.ndarray, config: dict) -> np.ndarray:
    """Merge small clusters into nearby larger ones to recover split objects."""
    max_dist = config["merge_max_centroid_dist"]
    max_z_gap = config["merge_max_z_gap"]
    small_thresh = config["merge_small_threshold"]

    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if len(unique_labels) < 2:
        return labels

    cluster_info = {}
    for cl in unique_labels:
        mask = labels == cl
        pts = points[mask]
        cluster_info[cl] = {
            "centroid": pts.mean(axis=0),
            "n_pts": len(pts),
            "z_min": pts[:, 2].min(),
            "z_max": pts[:, 2].max(),
        }

    small_cls = [cl for cl in unique_labels if cluster_info[cl]["n_pts"] <= small_thresh]
    large_cls = [cl for cl in unique_labels if cluster_info[cl]["n_pts"] > small_thresh]

    if not small_cls or not large_cls:
        return labels

    large_centroids = np.array([cluster_info[cl]["centroid"] for cl in large_cls])

    labels = labels.copy()
    for scl in small_cls:
        s_info = cluster_info[scl]
        dists = np.linalg.norm(large_centroids - s_info["centroid"], axis=1)
        order = np.argsort(dists)

        for idx in order:
            if dists[idx] > max_dist:
                break
            lcl = large_cls[idx]
            l_info = cluster_info[lcl]
            z_gap = max(s_info["z_min"] - l_info["z_max"], l_info["z_min"] - s_info["z_max"], 0)
            if z_gap <= max_z_gap:
                labels[labels == scl] = lcl
                cluster_info[lcl]["n_pts"] += s_info["n_pts"]
                all_pts = points[labels == lcl]
                cluster_info[lcl]["centroid"] = all_pts.mean(axis=0)
                cluster_info[lcl]["z_min"] = all_pts[:, 2].min()
                cluster_info[lcl]["z_max"] = all_pts[:, 2].max()
                break

    return labels


def _cluster_bev(points: np.ndarray, config: dict) -> np.ndarray:
    """BEV grid clustering via connected-component labeling.

    Projects points onto a 2D x-y grid, applies morphological opening
    (erosion then dilation) to break thin bridges between adjacent
    objects, then labels connected components.
    """
    res = config["bev_resolution"]
    kernel_size = config["bev_morph_kernel"]

    xy = points[:, :2]
    xy_min = xy.min(axis=0)
    ij = ((xy - xy_min) / res).astype(np.int32)
    h = ij[:, 0].max() + 1
    w = ij[:, 1].max() + 1

    grid = np.zeros((h, w), dtype=np.uint8)
    grid[ij[:, 0], ij[:, 1]] = 1

    if kernel_size > 0:
        struct = np.ones((kernel_size, kernel_size), dtype=bool)
        opened = binary_erosion(grid, structure=struct)
        opened = binary_dilation(opened, structure=struct).astype(np.uint8)
    else:
        opened = grid

    labeled_grid, _ = ndimage_label(opened)

    labels = labeled_grid[ij[:, 0], ij[:, 1]] - 1
    return labels.astype(np.int32)


def cluster_objects(
    objects_pcd: o3d.geometry.PointCloud, config: dict = PIPELINE_CONFIG
) -> np.ndarray:
    """Cluster non-ground points into object candidates (Step 4).

    Dispatches to HDBSCAN or BEV clustering based on
    ``config["clustering_method"]``.

    Args:
        objects_pcd: Non-ground point cloud from :func:`remove_ground`.
        config: Pipeline parameters dict.

    Returns:
        Integer label array aligned with ``objects_pcd`` points.
        Each element is a cluster ID (>= 0) or -1 for noise.
    """
    points = np.asarray(objects_pcd.points)
    if len(points) == 0:
        return np.array([], dtype=np.int32)

    method = config.get("clustering_method", "hdbscan")
    if method == "bev":
        labels = _cluster_bev(points, config)
    elif config.get("adaptive_hdbscan", False):
        labels = _cluster_adaptive_hdbscan(points, config)
    else:
        labels = _cluster_hdbscan(points, config)

    if config.get("merge_fragments", False):
        labels = _merge_nearby_clusters(points, labels, config)

    return labels


def _center_height_above_ground(center: np.ndarray, ground_plane) -> float:
    """Compute signed distance from a point to the fitted ground plane.

    Falls back to the raw z-coordinate when no ground plane is available.

    Args:
        center: 3D point as a numpy array.
        ground_plane: ``(a, b, c, d)`` plane coefficients, or ``None``.

    Returns:
        Height above the ground plane in metres.
    """
    if ground_plane is None:
        return float(center[2])
    a, b, c, d = ground_plane
    norm = np.sqrt(a**2 + b**2 + c**2)
    return float((a * center[0] + b * center[1] + c * center[2] + d) / norm)


def _passes_geometric_filter(
    cluster_pcd: o3d.geometry.PointCloud,
    ground_plane,
    config: dict,
) -> o3d.geometry.OrientedBoundingBox | None:
    """Apply geometric filter checks to a single cluster.

    Checks (in order): minimum point count, OBB volume, dimension
    bounds, center height above ground, vertical span, and aspect ratio.

    Args:
        cluster_pcd: Point cloud of one cluster.
        ground_plane: ``(a, b, c, d)`` plane coefficients, or ``None``.
        config: Pipeline parameters dict.

    Returns:
        The cluster's oriented bounding box if all checks pass,
        otherwise ``None``.
    """
    if len(cluster_pcd.points) < config["min_points_in_cluster"]:
        return None

    bbox = cluster_pcd.get_oriented_bounding_box()

    volume = bbox.volume()
    if volume < config["min_volume"] or volume > config["max_volume"]:
        return None

    min_dim, med_dim, max_dim = np.sort(bbox.extent)

    if max_dim > config["max_dim_length"]:
        return None
    if max_dim < config["min_max_dim"] or med_dim < config["min_med_dim"]:
        return None

    height = _center_height_above_ground(bbox.get_center(), ground_plane)
    if height > config["max_center_height_above_ground"]:
        return None

    cluster_pts = np.asarray(cluster_pcd.points)
    height_span = cluster_pts[:, 2].max() - cluster_pts[:, 2].min()
    if height_span > config["max_height_span"]:
        return None

    aspect_max_min = max_dim / min_dim if min_dim > 1e-6 else float("inf")
    if aspect_max_min > config["max_aspect_max_min"]:
        return None

    return bbox


def filter_clusters(
    objects_pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    ground_plane=None,
    config: dict = PIPELINE_CONFIG,
) -> List[Tuple[o3d.geometry.OrientedBoundingBox, int]]:
    """Filter HDBSCAN clusters by oriented-bounding-box geometry (Step 5).

    Iterates over unique cluster labels, computes an OBB for each, and
    applies geometric checks via :func:`_passes_geometric_filter`.

    Args:
        objects_pcd: Non-ground point cloud (same as passed to
            :func:`cluster_objects`).
        labels: Cluster label array from :func:`cluster_objects`.
        ground_plane: ``(a, b, c, d)`` plane coefficients from
            :func:`remove_ground`, or ``None``.
        config: Pipeline parameters dict.

    Returns:
        List of ``(bbox, cluster_label)`` tuples for clusters that
        survive all filters.
    """
    results = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        cluster_pcd = objects_pcd.select_by_index(cluster_indices)
        bbox = _passes_geometric_filter(cluster_pcd, ground_plane, config)
        if bbox is not None:
            results.append((bbox, int(label)))
    return results
