import glob
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tracker import CentroidTracker

# --- 1. Configuration & Parameters ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Preprocessing & Voxelization
Z_THRESHOLD = -2.0
VOXEL_SIZE = 0.05

# RANSAC (Local Frame)
RANSAC_DISTANCE_THRESHOLD = 0.2
RANSAC_N = 3
RANSAC_ITERATIONS = 1000

# Clustering (DBSCAN)
DBSCAN_EPS = 0.5
DBSCAN_MIN_POINTS = 10

# Bounding Box Heuristics
FILTER_CONFIG = {
    "min_points_in_cluster": 15,
    "min_volume": 0.2,  # m^3
    "max_volume": 100.0,  # m^3
    "max_dim_length": 8.0,  # Avoid long walls/fences
    "min_max_dim": 0.5,  # Smallest allowed 'longest' dimension
    "min_med_dim": 0.2,  # Smallest allowed 'medium' dimension
    "max_height_above_gnd": 3.0,
}

# --- 2. Setup File Paths ---
bin_paths = sorted(
    glob.glob(os.path.join(PROJECT_ROOT, "dataset/sequences/00/velodyne/*.bin"))
)[:100]

poses_path = os.path.join(PROJECT_ROOT, "dataset/sequences/00/poses.txt")
poses = []
with open(poses_path, "r") as f:
    for line in f:
        values = np.array([float(x) for x in line.strip().split()])
        T = np.eye(4)
        T[:3, :] = values.reshape(3, 4)
        poses.append(T)

# --- 3. Initialize Visualizer & Geometries ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="LiDAR Tracking Stream", width=1920, height=1440)

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 2.0

# Pre-allocate point cloud geometries for efficient updating
pcd_geo_ground = o3d.geometry.PointCloud()
pcd_geo_objects = o3d.geometry.PointCloud()
vis.add_geometry(pcd_geo_ground)
vis.add_geometry(pcd_geo_objects)

prev_bboxes = []  # To keep track of geometries we need to remove from the visualizer

# --- 4. Initialize Tracker ---
tracker = CentroidTracker(max_distance=5.0, max_disappeared=15)

# --- 5. Main Loop ---
if __name__ == "__main__":
    print(f"Starting playback for {len(bin_paths)} frames...")

    for frame_idx, bin_path in enumerate(bin_paths):

        # --- A. Load Data ---
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        # --- B. Local Frame Processing ---

        # Step 1: Filter Z and downsample in LOCAL frame
        mask = xyz[:, 2] > Z_THRESHOLD
        xyz_filtered = xyz[mask]

        pcd_local = o3d.geometry.PointCloud()
        pcd_local.points = o3d.utility.Vector3dVector(xyz_filtered)
        pcd_denoised, _ = pcd_local.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        pcd_down = pcd_denoised.voxel_down_sample(voxel_size=VOXEL_SIZE)

        # Step 2: Ground Segmentation in LOCAL frame
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_ITERATIONS,
        )

        [a, b, c, d] = plane_model
        ground_normal_len = np.sqrt(a**2 + b**2 + c**2)

        ground_pcd = pcd_down.select_by_index(inliers)
        objects_pcd = pcd_down.select_by_index(inliers, invert=True)

        # Step 3: Clustering
        # Turn off print_progress to avoid console spam
        labels = np.array(
            objects_pcd.cluster_dbscan(
                eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False
            )
        )

        # Step 4: Bounding Box Extraction & Filtering (Local)
        T = poses[frame_idx]
        bbox_objects = []
        bbox_cluster_labels = []

        unique_labels = np.unique(labels)
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        rejected = {
            "few_points": 0,
            "volume": 0,
            "max_dim": 0,
            "min_dim": 0,
            "height": 0,
        }

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.asarray(labels == label).nonzero()[0]
            cluster_pcd = objects_pcd.select_by_index(cluster_indices)

            # Filter: Minimum points
            if len(cluster_pcd.points) < FILTER_CONFIG["min_points_in_cluster"]:
                rejected["few_points"] += 1
                continue

            bbox = cluster_pcd.get_oriented_bounding_box()

            # Filter: Volume
            volume = bbox.volume()
            if (
                volume < FILTER_CONFIG["min_volume"]
                or volume > FILTER_CONFIG["max_volume"]
            ):
                rejected["volume"] += 1
                continue

            # Filter: Dimensions
            extent = bbox.extent
            sorted_ext = np.sort(extent)
            min_dim, med_dim, max_dim = sorted_ext[0], sorted_ext[1], sorted_ext[2]

            if max_dim > FILTER_CONFIG["max_dim_length"]:
                rejected["max_dim"] += 1
                continue
            if (
                max_dim < FILTER_CONFIG["min_max_dim"]
                or med_dim < FILTER_CONFIG["min_med_dim"]
            ):
                rejected["min_dim"] += 1
                continue

            # Filter: Height above local ground plane
            center = bbox.get_center()
            height_above_ground = (
                a * center[0] + b * center[1] + c * center[2] + d
            ) / ground_normal_len
            if height_above_ground > FILTER_CONFIG["max_height_above_gnd"]:
                rejected["height"] += 1
                continue

            # --- Transform passed Bounding Box to GLOBAL frame ---
            bbox.transform(T)
            bbox_objects.append(bbox)
            bbox_cluster_labels.append(label)

        # --- C. Transform Point Clouds to Global Frame ---
        ground_pcd.transform(T)
        objects_pcd.transform(T)

        ground_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # --- D. Tracking (Global Frame) ---
        tracked_objects, assignments = tracker.update(bbox_objects)

        # Step 6: Color global points and bboxes by track ID
        cmap = plt.get_cmap("tab20")
        point_colors = np.zeros((len(labels), 3))  # Default black for noise/untracked

        for bbox_idx, track_id in assignments.items():
            color = np.array(cmap(track_id % 20)[:3])
            cluster_label = bbox_cluster_labels[bbox_idx]
            point_colors[labels == cluster_label] = color
            bbox_objects[bbox_idx].color = color

        objects_pcd.colors = o3d.utility.Vector3dVector(point_colors)

        # --- E. Visualization Update ---

        # 1. Efficiently update point cloud geometry data
        pcd_geo_ground.points = ground_pcd.points
        pcd_geo_ground.colors = ground_pcd.colors

        pcd_geo_objects.points = objects_pcd.points
        pcd_geo_objects.colors = objects_pcd.colors

        vis.update_geometry(pcd_geo_ground)
        vis.update_geometry(pcd_geo_objects)

        # 2. Manage bounding box geometries manually
        for bbox in prev_bboxes:
            vis.remove_geometry(bbox, reset_bounding_box=False)

        for bbox in bbox_objects:
            vis.add_geometry(bbox, reset_bounding_box=False)

        prev_bboxes = bbox_objects

        # 3. Control camera (Follow ego vehicle in global frame)
        ctr = vis.get_view_control()
        ego_pos = T[:3, 3]
        ctr.set_lookat(ego_pos)
        ctr.set_zoom(0.15)

        # Initialize rotation only on the first frame to allow manual control later
        if frame_idx == 0:
            ctr.rotate(0.0, -350.0)

        vis.poll_events()
        vis.update_renderer()

        print(
            f"Frame {frame_idx:03d}: {n_clusters} clusters, {len(bbox_objects)} passed filters, "
            f"tracking {len(tracker.objects)} objects | rejected: {rejected}"
        )

    vis.run()
