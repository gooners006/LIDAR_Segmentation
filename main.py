import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tracker import CentroidTracker

z_threshold = -2.0
voxel_size = 0.05
# RANSAC parameters
distance_threshold = 0.2  # Max distance to plane (m)
ransac_n = 3  # Number of points that are randomly sampled to estimate a plane
num_iterations = 1000  # RANSAC iterations
eps = 0.5  # 50 cm
min_points = 10  # Minimum points to form a cluster
min_volume_threshold = 0.5  # Minimum volume in cubic meters (m^3)
max_volume_threshold = 100  # Maximum volume in cubic meters


# 1. Setup File Paths
bin_paths = sorted(glob.glob("dataset/sequences/00/velodyne/*.bin"))[:100]

# 2. Initialize the Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="LiDAR Tracking Stream")

# Setup render options
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 2.0

# 3. Initialize Tracker
tracker = CentroidTracker(max_distance=2.0, max_disappeared=5)

# 4. Main Loop
if __name__ == "__main__":
    print(f"Starting playback for {len(bin_paths)} frames...")

    for frame_idx, bin_path in enumerate(bin_paths):

        # --- A. Load Data ---
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        # Create base PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # --- B. Pipeline Execution ---

        # Step 1: Filter & Downsample
        mask = xyz[:, 2] > z_threshold
        xyz_filtered = xyz[mask]
        pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
        pcd_denoised, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        pcd_down = pcd_denoised.voxel_down_sample(voxel_size=voxel_size)

        # Step 2: Ground Segmentation
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        [a, b, c, d] = plane_model
        if abs(c) < 0.5:
            print("WARNING: The detected plane looks like a WALL, not the ground!")
            print("Recommendation: Adjust RANSAC parameters or crop the ROI further.")
        ground_pcd = pcd_down.select_by_index(inliers)
        objects_pcd = pcd_down.select_by_index(inliers, invert=True)
        ground_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray ground
        objects_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red objects

        # Step 3: Clustering
        labels = np.array(
            objects_pcd.cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=True
            )
        )
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # Noise set to black
        objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Step 4: Bounding Boxes
        bbox_objects = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = np.asarray(labels == label).nonzero()[0]
            cluster_pcd = objects_pcd.select_by_index(cluster_indices)

            points_in_cluster = np.asarray(cluster_pcd.points)
            if len(points_in_cluster) < 15:
                continue
            bbox = cluster_pcd.get_oriented_bounding_box()
            bbox.color = (0, 1, 0)

            volume = bbox.volume()
            if volume < min_volume_threshold or volume > max_volume_threshold:
                continue
            extent = bbox.extent
            sorted_ext = np.sort(extent)
            min_dim, med_dim, max_dim = sorted_ext[0], sorted_ext[1], sorted_ext[2]
            if max_dim > 8.0:
                continue
            if max_dim < 0.5 or med_dim < 0.2:
                continue
            center = bbox.get_center()
            if center[2] > 0.5:
                continue
            # If it passes the checks, add it to our list
            bbox_objects.append(bbox)

        # Step 5: Tracking
        # Use the tracker to update IDs
        tracked_objects = tracker.update(bbox_objects)

        # --- C. Visualization Update ---
        # The visualizer holds "geometry" objects.
        # We need to clear the old ones and add the new ones for this frame.

        vis.clear_geometries()

        # Add the points
        vis.add_geometry(ground_pcd)
        vis.add_geometry(objects_pcd)

        # Add the Bounding Boxes
        for bbox in bbox_objects:
            vis.add_geometry(bbox)

        # (Optional) Control the camera
        # ctr = vis.get_view_control()
        # ctr.rotate(5.0, 0.0) # Rotate camera slightly every frame for effect

        # Render this frame
        vis.poll_events()
        vis.update_renderer()

        # Print status to console
        print(f"Frame {frame_idx}: Tracking {len(tracker.objects)} objects")

        # Optional: Slow down if it's too fast
        # time.sleep(0.1)

    # Keep window open after loop finishes
    vis.run()
