import glob
import numpy as np
import open3d as o3d
import scipy.spatial
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

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


class CentroidTracker:
    def __init__(self, max_distance=2.0, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}  # {ID: np.array([x,y,z])}
        self.disappeared = {}  # {ID: count}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0  # Initialize counter
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, new_bboxes):
        # 1. Get centroids from new boxes
        if len(new_bboxes) == 0:
            # If no new detections, mark all existing objects as 'disappeared'
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([b.get_center() for b in new_bboxes])

        # If no tracking objects exist, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
            return self.objects

        # 2. Match Old to New
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Distance Matrix (Rows = Old, Cols = New)
        D = scipy.spatial.distance.cdist(np.array(object_centroids), input_centroids)

        # Hungarian Algorithm
        rows, cols = linear_sum_assignment(D)

        # Sets to keep track of what we have used
        used_rows = set(rows)
        used_cols = set(cols)

        # 3. Update Matched Objects
        for row, col in zip(rows, cols):
            # Retrieve the id using the row index
            obj_id = object_ids[row]

            if D[row, col] < self.max_distance:
                self.objects[obj_id] = input_centroids[col]  # Update position
                self.disappeared[obj_id] = 0  # Reset counter
            else:
                self.disappeared[obj_id] += 1  # Match found but too far -> Lost

        # 4. Handle UNMATCHED Old Objects (The ones Hungarian alg skipped)
        all_rows = set(range(len(object_ids)))
        unused_rows = all_rows - used_rows

        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1

        # 5. Handle UNMATCHED New Detections (Register as new)
        all_cols = set(range(len(input_centroids)))
        unused_cols = all_cols - used_cols

        for col in unused_cols:
            self.register(input_centroids[col])

        # 6. Clean up dead tracks
        # Create a copy of keys to modify dict while iterating
        for obj_id in list(self.disappeared.keys()):
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        return self.objects


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
