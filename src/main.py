import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from classifier import classify_bbox
from pipeline import (
    PIPELINE_CONFIG,
    cluster_objects,
    filter_clusters,
    load_calib,
    load_poses,
    preprocess_frame,
    remove_ground,
)
from tracker import CentroidTracker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="LiDAR segmentation pipeline")
    parser.add_argument("--seq", default="00", help="Sequence ID")
    parser.add_argument("--frames", type=int, default=100, help="Max frames to process")
    parser.add_argument(
        "--save-output", action="store_true", help="Write .ply and tracks.json"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Disable Open3D visualization"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- File paths ---
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[
        : args.frames
    ]

    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    calib = load_calib(os.path.join(seq_dir, "calib.txt"))
    Tr = calib["Tr"]

    # --- Tracker ---
    tracker = CentroidTracker(
        max_distance=PIPELINE_CONFIG["tracker_max_distance"],
        max_disappeared=PIPELINE_CONFIG["tracker_max_disappeared"],
    )

    # --- Visualization ---
    vis = None
    pcd_geo_ground = o3d.geometry.PointCloud()
    pcd_geo_objects = o3d.geometry.PointCloud()
    prev_bboxes = []
    vis_initialized = False

    if not args.no_gui:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LiDAR Tracking Stream", width=1280, height=720)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 2.0

    # --- Accumulation state (for --save-output) ---
    track_points: dict[int, list[np.ndarray]] = {}
    track_classes: dict[int, str] = {}
    track_frames: dict[int, list[int]] = {}
    track_centroids: dict[int, list[list[float]]] = {}

    # --- Main loop ---
    print(f"Starting playback for {len(bin_paths)} frames...")
    cmap = plt.get_cmap("tab20")

    for frame_idx, bin_path in enumerate(bin_paths):

        # --- A. Load ---
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        # --- B. Steps 1-5 in local frame ---
        pcd_down = preprocess_frame(xyz)
        ground_pcd, objects_pcd, ground_plane, _ = remove_ground(pcd_down)
        labels = cluster_objects(objects_pcd)
        n_clusters = (
            int(labels.max() + 1) if len(labels) > 0 and labels.max() >= 0 else 0
        )
        clusters = filter_clusters(objects_pcd, labels, ground_plane)

        # --- C. Step 6: Classification (local frame) ---
        bbox_objects = []
        bbox_cluster_labels = []
        cluster_classes = []

        for bbox, cluster_label in clusters:
            result = classify_bbox(bbox.extent, bbox.get_center())
            cluster_classes.append(result.label)

            bbox_cluster_labels.append(cluster_label)
            bbox_objects.append(bbox)

        # --- D. Transform to global frame ---
        T_total = poses[frame_idx] @ Tr

        R_total = T_total[:3, :3]
        t_total = T_total[:3, 3]
        for bbox in bbox_objects:
            bbox.rotate(R_total, center=np.zeros(3))
            bbox.translate(t_total)

        ground_pcd.transform(T_total)
        objects_pcd.transform(T_total)
        ground_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # --- E. Tracking (global frame) ---
        tracked_objects, assignments = tracker.update(bbox_objects)

        # Color by track ID
        point_colors = np.zeros((len(labels), 3))

        for bbox_idx, track_id in assignments.items():
            color = np.array(cmap(track_id % 20)[:3])
            cluster_label = bbox_cluster_labels[bbox_idx]
            point_colors[labels == cluster_label] = color
            bbox_objects[bbox_idx].color = color

            # Accumulate for output
            if args.save_output:
                if track_id not in track_points:
                    track_points[track_id] = []
                    track_classes[track_id] = cluster_classes[bbox_idx]
                    track_frames[track_id] = []
                    track_centroids[track_id] = []

                cluster_pts = np.asarray(objects_pcd.points)[labels == cluster_label]
                track_points[track_id].append(cluster_pts)
                track_frames[track_id].append(frame_idx)
                track_centroids[track_id].append(
                    bbox_objects[bbox_idx].get_center().tolist()
                )

        objects_pcd.colors = o3d.utility.Vector3dVector(point_colors)

        # --- F. Visualization ---
        if vis is not None:
            pcd_geo_ground.points = ground_pcd.points
            pcd_geo_ground.colors = ground_pcd.colors
            pcd_geo_objects.points = objects_pcd.points
            pcd_geo_objects.colors = objects_pcd.colors

            if not vis_initialized:
                vis.add_geometry(pcd_geo_ground)
                vis.add_geometry(pcd_geo_objects)
                vis_initialized = True
            else:
                vis.update_geometry(pcd_geo_ground)
                vis.update_geometry(pcd_geo_objects)

            for bbox in prev_bboxes:
                vis.remove_geometry(bbox, reset_bounding_box=False)
            for bbox in bbox_objects:
                vis.add_geometry(bbox, reset_bounding_box=False)
            prev_bboxes = bbox_objects

            ctr = vis.get_view_control()
            ego_pos = T_total[:3, 3]
            ctr.set_lookat(ego_pos)
            ctr.set_zoom(0.15)
            if frame_idx == 0:
                ctr.rotate(-200.0, 150.0)

            vis.poll_events()
            vis.update_renderer()

        # --- G. Print ---
        class_counts: dict[str, int] = {}
        for cls in cluster_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print(
            f"Frame {frame_idx:03d}: {n_clusters} clusters, "
            f"{len(bbox_objects)} passed, "
            f"tracking {len(tracker.objects)} | classes: {class_counts}"
        )

    # --- Output writing ---
    if args.save_output:
        output_dir = os.path.join(PROJECT_ROOT, f"output/{args.seq}/objects")
        os.makedirs(output_dir, exist_ok=True)

        tracks_meta = []
        for track_id, pts_list in track_points.items():
            all_pts = np.vstack(pts_list)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            o3d.io.write_point_cloud(os.path.join(output_dir, f"{track_id}.ply"), pcd)

            tracks_meta.append(
                {
                    "track_id": track_id,
                    "class": track_classes[track_id],
                    "first_frame": min(track_frames[track_id]),
                    "last_frame": max(track_frames[track_id]),
                    "point_count": len(all_pts),
                    "centroid_history": track_centroids[track_id],
                }
            )

        meta = {
            "sequence": args.seq,
            "frames_processed": len(bin_paths),
            "tracks": tracks_meta,
        }
        json_path = os.path.join(PROJECT_ROOT, f"output/{args.seq}/tracks.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {len(tracks_meta)} tracks to {output_dir}")

    if vis is not None:
        vis.run()


if __name__ == "__main__":
    main()
