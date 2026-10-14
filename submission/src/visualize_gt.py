"""Visualize SemanticKITTI: toggle between GT labels and pipeline output.

Press SPACE to toggle between GT coloring and pipeline coloring.

Usage:
    python src/visualize_gt.py --seq 08 --frames 100
    python src/visualize_gt.py --seq 08 --frames 100 --gt-only
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from classifier import classify_cluster, load_classifier
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

SEMKITTI_COLORS = {
    10: [0.39, 0.58, 0.93],     # car
    11: [0.0, 0.0, 0.55],       # bicycle
    13: [1.0, 0.65, 0.0],       # bus
    15: [1.0, 0.0, 0.0],        # motorcycle
    18: [1.0, 1.0, 0.0],        # truck
    20: [1.0, 0.0, 1.0],        # other-vehicle
    30: [1.0, 0.75, 0.8],       # person
    31: [0.0, 1.0, 1.0],        # bicyclist
    32: [0.5, 0.0, 0.5],        # motorcyclist
    40: [0.3, 0.3, 0.3],        # road
    44: [0.35, 0.35, 0.35],     # parking
    48: [0.25, 0.25, 0.25],     # sidewalk
    50: [0.2, 0.4, 0.2],        # building
    51: [0.4, 0.4, 0.2],        # fence
    70: [0.0, 0.6, 0.0],        # vegetation
    71: [0.2, 0.3, 0.0],        # trunk
    72: [0.3, 0.5, 0.1],        # terrain
    80: [0.8, 0.8, 0.8],        # pole
    81: [0.7, 0.7, 0.7],        # traffic-sign
    252: [0.39, 0.58, 0.93],    # moving-car
    253: [0.0, 1.0, 1.0],       # moving-bicyclist
    254: [1.0, 0.75, 0.8],      # moving-person
    255: [1.0, 0.0, 0.0],       # moving-motorcyclist
    257: [1.0, 0.65, 0.0],      # moving-bus
    258: [1.0, 1.0, 0.0],       # moving-truck
    259: [1.0, 0.0, 1.0],       # moving-other-vehicle
}

DEFAULT_COLOR = [0.15, 0.15, 0.15]


class ViewState:
    def __init__(self):
        self.show_gt = True
        self.colors_gt = None
        self.colors_pipe = None
        self.pcd = None
        self.vis = None

    def toggle(self, vis):
        self.show_gt = not self.show_gt
        mode = "GT labels" if self.show_gt else "Pipeline"
        print(f"\r[{mode}]          ", end="", flush=True)
        if self.pcd is not None:
            colors = self.colors_gt if self.show_gt else self.colors_pipe
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(self.pcd)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs pipeline")
    parser.add_argument("--seq", type=str, default="08")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--gt-only", action="store_true",
                        help="Show only GT labels, no pipeline comparison")
    args = parser.parse_args()

    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:args.frames]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[:args.frames]

    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    calib = load_calib(os.path.join(seq_dir, "calib.txt"))
    Tr = calib["Tr"]

    compare = not args.gt_only

    if compare:
        import torch
        from scipy.spatial import cKDTree
        cfg = PIPELINE_CONFIG
        cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth")
        cls_model, cls_bbox_stats = load_classifier(default_ckpt, cls_device)
        tracker = CentroidTracker(
            max_distance=cfg["tracker_max_distance"],
            max_disappeared=cfg["tracker_max_disappeared"],
        )
        cmap = plt.get_cmap("tab20")

    state = ViewState()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("GT vs Pipeline (SPACE to toggle)", width=1600, height=900)
    vis.register_key_callback(ord(" "), state.toggle)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    pcd_geo = o3d.geometry.PointCloud()
    state.pcd = pcd_geo
    prev_bboxes = []
    vis_initialized = False

    print(f"Playing {len(bin_paths)} frames from seq {args.seq}...")
    if compare:
        print("Press SPACE to toggle between GT labels and pipeline output")

    for frame_idx in range(len(bin_paths)):
        points = np.fromfile(bin_paths[frame_idx], dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]

        raw_labels = np.fromfile(label_paths[frame_idx], dtype=np.uint32)
        sem = (raw_labels & 0xFFFF).astype(np.int32)
        instance_ids = (raw_labels >> 16).astype(np.int32)

        T_total = poses[frame_idx] @ Tr

        # --- Count GT object instances ---
        SEMKITTI_CLASS_NAMES = {
            10: "car", 13: "bus", 15: "motorcycle", 18: "truck",
            20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist",
            252: "car", 253: "bicyclist", 254: "person",
            255: "motorcycle", 257: "bus", 258: "truck", 259: "other-vehicle",
        }
        gt_counts: dict[str, int] = {}
        gt_instance_set: set[tuple[int, int]] = set()
        for label_id, cls_name in SEMKITTI_CLASS_NAMES.items():
            mask = sem == label_id
            if not mask.any():
                continue
            for inst in np.unique(instance_ids[mask]):
                key = (label_id, int(inst))
                if key not in gt_instance_set:
                    gt_instance_set.add(key)
                    gt_counts[cls_name] = gt_counts.get(cls_name, 0) + 1

        # --- GT colors for all raw points ---
        colors_gt = np.tile(DEFAULT_COLOR, (len(xyz), 1))
        for label_id, color in SEMKITTI_COLORS.items():
            mask = sem == label_id
            if mask.any():
                colors_gt[mask] = color

        xyz_global = (T_total[:3, :3] @ xyz.T).T + T_total[:3, 3]

        # --- Pipeline colors (same raw points, different coloring) ---
        if compare:
            colors_pipe = np.tile(DEFAULT_COLOR, (len(xyz), 1))

            pcd_down = preprocess_frame(xyz)
            ground_pcd, objects_pcd, ground_plane, ground_inliers = remove_ground(pcd_down)
            labels = cluster_objects(objects_pcd)
            clusters = filter_clusters(objects_pcd, labels, ground_plane)

            # Map pipeline results back to raw points via nearest-neighbor
            xyz_down = np.asarray(pcd_down.points)
            tree_raw = cKDTree(xyz)
            _, nn_raw = tree_raw.query(xyz_down)

            # Color ground points
            ground_pts = np.asarray(ground_pcd.points)
            if len(ground_pts) > 0:
                _, nn_ground = tree_raw.query(ground_pts)
                colors_pipe[nn_ground] = [0.3, 0.3, 0.3]

            # Classify and track
            bbox_objects = []
            bbox_cluster_labels = []
            cluster_classes = []
            for bbox, cluster_label in clusters:
                mask_cl = labels == cluster_label
                cluster_points = np.asarray(objects_pcd.points)[mask_cl]
                result = classify_cluster(
                    cluster_points, cls_model, cls_device, cls_bbox_stats,
                    unknown_threshold=0.50,
                )
                cluster_classes.append(result.label)
                bbox_cluster_labels.append(cluster_label)
                bbox_objects.append(bbox)

            R_total = T_total[:3, :3]
            t_total = T_total[:3, 3]
            for bbox in bbox_objects:
                bbox.rotate(R_total, center=np.zeros(3))
                bbox.translate(t_total)

            tracked_objects, assignments = tracker.update(bbox_objects)

            # Color tracked clusters on the raw point cloud
            obj_pts = np.asarray(objects_pcd.points)
            for bbox_idx, track_id in assignments.items():
                color = np.array(cmap(track_id % 20)[:3])
                cl = bbox_cluster_labels[bbox_idx]
                cluster_mask = labels == cl
                cluster_pts_local = obj_pts[cluster_mask]
                if len(cluster_pts_local) > 0:
                    _, nn_cl = tree_raw.query(cluster_pts_local)
                    colors_pipe[nn_cl] = color
                bbox_objects[bbox_idx].color = color

            state.colors_pipe = colors_pipe
        else:
            bbox_objects = []

        state.colors_gt = colors_gt
        active_colors = colors_gt if state.show_gt else colors_pipe if compare else colors_gt

        pcd_geo.points = o3d.utility.Vector3dVector(xyz_global)
        pcd_geo.colors = o3d.utility.Vector3dVector(active_colors)

        if not vis_initialized:
            vis.add_geometry(pcd_geo)
            vis_initialized = True
        else:
            vis.update_geometry(pcd_geo)

        # Bboxes only in pipeline mode
        if compare:
            for bbox in prev_bboxes:
                vis.remove_geometry(bbox, reset_bounding_box=False)
            if not state.show_gt:
                for bbox in bbox_objects:
                    vis.add_geometry(bbox, reset_bounding_box=False)
                prev_bboxes = bbox_objects
            else:
                prev_bboxes = []

        ctr = vis.get_view_control()
        ego_pos = T_total[:3, 3]
        ctr.set_lookat(ego_pos)
        ctr.set_zoom(0.15)
        if frame_idx == 0:
            ctr.rotate(-200.0, 150.0)

        vis.poll_events()
        vis.update_renderer()

        # --- Log ---
        gt_str = ", ".join(f"{v} {k}" for k, v in sorted(gt_counts.items())) or "none"
        if compare:
            det_counts: dict[str, int] = {}
            for cls in cluster_classes:
                det_counts[cls] = det_counts.get(cls, 0) + 1
            det_str = ", ".join(f"{v} {k}" for k, v in sorted(det_counts.items())) or "none"
            mode = "GT" if state.show_gt else "Pipeline"
            print(f"Frame {frame_idx:03d} [{mode}] | detected: {det_str} | GT: {gt_str}")
        else:
            print(f"Frame {frame_idx:03d} | GT: {gt_str}")

    print("\nDone. Press SPACE to toggle, close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
