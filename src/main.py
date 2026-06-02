import argparse
import glob
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from classifier import classify_bbox_heuristic, classify_cluster, load_classifier
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


def resolve_track_class(votes, *, min_known_votes=2, min_known_ratio=0.5):
    """Majority vote over non-rejected labels with evidence thresholds.

    Returns the winning class name, or None if evidence is insufficient
    (too few known votes, low known ratio, or ambiguous tie).
    """
    known = [v for v in votes if v != "not-car"]
    if len(known) < min_known_votes:
        return None
    if len(known) / len(votes) < min_known_ratio:
        return None
    counts: dict[str, int] = {}
    for v in known:
        counts[v] = counts.get(v, 0) + 1
    top_count = max(counts.values())
    winners = [cls for cls, c in counts.items() if c == top_count]
    if len(winners) > 1:
        return None
    return winners[0]


def parse_args():
    """Parse command-line arguments for the pipeline runner.

    Returns:
        Namespace with fields: seq, frames, save_output, no_gui,
        classifier_ckpt, classifier_unknown_threshold,
        no_learned_classifier.
    """
    parser = argparse.ArgumentParser(description="LiDAR segmentation pipeline")
    parser.add_argument("--seq", default="00", help="Sequence ID")
    parser.add_argument("--frames", type=int, default=100, help="Max frames to process")
    parser.add_argument(
        "--save-output", action="store_true", help="Write .ply and tracks.json"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Disable Open3D visualization"
    )
    parser.add_argument(
        "--classifier-ckpt", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "classifier_best.pth"),
        help="Path to learned classifier checkpoint",
    )
    parser.add_argument(
        "--classifier-unknown-threshold", type=float, default=0.50,
        help="Softmax threshold below which clusters are labeled not-car",
    )
    parser.add_argument(
        "--no-learned-classifier", action="store_true",
        help="Use heuristic classifier instead of learned model",
    )
    parser.add_argument(
        "--pcn-ckpt", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", "pcn_best.pth"),
        help="Path to PCN checkpoint for point cloud completion",
    )
    parser.add_argument(
        "--no-completion", action="store_true",
        help="Disable PCN point cloud completion",
    )
    parser.add_argument(
        "--mine-pairs", type=str, default=None, metavar="DIR",
        help="Mine sparse/dense completion pairs to DIR (implies --save-output --no-completion)",
    )
    return parser.parse_args()


def main():
    """Run the full LiDAR segmentation and tracking pipeline.

    Processes a KITTI sequence frame-by-frame through: preprocessing,
    ground removal, HDBSCAN clustering, geometric filtering,
    classification, and centroid tracking.  Optionally visualizes
    results in Open3D and/or saves per-track PLY files with a
    ``tracks.json`` manifest.
    """
    args = parse_args()

    if args.mine_pairs:
        args.save_output = True
        args.no_completion = True

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

    # --- Classifier ---
    cls_model, cls_bbox_stats = None, None
    cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.no_learned_classifier:
        cls_model, cls_bbox_stats = load_classifier(args.classifier_ckpt, cls_device)
        if cls_model is not None:
            print(f"Loaded learned classifier from {args.classifier_ckpt}")
        else:
            warnings.warn(
                f"Classifier checkpoint not found: {args.classifier_ckpt}. "
                "Falling back to heuristic classifier.")
    use_learned_classifier = cls_model is not None

    # --- Point completion ---
    from completion import PointCloudCompleter

    if args.save_output and not args.no_completion:
        if not os.path.isfile(args.pcn_ckpt):
            raise FileNotFoundError(
                f"PCN checkpoint not found: {args.pcn_ckpt}. "
                "Use --no-completion to disable, or provide a valid --pcn-ckpt."
            )
        completer = PointCloudCompleter(
            model_path=args.pcn_ckpt,
            seed=PIPELINE_CONFIG["pcn_sample_seed"],
        )
        print(f"Loaded PCN completer from {args.pcn_ckpt}")
    else:
        completer = PointCloudCompleter(seed=PIPELINE_CONFIG["pcn_sample_seed"])

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
    track_class_votes: dict[int, list[str]] = {}
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

        assert len(labels) == len(np.asarray(objects_pcd.points)), \
            "Cluster labels must align with objects_pcd.points"

        for bbox, cluster_label in clusters:
            if use_learned_classifier:
                mask = labels == cluster_label
                cluster_points = np.asarray(objects_pcd.points)[mask]
                result = classify_cluster(
                    cluster_points, cls_model, cls_device, cls_bbox_stats,
                    unknown_threshold=args.classifier_unknown_threshold,
                )
            else:
                result = classify_bbox_heuristic(bbox.extent, bbox.get_center())
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
                    track_class_votes[track_id] = []
                    track_frames[track_id] = []
                    track_centroids[track_id] = []

                track_class_votes[track_id].append(cluster_classes[bbox_idx])
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

        cfg = PIPELINE_CONFIG
        min_len = cfg["min_track_length"]
        use_vote = cfg["track_class_vote"]
        min_kv = cfg["min_track_known_votes"]
        min_kr = cfg["min_track_known_ratio"]

        n_total = len(track_points)
        n_short = 0
        n_rejected_class = 0

        pcn_classes = set(cfg["pcn_completion_classes"])
        pcn_min_pts = cfg["pcn_min_points"]
        completion_enabled = completer.is_loaded
        n_completed = 0

        tracks_meta = []
        for track_id, pts_list in track_points.items():
            if len(track_frames[track_id]) < min_len:
                n_short += 1
                continue

            if use_vote:
                resolved = resolve_track_class(
                    track_class_votes[track_id],
                    min_known_votes=min_kv,
                    min_known_ratio=min_kr,
                )
            else:
                resolved = track_class_votes[track_id][0]

            if resolved is None:
                n_rejected_class += 1
                continue

            all_pts = np.vstack(pts_list)

            if not completion_enabled:
                output_pts = all_pts
                skip_reason = "disabled"
            elif resolved not in pcn_classes:
                output_pts = all_pts
                skip_reason = "unsupported_class"
            elif len(all_pts) < pcn_min_pts:
                output_pts = all_pts
                skip_reason = "too_few_points"
            else:
                output_pts, skip_reason = completer.complete(all_pts, resolved)

            completed = skip_reason is None
            if completed:
                n_completed += 1

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(output_pts)
            o3d.io.write_point_cloud(os.path.join(output_dir, f"{track_id}.ply"), pcd)

            track_entry = {
                "track_id": track_id,
                "class": resolved,
                "first_frame": min(track_frames[track_id]),
                "last_frame": max(track_frames[track_id]),
                "point_count": int(len(output_pts)),
                "raw_point_count": int(len(all_pts)),
                "completed": completed,
                "completion_enabled": completion_enabled,
                "centroid_history": track_centroids[track_id],
            }
            if completion_enabled:
                track_entry["completion_method"] = "pcn"
                track_entry["pcn_checkpoint"] = args.pcn_ckpt
            if skip_reason is not None:
                track_entry["completion_skip_reason"] = skip_reason
            tracks_meta.append(track_entry)

        meta = {
            "sequence": args.seq,
            "frames_processed": len(bin_paths),
            "completion_enabled": completion_enabled,
            "pcn_checkpoint": args.pcn_ckpt if completion_enabled else None,
            "tracks": tracks_meta,
        }
        json_path = os.path.join(PROJECT_ROOT, f"output/{args.seq}/tracks.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Track-level filtering: {n_total} total, "
              f"{len(tracks_meta)} accepted, "
              f"{n_short} too short, {n_rejected_class} rejected by class")
        if completion_enabled:
            print(f"PCN completion: {n_completed}/{len(tracks_meta)} tracks completed")
        print(f"Saved {len(tracks_meta)} tracks to {output_dir}")

        # --- Mine sparse/dense completion pairs ---
        if args.mine_pairs:
            pairs_dir = args.mine_pairs
            n_pairs = 0
            accepted_ids = {t["track_id"] for t in tracks_meta}
            for track_id, pts_list in track_points.items():
                if track_id not in accepted_ids:
                    continue
                entry = next(t for t in tracks_meta if t["track_id"] == track_id)
                cls = entry["class"]
                cls_dir = os.path.join(pairs_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)

                dense = np.vstack(pts_list).astype(np.float32)
                tag = f"s{args.seq}_{track_id:04d}"
                np.save(os.path.join(cls_dir, f"dense_{tag}.npy"), dense)

                frames = track_frames[track_id]
                for i, (frame_pts, fidx) in enumerate(zip(pts_list, frames)):
                    sparse = frame_pts.astype(np.float32)
                    np.save(
                        os.path.join(cls_dir, f"sparse_{tag}_f{fidx:04d}.npy"),
                        sparse,
                    )
                    n_pairs += 1

            print(f"Mined {n_pairs} sparse/dense pairs to {pairs_dir}")

    if vis is not None:
        vis.run()


if __name__ == "__main__":
    main()
