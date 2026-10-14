"""Side-by-side visualization of raw vs PCN-completed track point clouds."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_comparison(raw_pts, comp_pts, title, output_path=None):
    """Render raw (blue) and completed (green) side by side using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={"projection": "3d"})
    fig.suptitle(title, fontsize=12)

    for ax, pts, label, color in [
        (axes[0], raw_pts, "Raw (accumulated)", "#4da6ff"),
        (axes[1], comp_pts, "PCN completed", "#00e64d"),
    ]:
        if len(pts) > 5000:
            idx = np.random.choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=color, alpha=0.6)
        ax.set_title(f"{label} ({len(raw_pts) if 'Raw' in label else len(comp_pts)} pts)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs completed tracks")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--track-id", type=int, default=None,
                        help="Specific track ID to show (default: largest raw track)")
    parser.add_argument("--all", action="store_true",
                        help="Render all completed tracks as saved images")
    parser.add_argument("--output-dir",
                        default=os.path.join(PROJECT_ROOT, "output", "completion_comparison"))
    args = parser.parse_args()

    raw_dir = os.path.join(PROJECT_ROOT, f"output/{args.seq}/objects_raw")
    completed_dir = os.path.join(PROJECT_ROOT, f"output/{args.seq}/objects")
    tracks_path = os.path.join(PROJECT_ROOT, f"output/{args.seq}/tracks.json")

    if not os.path.isdir(raw_dir):
        print(f"Raw objects dir not found: {raw_dir}")
        print("Run with --no-completion first, then copy to objects_raw.")
        sys.exit(1)

    with open(tracks_path) as f:
        meta = json.load(f)

    completed_tracks = [t for t in meta["tracks"] if t.get("completed")]
    if not completed_tracks:
        print("No completed tracks found.")
        sys.exit(1)

    if args.track_id is not None:
        tracks_to_show = [t for t in completed_tracks if t["track_id"] == args.track_id]
        if not tracks_to_show:
            print(f"Track {args.track_id} not found or not completed.")
            sys.exit(1)
    elif args.all:
        tracks_to_show = sorted(completed_tracks, key=lambda t: t["raw_point_count"], reverse=True)
    else:
        tracks_to_show = [max(completed_tracks, key=lambda t: t["raw_point_count"])]

    if args.all:
        os.makedirs(args.output_dir, exist_ok=True)

    for track in tracks_to_show:
        tid = track["track_id"]
        raw_path = os.path.join(raw_dir, f"{tid}.ply")
        comp_path = os.path.join(completed_dir, f"{tid}.ply")

        if not os.path.isfile(raw_path):
            print(f"  Track {tid}: raw PLY not found, skipping")
            continue
        if not os.path.isfile(comp_path):
            print(f"  Track {tid}: completed PLY not found, skipping")
            continue

        raw_pcd = o3d.io.read_point_cloud(raw_path)
        comp_pcd = o3d.io.read_point_cloud(comp_path)

        raw_pts = np.asarray(raw_pcd.points)
        comp_pts = np.asarray(comp_pcd.points)

        centroid = raw_pts.mean(axis=0)
        raw_pts = raw_pts - centroid
        comp_pts = comp_pts - centroid

        title = (f"Track {tid} ({track['class']}) | "
                 f"raw={track['raw_point_count']} -> completed={track['point_count']}")
        print(title)

        if args.all:
            out_path = os.path.join(args.output_dir, f"track_{tid}.png")
            render_comparison(raw_pts, comp_pts, title, out_path)
            print(f"  Saved: {out_path}")
        else:
            render_comparison(raw_pts, comp_pts, title)


if __name__ == "__main__":
    main()
