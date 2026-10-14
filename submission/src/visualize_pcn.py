"""Qualitative PCN evaluation — visualize input/coarse/fine on ShapeNet val samples.

Usage:
    python src/visualize_pcn.py
    python src/visualize_pcn.py --n-samples 5 --checkpoint checkpoints/pcn_best.pth
    python src/visualize_pcn.py --interactive  # Open3D window instead of saved images
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.dirname(__file__))
from completion import chamfer_distance, f_score
from pcn import PCN
from train_pcn import TRAIN_CONFIG, ShapeNetCompletionDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pcn(checkpoint_path: str, device: torch.device) -> PCN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", TRAIN_CONFIG)
    model = PCN(num_coarse=cfg.get("coarse_n_points", 1024), grid_size=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    print(f"Loaded PCN from {checkpoint_path} (epoch {epoch})")
    if metrics:
        cd = metrics.get('val_cd_fine')
        fs = metrics.get('val_fscore')
        cd_str = f"{cd:.5f}" if isinstance(cd, (int, float)) else "?"
        fs_str = f"{fs:.2%}" if isinstance(fs, (int, float)) else "?"
        print(f"  val_cd_fine={cd_str}, val_fscore={fs_str}")
    return model


def make_pcd(pts: np.ndarray, color: list[float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color)
    return pcd


def render_side_by_side(partial, coarse, gt, fine, title: str, output_path: str):
    """Render 4 point clouds in a 2x2 grid and save as image."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=800, visible=False)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    spacing = 2.5

    panels = [
        (partial, [-spacing, 0, 0], [0.3, 0.7, 1.0]),   # blue - input
        (coarse, [0, 0, 0], [1.0, 0.6, 0.0]),            # orange - coarse
        (fine, [spacing, 0, 0], [0.0, 1.0, 0.4]),        # green - fine
        (gt, [2 * spacing, 0, 0], [0.8, 0.8, 0.8]),      # white - GT
    ]

    for pts, offset, color in panels:
        pcd = make_pcd(pts, color)
        pcd.translate(offset)
        vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.35)
    ctr.set_front([0.0, -0.5, 0.5])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_lookat([spacing, 0, 0])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()


def show_interactive(partial, coarse, gt, fine, title: str):
    """Show 4 point clouds side by side in an interactive Open3D window."""
    spacing = 2.5
    panels = [
        (partial, [-spacing, 0, 0], [0.3, 0.7, 1.0]),
        (coarse, [0, 0, 0], [1.0, 0.6, 0.0]),
        (fine, [spacing, 0, 0], [0.0, 1.0, 0.4]),
        (gt, [2 * spacing, 0, 0], [0.8, 0.8, 0.8]),
    ]

    geometries = []
    for pts, offset, color in panels:
        pcd = make_pcd(pts, color)
        pcd.translate(offset)
        geometries.append(pcd)

    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1600, height=800)


def main():
    parser = argparse.ArgumentParser(description="Visualize PCN completions")
    parser.add_argument(
        "--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "pcn_best.pth"),
    )
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Samples per category to visualize")
    parser.add_argument("--interactive", action="store_true",
                        help="Open3D interactive window instead of saved images")
    parser.add_argument("--output-dir",
                        default=os.path.join(PROJECT_ROOT, "output", "pcn_qualitative"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pcn(args.checkpoint, device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    val_ds = ShapeNetCompletionDataset(TRAIN_CONFIG, split="val")
    print(f"Val set: {len(val_ds)} samples")

    # Group by category
    cat_indices: dict[str, list[int]] = {}
    for i, (_, synset_id) in enumerate(val_ds.items):
        cat_name = TRAIN_CONFIG["categories"][synset_id]
        cat_indices.setdefault(cat_name, []).append(i)

    if not args.interactive:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLayout: [blue=input] [orange=coarse] [green=fine] [white=GT]")
    print("=" * 70)

    all_cd_coarse, all_cd_fine, all_fs = [], [], []

    for cat_name, indices in sorted(cat_indices.items()):
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(indices, size=min(args.n_samples, len(indices)), replace=False)

        for rank, idx in enumerate(chosen):
            sample = val_ds[idx]
            partial_t = sample["partial"].unsqueeze(0).to(device)
            gt_np = sample["gt"].numpy()
            gt_coarse_np = sample["gt_coarse"].numpy()

            with torch.no_grad():
                coarse_t, fine_t = model(partial_t)

            partial_np = sample["partial"].numpy()
            coarse_np = coarse_t[0].cpu().numpy()
            fine_np = fine_t[0].cpu().numpy()

            cd_c = chamfer_distance(coarse_np, gt_coarse_np)
            cd_f = chamfer_distance(fine_np, gt_np)
            fs = f_score(fine_np, gt_np, threshold=0.1)

            all_cd_coarse.append(cd_c)
            all_cd_fine.append(cd_f)
            all_fs.append(fs)

            title = (f"{cat_name}_{rank} | "
                     f"CD_c={cd_c:.4f}  CD_f={cd_f:.4f}  F={fs:.2%} | "
                     f"pts: {len(partial_np)}->{len(coarse_np)}->{len(fine_np)}")
            print(title)

            if args.interactive:
                show_interactive(partial_np, coarse_np, gt_np, fine_np, title)
            else:
                out_path = os.path.join(args.output_dir, f"{cat_name}_{rank}.png")
                render_side_by_side(partial_np, coarse_np, gt_np, fine_np, title, out_path)
                print(f"  Saved: {out_path}")

    print("=" * 70)
    print(f"Mean CD_coarse: {np.mean(all_cd_coarse):.5f}")
    print(f"Mean CD_fine:   {np.mean(all_cd_fine):.5f}")
    print(f"Mean F-score:   {np.mean(all_fs):.2%}")
    print(f"Samples: {len(all_fs)} ({args.n_samples}/category)")


if __name__ == "__main__":
    main()
