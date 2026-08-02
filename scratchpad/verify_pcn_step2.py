"""PCN verification — Step 2: real seq-08 single-frame clusters.

Mines single-frame car clusters (GT labels, velodyne frame: Z-up, ego at origin),
tracks them, identifies STATIC cars, and accumulates their frames into a chosen
reference frame to form a multi-view pseudo-GT.

For several examples across the point-count range it compares three completion paths
quantitatively (Chamfer / F-score vs pseudo-GT) and renders multi-view PNGs:
  A. pcn_best        via completion.py complete()       (documented blob baseline)
  B. pcn_kitti_best  via completion.py complete()        (current, PCA+partial-r)
  C. pcn_kitti_best  via CORRECTED inference (no-PCA, reorient, scale+center fix)
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mine_completion_pairs import process_frame  # noqa: E402
from pipeline import PIPELINE_CONFIG, load_calib, load_poses  # noqa: E402
from tracker import CentroidTracker  # noqa: E402
from pcn import PCN  # noqa: E402
from completion import PointCloudCompleter, chamfer_distance, f_score  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCN_N_INPUT = 256
SCALE_CORRECTION = 1.137   # median partial_radius/gt_radius from step 1
CAR_WIDTH_PRIOR = 1.9      # m, for occluded-side center shift
UP_SHIFT = 0.25            # m, undo ground-cut upward bias (calibrated in step 1)


# --------------------------------------------------------------------------
# Corrected inference (no PCA; reorient gravity->Y, length->Z; scale+center fix)
# --------------------------------------------------------------------------

def _fix_size(pts, n, rng):
    pts = pts.astype(np.float32)
    if len(pts) >= n:
        return pts[rng.choice(len(pts), n, replace=False)]
    return np.vstack([pts, pts[rng.choice(len(pts), n - len(pts), replace=True)]])


def complete_corrected(model, device, partial_velo, rng, use_width_prior=True):
    """Complete a single-frame cluster given in velodyne frame (Z up, ego at origin)."""
    pts = np.asarray(partial_velo, dtype=np.float64)
    if len(pts) < 16:
        return None

    # Reorient: gravity (real +Z) -> train up (Y); horizontal PCA major -> train length (Z)
    xy = pts[:, :2] - pts[:, :2].mean(0)
    cov = xy.T @ xy
    w, v = np.linalg.eigh(cov)
    e_len = np.array([v[0, 1], v[1, 1], 0.0])   # major horizontal eigenvector
    e_len /= np.linalg.norm(e_len)
    e_wid = np.array([-e_len[1], e_len[0], 0.0])  # perpendicular in ground plane
    e_up = np.array([0.0, 0.0, 1.0])
    B = np.column_stack([e_wid, e_up, e_len])     # real->train basis (columns are real-frame axes)
    pts_t = pts @ B                               # train-frame coords (X=width, Y=up, Z=length)

    # Center estimate (calibrated on synthetic val, step 1):
    #   - bbox center as the base
    #   - up-axis (train Y): shift down ~0.25 m to undo the ground-cut upward bias (dominant fix)
    #   - width-axis (train X): push toward the occluded (far-from-ego) side by the car-width prior
    center = 0.5 * (pts_t.min(0) + pts_t.max(0))
    center[1] -= UP_SHIFT
    if use_width_prior:
        sign = np.sign(center[0]) if abs(center[0]) > 1e-9 else 1.0  # +X is away from ego
        observed_w = pts_t[:, 0].max() - pts_t[:, 0].min()
        shift = max(0.5 * CAR_WIDTH_PRIOR - 0.5 * observed_w, 0.0)
        center[0] += sign * shift

    radius = float(np.linalg.norm(pts_t - center, axis=1).max()) / SCALE_CORRECTION
    if radius < 1e-6:
        return None

    p_norm = ((pts_t - center) / radius).astype(np.float32)
    p_fixed = _fix_size(p_norm, PCN_N_INPUT, rng)
    with torch.no_grad():
        inp = torch.from_numpy(p_fixed).float().unsqueeze(0).to(device)
        _, fine = model(inp)
        pred_t = fine.squeeze(0).cpu().numpy()
    pred_t = pred_t * radius + center
    pred_velo = pred_t @ B.T                       # back to velodyne frame
    return pred_velo.astype(np.float32)


# --------------------------------------------------------------------------
# Mining + static-car detection
# --------------------------------------------------------------------------

def mine_static_cars(seq, n_frames, min_frames, static_thresh):
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:n_frames]
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[:n_frames]
    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    Tr = load_calib(os.path.join(seq_dir, "calib.txt"))["Tr"]

    tracker = CentroidTracker(
        max_distance=PIPELINE_CONFIG["tracker_max_distance"],
        max_disappeared=PIPELINE_CONFIG["tracker_max_disappeared"],
    )
    # track_id -> list of (frame_idx, velo_pts, T_total)
    tracks: dict[int, list] = {}

    n = min(len(bin_paths), len(label_paths))
    print(f"Scanning {n} frames of seq {seq} ...")
    for fi in range(n):
        res = process_frame(bin_paths[fi], label_paths[fi], purity_threshold=0.75)
        if res[0] is None:
            continue
        objects_pcd, clusters, gt_classes, cluster_labels = res
        T_total = poses[fi] @ Tr
        R, t = T_total[:3, :3], T_total[:3, 3]

        bboxes, keep = [], []
        for (bbox, cl), gt_cls in zip(clusters, gt_classes):
            if gt_cls != "car":
                continue
            bb = bbox  # velodyne-frame bbox; transform a copy for tracking
            bb_g = bbox
            bb_g.rotate(R, center=np.zeros(3))
            bb_g.translate(t)
            bboxes.append(bb_g)
            keep.append(cl)

        if not bboxes:
            tracker.update([])
            continue
        _, assignments = tracker.update(bboxes)

        obj_pts = np.asarray(objects_pcd.points)
        for bi, tid in assignments.items():
            cl = keep[bi]
            velo_pts = obj_pts[cluster_labels == cl].astype(np.float32)
            tracks.setdefault(tid, []).append((fi, velo_pts, T_total))

        if (fi + 1) % 200 == 0:
            print(f"  frame {fi+1}: {len(tracker.objects)} active tracks")

    # Identify static tracks via global bbox-center displacement.
    # (Bbox center of accumulated-so-far is more robust than the mass centroid,
    #  which shifts as the visible surface changes for a parked car.)
    static = []
    disps = []
    for tid, entries in tracks.items():
        if len(entries) < min_frames:
            continue
        gcents = []
        for fi, velo_pts, T in entries:
            g = (T[:3, :3] @ velo_pts.T).T + T[:3, 3]
            gcents.append(0.5 * (g.min(0) + g.max(0)))  # bbox center in global frame
        gcents = np.array(gcents)
        disp = float(np.linalg.norm(gcents.max(0) - gcents.min(0)))
        disps.append((disp, tid, len(entries)))
        if disp < static_thresh:
            static.append((tid, entries, disp))
    disps.sort()
    print("Displacement distribution (multi-frame tracks): "
          + ", ".join(f"{d:.1f}m(t{t},{n}f)" for d, t, n in disps[:12]))
    print(f"Found {len(static)} static car tracks (>= {min_frames} frames, disp < {static_thresh} m) "
          f"out of {len(tracks)} total tracks")
    return static


def build_pseudo_gt(entries, ref_idx):
    """Accumulate all frames into the reference frame's velodyne coords."""
    _, _, T_ref = entries[ref_idx]
    T_ref_inv = np.linalg.inv(T_ref)
    acc = []
    for fi, velo_pts, T in entries:
        g = (T[:3, :3] @ velo_pts.T).T + T[:3, 3]
        ref = (T_ref_inv[:3, :3] @ g.T).T + T_ref_inv[:3, 3]
        acc.append(ref)
    return np.vstack(acc).astype(np.float32)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

VIEWS = [("top", 89, -90), ("side", 5, 0), ("iso", 25, -60)]


def render_example(partial, preds_named, pseudo_gt, suptitle, out_path):
    cols = [("Partial (input)", partial, "#4da6ff")]
    for name, p in preds_named:
        cols.append((name, p, "#00e64d"))
    cols.append(("Pseudo-GT (accum.)", pseudo_gt, "#bbbbbb"))

    nrow, ncol = len(VIEWS), len(cols)
    fig = plt.figure(figsize=(4 * ncol, 4 * nrow))
    fig.suptitle(suptitle, fontsize=12)
    center = partial.mean(0)
    rng = max(np.ptp(pseudo_gt, axis=0).max(), 3.0) / 2
    for r, (vname, elev, azim) in enumerate(VIEWS):
        for c, (cname, pts, color) in enumerate(cols):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            show = pts
            if len(show) > 4000:
                show = show[np.random.choice(len(show), 4000, replace=False)]
            ax.scatter(show[:, 0], show[:, 1], show[:, 2], s=0.6, c=color, alpha=0.7)
            ax.view_init(elev=elev, azim=azim)
            for axis, mid in zip("xyz", center):
                getattr(ax, f"set_{axis}lim")(mid - rng, mid + rng)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if r == 0:
                ax.set_title(f"{cname}\n({len(pts)} pts)", fontsize=9)
            if c == 0:
                ax.text2D(-0.1, 0.5, vname, transform=ax.transAxes, fontsize=10, rotation=90, va="center")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--static-thresh", type=float, default=0.5)
    ap.add_argument("--n-examples", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    out_dir = os.path.join(PROJECT_ROOT, "output", "verify_pcn_step2")
    os.makedirs(out_dir, exist_ok=True)

    kitti_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")
    best_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_best.pth")

    ck = torch.load(kitti_ckpt, map_location=device, weights_only=False)
    model = PCN(num_coarse=1024, grid_size=2).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    comp_kitti = PointCloudCompleter(model_path=kitti_ckpt, seed=0)
    comp_best = PointCloudCompleter(model_path=best_ckpt, seed=0)

    static = mine_static_cars(args.seq, args.frames, args.min_frames, args.static_thresh)
    if not static:
        print("No static cars found; increase --frames.")
        return

    # Pick examples spread across partial point-count (use each track's densest frame).
    # Restrict to the realistic single-car single-frame range; larger clusters are
    # under-segmented merges of adjacent parked cars (not valid completion targets).
    cand = []
    for tid, entries, disp in static:
        ref_idx = int(np.argmax([len(e[1]) for e in entries]))
        npts = len(entries[ref_idx][1])
        if not (40 <= npts <= 300):
            continue
        cand.append((npts, tid, entries, ref_idx, disp))
    cand.sort()
    if len(cand) > args.n_examples:
        sel = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, args.n_examples)]
    else:
        sel = cand

    rows = []
    for npts, tid, entries, ref_idx, disp in sel:
        partial = entries[ref_idx][1]
        pgt = build_pseudo_gt(entries, ref_idx)

        pred_best, sk1 = comp_best.complete(partial, "car")
        pred_kitti, sk2 = comp_kitti.complete(partial, "car")
        pred_corr = complete_corrected(model, device, partial, rng)

        def cd(p):
            return chamfer_distance(p, pgt) if p is not None else float("nan")

        def fs(p):
            return f_score(p, pgt, threshold=0.10) if p is not None else float("nan")

        cd_partial = cd(partial)
        row = dict(tid=tid, npts=npts, disp=disp, gt=len(pgt),
                   cd_partial=cd_partial,
                   cd_best=cd(pred_best), cd_kitti=cd(pred_kitti), cd_corr=cd(pred_corr),
                   fs_corr=fs(pred_corr), fs_kitti=fs(pred_kitti))
        rows.append(row)

        sup = (f"seq{args.seq} track {tid} | {npts} input pts | disp {disp:.2f}m | "
               f"CD partial={cd_partial:.3f} best={row['cd_best']:.3f} "
               f"kitti={row['cd_kitti']:.3f} corr={row['cd_corr']:.3f}")
        out_path = os.path.join(out_dir, f"t{tid:04d}_{npts}pts.png")
        render_example(partial,
                       [("pcn_best complete()", pred_best),
                        ("pcn_kitti complete()", pred_kitti),
                        ("pcn_kitti CORRECTED", pred_corr)],
                       pgt, sup, out_path)
        print(f"  saved {out_path}")

    print("\n==== REAL seq-08 static cars: CD(m) vs pseudo-GT (lower better) ====")
    print(f"{'track':>6} {'pts':>5} {'gtpts':>6} {'disp':>5} | "
          f"{'partial':>8} {'best':>8} {'kitti':>8} {'corr':>8} | {'F_corr':>7} {'F_kitti':>7}")
    for r in rows:
        print(f"{r['tid']:>6} {r['npts']:>5} {r['gt']:>6} {r['disp']:>5.2f} | "
              f"{r['cd_partial']:>8.3f} {r['cd_best']:>8.3f} {r['cd_kitti']:>8.3f} {r['cd_corr']:>8.3f} | "
              f"{r['fs_corr']:>7.3f} {r['fs_kitti']:>7.3f}")
    arr = lambda k: np.array([r[k] for r in rows])
    print("-" * 80)
    print(f"{'MEAN':>6} {'':>5} {'':>6} {'':>5} | "
          f"{arr('cd_partial').mean():>8.3f} {arr('cd_best').mean():>8.3f} "
          f"{arr('cd_kitti').mean():>8.3f} {arr('cd_corr').mean():>8.3f} | "
          f"{arr('fs_corr').mean():>7.3f} {arr('fs_kitti').mean():>7.3f}")
    print(f"\nRenders: {out_dir}")


if __name__ == "__main__":
    main()
