"""Direction 4a — Step 1: per-frame raw vs completed boxes for well-observed cars.

For every frame in which a well-observed static GT car (from Step 0's
output/<seq>/amodal_gt.json) was observed, run the detection pipeline with the
Stage B classifier, match detections to GT instances by point IoU, and for each
TP pair whose GT instance is well-observed:

  - fit an oriented box to the raw single-frame cluster (sensor frame ->
    world via poses @ Tr, minmax extents),
  - run the production completion path (completion.complete(): L-shape gate,
    lshape heading, pcn_kitti checkpoint) and fit a box to the completed cloud,
  - record both boxes plus the gate skip reason if completion was skipped.

GT boxes come from amodal_gt.json and are joined in Step 2 (records store the
instance ID). Raw, completed and GT boxes all use fit_oriented_box_xz so
fitter bias cancels in the paired comparison.

Run:
    .venv\\Scripts\\python.exe scratchpad\\completion_box_eval.py --seq 08
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amodal_gt import fit_oriented_box_xz  # noqa: E402
from classifier import load_classifier  # noqa: E402
from completion import PointCloudCompleter  # noqa: E402
from evaluate import THING_CLASSES_SUPPORTED, compute_iou, get_frame_detections  # noqa: E402
from pipeline import load_calib, load_poses  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def match_pairs(det_masks: dict, gt_masks: dict, iou_thresh: float):
    """Greedy IoU matching (same rule as evaluate.match_detections_to_gt) but
    returning the matched (det_id, gt_id, iou) pairs instead of counts."""
    pairs = []
    for det_id, det_m in det_masks.items():
        for gt_id, gt_m in gt_masks.items():
            iou = compute_iou(det_m, gt_m)
            if iou >= iou_thresh:
                pairs.append((iou, det_id, gt_id))
    pairs.sort(reverse=True)

    matched_det, matched_gt, out = set(), set(), []
    for iou, det_id, gt_id in pairs:
        if det_id in matched_det or gt_id in matched_gt:
            continue
        matched_det.add(det_id)
        matched_gt.add(gt_id)
        out.append((det_id, gt_id, float(iou)))
    return out


def world_box(pts_world: np.ndarray) -> dict:
    """Oriented box (minmax extents) in the world frame: L-shape yaw in X-Z,
    height from the Y span (world Y is down; span is sign-agnostic)."""
    box = fit_oriented_box_xz(pts_world[:, [0, 2]], extent="minmax")
    y_lo, y_hi = float(pts_world[:, 1].min()), float(pts_world[:, 1].max())
    return {
        "yaw_deg": box["yaw_deg"],
        "dims_lwh": [round(box["length"], 3), round(box["width"], 3),
                     round(y_hi - y_lo, 3)],
        "center_world": [box["center_xz"][0],
                         round(0.5 * (y_lo + y_hi), 3),
                         box["center_xz"][1]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="Process every k-th candidate frame")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="Cap on processed frames (smoke test)")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--cls-ckpt", default=os.path.join(
        PROJECT_ROOT, "checkpoints", "stage_b_best.pth"))
    ap.add_argument("--pcn-ckpt", default=os.path.join(
        PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth"))
    ap.add_argument("--out", default=None,
                    help="Output JSON (default output/experiments/"
                         "completion_box_eval/step1_records_<seq>.json)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        PROJECT_ROOT, "output", "experiments", "completion_box_eval",
        f"step1_records_{args.seq}.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    # --- Step 0 output: well-observed instances and their observation frames ---
    amodal_path = os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")
    with open(amodal_path) as f:
        amodal = json.load(f)
    well = {int(k): r for k, r in amodal["instances"].items()
            if r.get("well_observed")}
    if not well:
        raise SystemExit(f"No well-observed instances in {amodal_path}")

    frame_set = sorted({fi for r in well.values() for fi, _ in r["frames"]})
    frames = frame_set[::args.frame_step]
    if args.limit_frames:
        frames = frames[:args.limit_frames]
    print(f"{len(well)} well-observed instances; {len(frame_set)} candidate "
          f"frames, processing {len(frames)} (step={args.frame_step})")

    # --- Models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, bbox_stats = load_classifier(args.cls_ckpt, device)
    if cls_model is None:
        raise SystemExit(f"Classifier checkpoint not found: {args.cls_ckpt}")
    completer = PointCloudCompleter(model_path=args.pcn_ckpt)

    # --- Sequence data ---
    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "sequences", args.seq)
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne", "*.bin")))
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels", "*.label")))
    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    Tr = load_calib(os.path.join(seq_dir, "calib.txt"))["Tr"]

    records = []
    skip_counts: dict[str, int] = {}
    t0 = time.time()
    for k, fi in enumerate(frames):
        _, det_ids, _, cluster_labels, objects_pcd, det_masks, gt_masks = \
            get_frame_detections(
                bin_paths[fi], label_paths[fi],
                cls_model=cls_model, cls_device=device,
                cls_bbox_stats=bbox_stats,
                thing_classes=THING_CLASSES_SUPPORTED,
                keep_unknown=False,
            )

        pairs = [(d, g, iou) for d, g, iou in
                 match_pairs(det_masks, gt_masks, args.iou_threshold)
                 if int(g) in well]
        if not pairs:
            continue

        obj_points = np.asarray(objects_pcd.points)
        T = poses[fi] @ Tr
        R, t = T[:3, :3], T[:3, 3]

        for det_id, gt_id, iou in pairs:
            raw_sensor = obj_points[cluster_labels == det_id]
            completed_sensor, skip = completer.complete(raw_sensor, "car")
            if skip is not None:
                skip_counts[skip] = skip_counts.get(skip, 0) + 1

            rec = {
                "frame": int(fi),
                "inst_id": int(gt_id),
                "iou": round(iou, 4),
                "n_raw_pts": int(len(raw_sensor)),
                "skip_reason": skip,
                "raw_box": world_box(raw_sensor @ R.T + t),
                "comp_box": (world_box(completed_sensor @ R.T + t)
                             if skip is None else None),
            }
            records.append(rec)

        if (k + 1) % 50 == 0 or k == len(frames) - 1:
            el = time.time() - t0
            print(f"  frame {k + 1}/{len(frames)}  ({el:.0f}s, "
                  f"{el / (k + 1):.2f}s/frame, {len(records)} pairs)", flush=True)

    # --- Save ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "seq": args.seq,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "iou_threshold": args.iou_threshold,
            "frame_step": args.frame_step,
            "limit_frames": args.limit_frames,
            "cls_ckpt": os.path.basename(args.cls_ckpt),
            "pcn_ckpt": os.path.basename(args.pcn_ckpt),
            "amodal_gt": amodal_path,
            "box_fit": "fit_oriented_box_xz extent=minmax, world frame via poses @ Tr",
        },
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)

    # --- Summary ---
    n_comp = sum(1 for r in records if r["skip_reason"] is None)
    insts = sorted({r["inst_id"] for r in records})
    print(f"\n==== Step 1 summary ====")
    print(f"TP pairs on well-observed cars: {len(records)} "
          f"({len(insts)} distinct instances of {len(well)} well-observed)")
    print(f"Completed: {n_comp}; skipped: {skip_counts}")

    if n_comp:
        # Quick paired preview vs GT dims (full metrics are Step 2's job).
        d_raw, d_comp = [], []
        for r in records:
            if r["skip_reason"] is not None:
                continue
            gt_dims = np.array(well[r["inst_id"]]["dims_lwh"])
            d_raw.append(np.abs(np.array(r["raw_box"]["dims_lwh"]) - gt_dims))
            d_comp.append(np.abs(np.array(r["comp_box"]["dims_lwh"]) - gt_dims))
        d_raw, d_comp = np.array(d_raw), np.array(d_comp)
        print("\nPreview mean |dim error| vs amodal GT (completed pairs only):")
        for i, name in enumerate("LWH"):
            print(f"  |d{name}|  raw {d_raw[:, i].mean():.3f} m   "
                  f"completed {d_comp[:, i].mean():.3f} m")
    print(f"\nSaved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
