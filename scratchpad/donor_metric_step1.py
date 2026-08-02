"""Direction 1 — Step 1: pair sweep + cache for the donor-frame completion metric.

For every frame in which a well-observed static GT car (from Step 0's
output/<seq>/amodal_gt.json) was observed, run the detection pipeline with the
production Stage B classifier, match detections to GT instances by point IoU
(same machinery as completion_box_eval.py), and for each TP pair on a
well-observed car cache everything the metric steps need:

  - the raw single-frame cluster (sensor frame),
  - the completed cloud (sensor frame, production complete() path),
  - the canonical-frame estimate (basis/center/radius) so the mirrored-partial
    baseline in Step 2 reuses *exactly* complete()'s geometry,
  - the world transform poses[fi] @ Tr.

Outputs one .npz per pair under output/experiments/donor_metric/pairs_<seq>/
plus an index JSON. Step 2 (donor accumulation + visibility-mask metric) reads
only these caches — the detection sweep never has to rerun.

Run:
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_step1.py --seq 08
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
from completion_box_eval import match_pairs  # noqa: E402
from classifier import load_classifier  # noqa: E402
from completion import PointCloudCompleter  # noqa: E402
from evaluate import THING_CLASSES_SUPPORTED, get_frame_detections  # noqa: E402
from pipeline import load_calib, load_poses  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="Process every k-th candidate frame")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="Cap on processed frames (smoke test)")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--cls-ckpt", default=os.path.join(
        PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth"))
    ap.add_argument("--pcn-ckpt", default=os.path.join(
        PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth"))
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default output/experiments/donor_metric)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric")
    pairs_dir = os.path.join(out_dir, f"pairs_{args.seq}")
    index_path = os.path.join(out_dir, f"step1_index_{args.seq}.json")
    if os.path.exists(index_path) and not args.overwrite:
        raise SystemExit(f"{index_path} exists; pass --overwrite to replace it.")

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

    os.makedirs(pairs_dir, exist_ok=True)
    index = []
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

        for det_id, gt_id, iou in pairs:
            raw_sensor = obj_points[cluster_labels == det_id]

            # Canonical frame first (rng-free), then completion (consumes rng in
            # the same per-pair order regardless of gate outcome -> deterministic).
            frame_est, frame_skip = completer.estimate_canonical_frame(raw_sensor)
            completed_sensor, skip = completer.complete(raw_sensor, "car")
            assert skip == frame_skip, (skip, frame_skip)
            if skip is not None:
                skip_counts[skip] = skip_counts.get(skip, 0) + 1

            fname = f"p_f{fi:06d}_i{int(gt_id)}.npz"
            arrays = {
                "raw": raw_sensor.astype(np.float32),
                "T": T.astype(np.float64),
            }
            if skip is None:
                arrays.update({
                    "completed": completed_sensor.astype(np.float32),
                    "basis": frame_est["basis"].astype(np.float64),
                    "center": frame_est["center"].astype(np.float64),
                    "radius": np.float64(frame_est["radius"]),
                })
            np.savez_compressed(os.path.join(pairs_dir, fname), **arrays)

            index.append({
                "file": fname,
                "frame": int(fi),
                "inst_id": int(gt_id),
                "iou": round(float(iou), 4),
                "n_raw_pts": int(len(raw_sensor)),
                "skip_reason": skip,
                "fit_length": (round(frame_est["fit_length"], 3)
                               if frame_est else None),
                "fit_width": (round(frame_est["fit_width"], 3)
                              if frame_est else None),
            })

        if (k + 1) % 50 == 0 or k == len(frames) - 1:
            el = time.time() - t0
            print(f"  frame {k + 1}/{len(frames)}  ({el:.0f}s, "
                  f"{el / (k + 1):.2f}s/frame, {len(index)} pairs)", flush=True)

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
            "frames_convention": "raw/completed in sensor frame; world via T "
                                 "(= poses[fi] @ Tr; X right, Y down, Z fwd)",
        },
        "pairs": index,
    }
    with open(index_path, "w") as f:
        json.dump(payload, f)

    n_comp = sum(1 for r in index if r["skip_reason"] is None)
    insts = sorted({r["inst_id"] for r in index})
    print(f"\n==== Donor-metric Step 1 summary ====")
    print(f"TP pairs on well-observed cars: {len(index)} "
          f"({len(insts)} distinct instances of {len(well)} well-observed)")
    print(f"Completed: {n_comp}; skipped: {skip_counts}")
    print(f"Saved caches -> {pairs_dir}\nIndex -> {index_path}")


if __name__ == "__main__":
    main()
