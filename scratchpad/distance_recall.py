"""B4-lite (THESIS_PLAN.md §3, plan draft-ch-4): recall-by-distance table.

Stratifies detection quality by range-to-ego on seq 08 under the frozen
production config + classifier, so Chapter 4 §4.4 can show a real
distance-trend table instead of citing the stale Finding #5 (precision-only,
seq 00, pre-promotion).

Method (B2 pattern; imports evaluate.py read-only, edits nothing):
  * For each stride-20 frame, run get_frame_detections with the production
    classifier ON, keep_unknown=False (per-frame filtering), and
    thing_classes=THING_CLASSES_SUPPORTED ({10,252}) so the GT denominator
    matches the headline eval protocol exactly.
  * Recover WHICH GT are matched (TP) vs unmatched (FN) and which detections
    are unmatched (FP) by replicating match_detections_to_gt's greedy IoU>=0.3
    assignment verbatim (evaluate.py:88-108) -- the library function returns
    counts only.
  * Range of each detection / GT instance = ||mean(objects_pcd.points[mask])||
    (sensor frame, ego at origin -- the same distance_to_ego quantity as #5).
  * Bin TP+FN by GT range and FP by detection range into 0-20 / 20-40 / 40+ m
    (matching #5's bins for comparability). Aggregate TP/FP/FN -> P, R, F1 and
    the mean IoU of matched pairs, per bin.

DEVIATION (logged, same as Finding #5 / T10): this is PER-FRAME,
TRACK-FILTER-OFF (stride-20 breaks the tracker). The pooled overall recall
will therefore NOT equal the track-filtered headline R=0.730 -- read the table
as a distance TREND, cross-referenced to the headline, not a restatement of it.
Interpret alongside Finding #23 (close-range recall loss is HDBSCAN
over-segmentation, not far-range sparsity).

Usage:
    .venv\\Scripts\\python.exe scratchpad/distance_recall.py --seq 08 --stride 20
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from classifier import load_classifier  # noqa: E402
from evaluate import (  # noqa: E402
    PROJECT_ROOT,
    THING_CLASSES_SUPPORTED,
    compute_iou,
    get_frame_detections,
)

# Range bins in metres. Upper edge exclusive. Finer near-range split (0-10 /
# 10-20) probes Finding #23's 66%-split-at-0-10m over-segmentation regime; the
# #5-comparable 0-20 bin is recoverable as (0-10) union (10-20).
BIN_EDGES = [(0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, float("inf"))]
BIN_LABELS = ["0-10", "10-20", "20-40", "40+"]


def bin_index(dist: float) -> int:
    for k, (lo, hi) in enumerate(BIN_EDGES):
        if lo <= dist < hi:
            return k
    return len(BIN_EDGES) - 1  # inf upper edge already covers >=40


def greedy_match(det_masks: dict, gt_masks: dict, iou_thresh: float):
    """Replicate evaluate.match_detections_to_gt but keep the assignment.

    Returns (matches, matched_det, matched_gt) where matches is a list of
    (det_id, gt_id, iou). Assignment order is identical to evaluate.py so
    TP/FP/FN counts reproduce the library exactly.
    """
    pairs = []
    for det_id, det_m in det_masks.items():
        for gt_id, gt_m in gt_masks.items():
            iou = compute_iou(det_m, gt_m)
            if iou >= iou_thresh:
                pairs.append((iou, det_id, gt_id))
    pairs.sort(reverse=True)

    matched_det: set = set()
    matched_gt: set = set()
    matches: list = []
    for iou, det_id, gt_id in pairs:
        if det_id in matched_det or gt_id in matched_gt:
            continue
        matched_det.add(det_id)
        matched_gt.add(gt_id)
        matches.append((det_id, gt_id, iou))
    return matches, matched_det, matched_gt


def range_of(obj_pts: np.ndarray, mask: np.ndarray) -> float:
    """Euclidean range of a masked cluster's centroid from ego (sensor frame)."""
    return float(np.linalg.norm(obj_pts[mask].mean(axis=0)))


def main():
    parser = argparse.ArgumentParser(description="B4-lite: recall by range-to-ego")
    parser.add_argument("--seq", default="08", help="Sequence ID")
    parser.add_argument("--stride", type=int, default=20,
                        help="Sample every Nth frame (default 20)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on source frames before striding")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="Greedy match IoU threshold (frozen protocol: 0.3)")
    parser.add_argument(
        "--classifier-ckpt", type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth"),
        help="Classifier checkpoint (production default)")
    parser.add_argument("--unknown-threshold", type=float, default=0.50,
                        help="Classifier unknown-rejection threshold (frozen 0.50)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(PROJECT_ROOT, "output", "experiments",
                                             "distance_recall", "distance_recall_08.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{args.seq}")
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))
    if len(bin_paths) != len(label_paths):
        raise RuntimeError(
            f"Mismatched velodyne/label files: {len(bin_paths)} bins, "
            f"{len(label_paths)} labels")
    if args.max_frames is not None:
        bin_paths = bin_paths[: args.max_frames]
        label_paths = label_paths[: args.max_frames]

    idx = list(range(0, len(bin_paths), args.stride))

    # Production classifier (frozen checkpoint), same as evaluate.py main().
    cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, cls_bbox_stats = load_classifier(args.classifier_ckpt, cls_device)
    if cls_model is None:
        raise RuntimeError(f"Classifier not found: {args.classifier_ckpt}")

    print(f"Seq {args.seq}: {len(bin_paths)} frames total, sampling {len(idx)} "
          f"at stride {args.stride}")
    print(f"Classifier: {args.classifier_ckpt} on {cls_device}")
    print(f"Target GT classes (supported-vehicles): {sorted(THING_CLASSES_SUPPORTED)}")
    print(f"IoU threshold: {args.iou_threshold} | PER-FRAME, TRACK-FILTER-OFF")
    print("-" * 80)

    # Per-bin accumulators.
    n_bins = len(BIN_EDGES)
    tp = [0] * n_bins
    fp = [0] * n_bins
    fn = [0] * n_bins
    iou_sum = [0.0] * n_bins  # sum of matched IoUs (for mean-IoU-of-matched)

    for n, i in enumerate(idx):
        bp, lp = bin_paths[i], label_paths[i]
        (_, _, _, cluster_labels, objects_pcd,
         det_masks, gt_masks) = get_frame_detections(
            bp, lp,
            cls_model=cls_model, cls_device=cls_device,
            cls_bbox_stats=cls_bbox_stats,
            unknown_threshold=args.unknown_threshold,
            thing_classes=THING_CLASSES_SUPPORTED,
            keep_unknown=False,
        )

        obj_pts = np.asarray(objects_pcd.points)
        matches, matched_det, matched_gt = greedy_match(
            det_masks, gt_masks, args.iou_threshold)

        # TP + matched IoU, binned by GT range.
        for det_id, gt_id, iou in matches:
            b = bin_index(range_of(obj_pts, gt_masks[gt_id]))
            tp[b] += 1
            iou_sum[b] += iou
        # FN: unmatched GT, binned by GT range.
        for gt_id, gt_m in gt_masks.items():
            if gt_id not in matched_gt:
                fn[bin_index(range_of(obj_pts, gt_m))] += 1
        # FP: unmatched detections, binned by detection range.
        for det_id, det_m in det_masks.items():
            if det_id not in matched_det:
                fp[bin_index(range_of(obj_pts, det_m))] += 1

        if (n + 1) % 20 == 0 or n == len(idx) - 1:
            print(f"[{n + 1}/{len(idx)}] frame {i}: "
                  f"TP={sum(tp)} FP={sum(fp)} FN={sum(fn)}")

    # Per-bin + pooled metrics.
    def metrics(t, f, m, isum):
        prec = t / (t + f) if (t + f) else 0.0
        rec = t / (t + m) if (t + m) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        miou = isum / t if t else 0.0
        return prec, rec, f1, miou

    per_bin = []
    for k in range(n_bins):
        p, r, f1, miou = metrics(tp[k], fp[k], fn[k], iou_sum[k])
        per_bin.append({
            "range_m": BIN_LABELS[k],
            "tp": tp[k], "fp": fp[k], "fn": fn[k],
            "precision": p, "recall": r, "f1": f1, "mean_iou_matched": miou,
        })

    tot_tp, tot_fp, tot_fn = sum(tp), sum(fp), sum(fn)
    p, r, f1, miou = metrics(tot_tp, tot_fp, tot_fn, sum(iou_sum))
    pooled = {
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": p, "recall": r, "f1": f1, "mean_iou_matched": miou,
    }

    summary = {
        "seq": args.seq,
        "stride": args.stride,
        "frames_total": len(bin_paths),
        "frames_sampled": len(idx),
        "iou_threshold": args.iou_threshold,
        "classifier_ckpt": os.path.basename(args.classifier_ckpt),
        "unknown_threshold": args.unknown_threshold,
        "target_classes": sorted(THING_CLASSES_SUPPORTED),
        "evaluation_mode": "per-frame, track-filter-OFF (stride breaks tracker)",
        "bins": per_bin,
        "pooled": pooled,
    }

    print("-" * 80)
    print(f"{'range':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6}")
    for b in per_bin:
        print(f"{b['range_m']:>8} {b['tp']:>6} {b['fp']:>6} {b['fn']:>6} "
              f"{b['precision']:>6.3f} {b['recall']:>6.3f} {b['f1']:>6.3f} "
              f"{b['mean_iou_matched']:>6.3f}")
    print(f"{'POOLED':>8} {tot_tp:>6} {tot_fp:>6} {tot_fn:>6} "
          f"{pooled['precision']:>6.3f} {pooled['recall']:>6.3f} "
          f"{pooled['f1']:>6.3f} {pooled['mean_iou_matched']:>6.3f}")
    print("(POOLED is per-frame/track-filter-off; NOT the 0.730 headline recall.)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
