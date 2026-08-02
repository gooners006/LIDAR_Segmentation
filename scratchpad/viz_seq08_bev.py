"""Headless BEV visualization of pipeline detections vs GT for seq 08.

Runs the real pipeline (ground removal -> cluster -> filter -> learned
classifier) on a spread of frames, matches detections to GT cars with the
same greedy-IoU logic as evaluate.py, and renders a top-down (BEV) panel per
frame colouring TP / FP / FN. Saves a multi-panel PNG. No Open3D GUI.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from classifier import load_classifier  # noqa: E402
from evaluate import (  # noqa: E402
    THING_CLASSES_SUPPORTED,
    compute_iou,
    get_frame_detections,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def match_assignments(det_masks, gt_masks, iou_thresh):
    """Greedy IoU matching; returns (matched_det, matched_gt, det2gt)."""
    pairs = []
    for det_id, dm in det_masks.items():
        for gt_id, gm in gt_masks.items():
            iou = compute_iou(dm, gm)
            if iou >= iou_thresh:
                pairs.append((iou, det_id, gt_id))
    pairs.sort(reverse=True)
    matched_det, matched_gt, det2gt = set(), set(), {}
    for iou, det_id, gt_id in pairs:
        if det_id in matched_det or gt_id in matched_gt:
            continue
        matched_det.add(det_id)
        matched_gt.add(gt_id)
        det2gt[det_id] = gt_id
    return matched_det, matched_gt, det2gt


def render(seq="08", frame_idxs=(200, 900, 1700, 2500, 3300, 3900),
           iou_thresh=0.3, out_name="seq08_bev_detections.png"):
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
    bins = sorted(__import__("glob").glob(os.path.join(seq_dir, "velodyne/*.bin")))
    labels = sorted(__import__("glob").glob(os.path.join(seq_dir, "labels/*.label")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_best.pth")
    model, bbox_stats = load_classifier(ckpt, device)

    n = len(frame_idxs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, fi in zip(axes, frame_idxs):
        (_, det_ids, _, cluster_labels, objects_pcd,
         det_masks, gt_masks) = get_frame_detections(
            bins[fi], labels[fi],
            cls_model=model, cls_device=device, cls_bbox_stats=bbox_stats,
            unknown_threshold=0.50, thing_classes=THING_CLASSES_SUPPORTED,
            keep_unknown=False,
        )
        obj = np.asarray(objects_pcd.points)
        matched_det, matched_gt, _ = match_assignments(det_masks, gt_masks, iou_thresh)

        # faint context: full object cloud
        ax.scatter(obj[:, 0], obj[:, 1], s=0.5, c="0.82", linewidths=0)

        # FN: unmatched GT cars (orange)
        for gid, gm in gt_masks.items():
            if gid not in matched_gt:
                ax.scatter(obj[gm, 0], obj[gm, 1], s=3, c="#ff8c00", linewidths=0)
        # FP: unmatched detections (red)
        for did, dm in det_masks.items():
            if did not in matched_det:
                ax.scatter(obj[dm, 0], obj[dm, 1], s=3, c="#e02020", linewidths=0)
        # TP: matched detections (green)
        for did, dm in det_masks.items():
            if did in matched_det:
                ax.scatter(obj[dm, 0], obj[dm, 1], s=3, c="#1faa1f", linewidths=0)

        tp = len(matched_det)
        fp = len(det_masks) - tp
        fn = len(gt_masks) - tp
        ax.scatter([0], [0], marker="^", s=120, c="black", zorder=5)  # ego
        ax.set_title(f"frame {fi}: TP={tp} FP={fp} FN={fn}", fontsize=11)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_xlabel("x fwd (m)")
        ax.set_ylabel("y left (m)")
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#1faa1f", label="TP (matched car)"),
        plt.Line2D([], [], marker="o", ls="", color="#e02020", label="FP (false detection)"),
        plt.Line2D([], [], marker="o", ls="", color="#ff8c00", label="FN (missed GT car)"),
        plt.Line2D([], [], marker="^", ls="", color="black", label="ego sensor"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Seq {seq} BEV: pipeline detections vs GT cars (IoU≥{iou_thresh})",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", out_name)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    render()
