"""Zoomed close-ups of seq-08 failure frames.

For each frame: left panel = TP/FP/FN colouring (as in the BEV overview);
right panel = per-detection-cluster colours with GT car footprints outlined,
so cluster *merges* (one detection spanning two GT cars) and *splits* (one GT
car covered by multiple small fragments / none) are visually obvious.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

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
    pairs = []
    for did, dm in det_masks.items():
        for gid, gm in gt_masks.items():
            iou = compute_iou(dm, gm)
            if iou >= iou_thresh:
                pairs.append((iou, did, gid))
    pairs.sort(reverse=True)
    matched_det, matched_gt = set(), set()
    for iou, did, gid in pairs:
        if did in matched_det or gid in matched_gt:
            continue
        matched_det.add(did)
        matched_gt.add(gid)
    return matched_det, matched_gt


def outline(ax, pts2d, **kw):
    if len(pts2d) < 3:
        return
    try:
        h = ConvexHull(pts2d)
        ax.add_patch(Polygon(pts2d[h.vertices], closed=True, fill=False, **kw))
    except Exception:
        pass


def render_frame(ax_tp, ax_inst, bin_p, lbl_p, fi, model, device, stats,
                 iou_thresh=0.3):
    (_, det_ids, _, cluster_labels, objects_pcd,
     det_masks, gt_masks) = get_frame_detections(
        bin_p, lbl_p, cls_model=model, cls_device=device, cls_bbox_stats=stats,
        unknown_threshold=0.50, thing_classes=THING_CLASSES_SUPPORTED,
        keep_unknown=False)
    obj = np.asarray(objects_pcd.points)
    matched_det, matched_gt = match_assignments(det_masks, gt_masks, iou_thresh)

    # zoom window = bbox of all GT car points (+pad)
    gt_all = np.concatenate([obj[m] for m in gt_masks.values()]) if gt_masks else obj
    pad = 4.0
    xmin, xmax = gt_all[:, 0].min() - pad, gt_all[:, 0].max() + pad
    ymin, ymax = gt_all[:, 1].min() - pad, gt_all[:, 1].max() + pad

    for ax in (ax_tp, ax_inst):
        ax.scatter(obj[:, 0], obj[:, 1], s=1, c="0.85", linewidths=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x fwd (m)")
        ax.scatter([0], [0], marker="^", s=80, c="black", zorder=6)

    # --- TP/FP/FN panel ---
    for gid, gm in gt_masks.items():
        if gid not in matched_gt:
            ax_tp.scatter(obj[gm, 0], obj[gm, 1], s=6, c="#ff8c00", linewidths=0)
    for did, dm in det_masks.items():
        c = "#1faa1f" if did in matched_det else "#e02020"
        ax_tp.scatter(obj[dm, 0], obj[dm, 1], s=6, c=c, linewidths=0)
    tp = len(matched_det); fp = len(det_masks) - tp; fn = len(gt_masks) - tp
    ax_tp.set_title(f"frame {fi}  TP={tp} FP={fp} FN={fn}", fontsize=11)
    ax_tp.set_ylabel("y left (m)")

    # --- instance panel: detections coloured, GT outlined ---
    cmap = plt.get_cmap("tab20")
    for i, (did, dm) in enumerate(det_masks.items()):
        ax_inst.scatter(obj[dm, 0], obj[dm, 1], s=6,
                        c=[cmap(i % 20)], linewidths=0)
    for gid, gm in gt_masks.items():
        outline(ax_inst, obj[gm][:, :2], edgecolor="black", lw=1.3, ls="--")
    ax_inst.set_title("detections (colours) vs GT cars (dashed)", fontsize=11)


def main(frames=(3900, 2500, 250)):
    seq_dir = os.path.join(PROJECT_ROOT, "dataset/sequences/08")
    bins = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
    lbls = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_classifier(
        os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_best.pth"), device)

    n = len(frames)
    fig, axes = plt.subplots(n, 2, figsize=(13, 6 * n))
    axes = np.atleast_2d(axes)
    for r, fi in enumerate(frames):
        render_frame(axes[r, 0], axes[r, 1], bins[fi], lbls[fi], fi,
                     model, device, stats)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#1faa1f", label="TP"),
        plt.Line2D([], [], marker="o", ls="", color="#e02020", label="FP"),
        plt.Line2D([], [], marker="o", ls="", color="#ff8c00", label="FN (missed GT)"),
        plt.Line2D([], [], ls="--", color="black", label="GT car footprint"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Seq 08 failure close-ups: cluster merge / split vs GT",
                 y=1.015, fontsize=14)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", "seq08_failure_zooms.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
