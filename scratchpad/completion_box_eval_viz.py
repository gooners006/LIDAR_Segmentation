"""Direction 4a — Step 3: BEV box-overlay figures (GT black / raw blue / completed green).

Selects 6 representative (frame, car) pairs from the Step 1 records via the
Step 2 metrics — biggest completion wins at three input densities, biggest
width recovery, the worst regression, and the strongest length undershoot —
re-runs the pipeline + completion on just those frames, and renders one 2x3
panel figure of the world-frame BEV (X-Z plane, per Finding #27 the ground
plane of the poses @ Tr world frame).

Boxes and metrics shown are recomputed fresh (completion subsampling RNG state
differs from the Step 1 sweep, so values can differ in the last digits from
step2_metrics_08.json; the figure is self-consistent).

Run:
    .venv\\Scripts\\python.exe scratchpad\\completion_box_eval_viz.py --seq 08
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from completion_box_eval import match_pairs, world_box  # noqa: E402
from completion_box_eval_step2 import bev_iou, box_corners_xz, pair_metrics  # noqa: E402
from classifier import load_classifier  # noqa: E402
from completion import PointCloudCompleter  # noqa: E402
from evaluate import THING_CLASSES_SUPPORTED, get_frame_detections  # noqa: E402
from pipeline import load_calib, load_poses  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_panels(records: list, gt_by_inst: dict) -> list:
    """Pick 6 representative completed pairs, one car each where possible."""
    scored = []
    for r in records:
        if r["skip_reason"] is not None:
            continue
        gt = gt_by_inst[r["inst_id"]]
        raw_m = pair_metrics(r["raw_box"], gt)
        comp_m = pair_metrics(r["comp_box"], gt)
        scored.append((r, raw_m, comp_m, comp_m["bev_iou"] - raw_m["bev_iou"]))

    picks: list = []
    used_insts: set = set()

    def take(candidates, title):
        for r, raw_m, comp_m, gain in candidates:
            if r["inst_id"] not in used_insts:
                used_insts.add(r["inst_id"])
                picks.append((r, raw_m, comp_m, title))
                return

    by_gain = sorted(scored, key=lambda s: -s[3])
    take([s for s in by_gain if s[0]["n_raw_pts"] < 100],
         "best gain, sparse (<100 pts)")
    take([s for s in by_gain if 100 <= s[0]["n_raw_pts"] < 300],
         "best gain, mid (100-300 pts)")
    take(sorted(scored, key=lambda s: -(s[1]["adW"] - s[2]["adW"])),
         "largest width recovery")
    dense = [s for s in scored if s[0]["n_raw_pts"] >= 300]
    med_gain = float(np.median([s[3] for s in dense]))
    take(sorted(dense, key=lambda s: abs(s[3] - med_gain)),
         "typical dense (median gain)")
    take(sorted(scored, key=lambda s: s[3]), "worst regression")
    take(sorted((s for s in scored if s[0]["inst_id"] in gt_by_inst
                 and gt_by_inst[s[0]["inst_id"]]["dims_lwh"][0] >= 3.6),
                key=lambda s: s[2]["dL"]),
         "strongest length undershoot")
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--out", default=None,
                    help="Output PNG (default output/figures/"
                         "completion_box_overlays_<seq>.png)")
    args = ap.parse_args()
    out_png = args.out or os.path.join(
        PROJECT_ROOT, "output", "figures",
        f"completion_box_overlays_{args.seq}.png")

    rec_path = os.path.join(PROJECT_ROOT, "output", "experiments",
                            "completion_box_eval",
                            f"step1_records_{args.seq}.json")
    with open(rec_path) as f:
        records = json.load(f)["records"]
    with open(os.path.join(PROJECT_ROOT, "output", args.seq,
                           "amodal_gt.json")) as f:
        amodal = json.load(f)
    gt_by_inst = {int(k): r for k, r in amodal["instances"].items()
                  if r.get("well_observed")}

    picks = select_panels(records, gt_by_inst)
    print("Selected panels:")
    for r, _, _, title in picks:
        print(f"  {title}: inst {r['inst_id']} frame {r['frame']} "
              f"({r['n_raw_pts']} pts)")

    # Models + sequence data for the fresh re-run of the selected frames.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, bbox_stats = load_classifier(
        os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_best.pth"), device)
    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "sequences", args.seq)
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne", "*.bin")))
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels", "*.label")))
    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    Tr = load_calib(os.path.join(seq_dir, "calib.txt"))["Tr"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (rec, _, _, title) in zip(axes.ravel(), picks):
        fi, inst_id = rec["frame"], rec["inst_id"]
        completer = PointCloudCompleter(model_path=os.path.join(
            PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth"))

        _, _, _, cluster_labels, objects_pcd, det_masks, gt_masks = \
            get_frame_detections(
                bin_paths[fi], label_paths[fi],
                cls_model=cls_model, cls_device=device,
                cls_bbox_stats=bbox_stats,
                thing_classes=THING_CLASSES_SUPPORTED, keep_unknown=False)
        det_id = next((d for d, g, _ in
                       match_pairs(det_masks, gt_masks, 0.3) if int(g) == inst_id),
                      None)
        if det_id is None:
            ax.set_title(f"{title}\ninst {inst_id} frame {fi}: no match on re-run")
            continue

        raw_sensor = np.asarray(objects_pcd.points)[cluster_labels == det_id]
        comp_sensor, skip = completer.complete(raw_sensor, "car")
        T = poses[fi] @ Tr
        R, t = T[:3, :3], T[:3, 3]
        raw_w = raw_sensor @ R.T + t
        comp_w = comp_sensor @ R.T + t

        gt = gt_by_inst[inst_id]
        raw_box = world_box(raw_w)
        boxes = [(gt, "black", "GT (amodal)"), (raw_box, "tab:blue", "raw")]
        ax.scatter(comp_w[:, 0], comp_w[:, 2], s=1, c="palegreen", zorder=1)
        ax.scatter(raw_w[:, 0], raw_w[:, 2], s=3, c="dimgray", zorder=2)
        if skip is None:
            comp_box = world_box(comp_w)
            boxes.append((comp_box, "tab:green", "completed"))
        for box, color, label in boxes:
            c = box_corners_xz([box["center_world"][0], box["center_world"][2]],
                               box["yaw_deg"], *box["dims_lwh"][:2])
            c = np.vstack([c, c[:1]])
            ax.plot(c[:, 0], c[:, 1], color=color, lw=2, label=label, zorder=3)

        iou_r = bev_iou(raw_box, gt)
        sub = f"IoU raw {iou_r:.2f}"
        if skip is None:
            sub += f" -> comp {bev_iou(comp_box, gt):.2f}"
        else:
            sub += f" (completion skipped: {skip})"
        ax.set_title(f"{title}\ninst {inst_id}, frame {fi}, "
                     f"{len(raw_sensor)} pts | {sub}", fontsize=10)
        ax.set_aspect("equal")
        ax.margins(0.4)
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Z (m)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Seq {args.seq}: amodal GT vs raw-partial vs completed boxes (BEV, "
        "world X-Z)\ngray = raw cluster, light green = completed points",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
