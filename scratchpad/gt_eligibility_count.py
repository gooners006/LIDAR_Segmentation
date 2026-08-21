"""B2 (THESIS_PLAN.md §3): quantify GT cars excluded by the eligibility rule.

The reported detection recall (e.g. seq-08 R=0.730) is TP / (TP+FN) where the
denominator counts only GT car instances with **>=10 points surviving
preprocessing** (z-filter, voxel, denoise, ground removal) -- the same
``gt_masks`` rule ``evaluate.py`` uses. This script measures how many *annotated*
cars never enter that denominator at all, so the thesis can state recall against
all annotated cars (examiner Q9), not just survivors.

Two quantities, deliberately kept distinct (project_state warns they are not
interchangeable):

  * raw_all   -- raw label scan: sem in {10, 252}, inst > 0, ANY point count.
                 "All annotated car instances" in the frame.
  * raw_ge10  -- raw label scan restricted to instances with >=10 RAW points.
                 This is the distinct-car *count* rule used in the advisor
                 reports (a threshold on unfiltered label points).
  * eligible  -- >=10 points SURVIVING preprocessing == len(gt_masks) from
                 get_frame_detections. This is the eval recall denominator.

The eligible count is taken verbatim from get_frame_detections (no classifier
needed -- gt_masks are independent of the classifier) so it matches evaluate.py
exactly. Exclusion percentages are reported both micro-averaged (pooled totals,
consistent with the micro-averaged eval protocol) and as the per-frame mean.

Usage:
    .venv\\Scripts\\python.exe scratchpad/gt_eligibility_count.py --seq 08 --stride 20
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evaluate import (  # noqa: E402
    PROJECT_ROOT,
    THING_CLASSES_SUPPORTED,
    get_frame_detections,
)

CAR_CLASSES = list(THING_CLASSES_SUPPORTED)  # {10 car, 252 moving-car}


def raw_car_instance_counts(label_path: str):
    """Count annotated car instances in one raw .label file.

    Returns (raw_all, raw_ge10): instances with sem in {10,252} & inst>0, at
    ANY point count and with >=10 raw points respectively.
    """
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem = raw_labels & 0xFFFF
    inst = raw_labels >> 16
    car_mask = np.isin(sem, CAR_CLASSES) & (inst > 0)
    if not car_mask.any():
        return 0, 0
    insts, counts = np.unique(inst[car_mask], return_counts=True)
    raw_all = len(insts)
    raw_ge10 = int((counts >= 10).sum())
    return raw_all, raw_ge10


def main():
    parser = argparse.ArgumentParser(description="B2: GT car eligibility exclusion count")
    parser.add_argument("--seq", default="08", help="Sequence ID")
    parser.add_argument("--stride", type=int, default=20,
                        help="Sample every Nth frame (default 20)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on number of source frames before striding")
    parser.add_argument("--out", type=str,
                        default=os.path.join(PROJECT_ROOT, "output", "experiments",
                                             "gt_eligibility", "gt_eligibility_08.json"),
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

    # Sample every `stride`th frame across the whole drive.
    idx = list(range(0, len(bin_paths), args.stride))
    print(f"Seq {args.seq}: {len(bin_paths)} frames total, sampling {len(idx)} "
          f"at stride {args.stride}")
    print(f"Car classes (raw + eligible): {sorted(CAR_CLASSES)}")
    print("-" * 80)

    per_frame = []
    tot_raw_all = tot_raw_ge10 = tot_eligible = 0
    excl_all_fracs = []   # per-frame (raw_all - eligible) / raw_all
    excl_ge10_fracs = []  # per-frame (raw_ge10 - eligible) / raw_ge10

    for n, i in enumerate(idx):
        bp, lp = bin_paths[i], label_paths[i]
        raw_all, raw_ge10 = raw_car_instance_counts(lp)
        # Eligible = len(gt_masks) from the exact eval preprocessing path.
        # No classifier: gt_masks are independent of it.
        *_, gt_masks = get_frame_detections(
            bp, lp,
            cls_model=None,
            thing_classes=THING_CLASSES_SUPPORTED,
            keep_unknown=False,
        )
        eligible = len(gt_masks)

        tot_raw_all += raw_all
        tot_raw_ge10 += raw_ge10
        tot_eligible += eligible
        if raw_all > 0:
            excl_all_fracs.append((raw_all - eligible) / raw_all)
        if raw_ge10 > 0:
            excl_ge10_fracs.append((raw_ge10 - eligible) / raw_ge10)

        per_frame.append({
            "frame_index": i,
            "raw_all": raw_all,
            "raw_ge10": raw_ge10,
            "eligible": eligible,
        })
        if (n + 1) % 20 == 0 or n == len(idx) - 1:
            print(f"[{n + 1}/{len(idx)}] frame {i}: raw_all={raw_all} "
                  f"raw_ge10={raw_ge10} eligible={eligible}")

    # Micro (pooled) exclusion percentages.
    micro_excl_all = (tot_raw_all - tot_eligible) / tot_raw_all if tot_raw_all else 0.0
    micro_excl_ge10 = (tot_raw_ge10 - tot_eligible) / tot_raw_ge10 if tot_raw_ge10 else 0.0

    summary = {
        "seq": args.seq,
        "stride": args.stride,
        "frames_total": len(bin_paths),
        "frames_sampled": len(idx),
        "car_classes": sorted(CAR_CLASSES),
        "totals": {
            "raw_all": tot_raw_all,
            "raw_ge10": tot_raw_ge10,
            "eligible": tot_eligible,
        },
        "micro_exclusion": {
            "vs_raw_all": micro_excl_all,
            "vs_raw_ge10": micro_excl_ge10,
        },
        "per_frame_mean_exclusion": {
            "vs_raw_all": float(np.mean(excl_all_fracs)) if excl_all_fracs else 0.0,
            "vs_raw_ge10": float(np.mean(excl_ge10_fracs)) if excl_ge10_fracs else 0.0,
        },
    }

    print("-" * 80)
    print(f"TOTALS (pooled over {len(idx)} frames):")
    print(f"  raw_all  (any pts)      = {tot_raw_all}")
    print(f"  raw_ge10 (>=10 raw pts) = {tot_raw_ge10}")
    print(f"  eligible (>=10 surviving pts, == eval denominator) = {tot_eligible}")
    print(f"MICRO exclusion vs raw_all  : {micro_excl_all * 100:.1f}%")
    print(f"MICRO exclusion vs raw_ge10 : {micro_excl_ge10 * 100:.1f}%")
    print(f"PER-FRAME MEAN exclusion vs raw_all  : "
          f"{summary['per_frame_mean_exclusion']['vs_raw_all'] * 100:.1f}%")
    print(f"PER-FRAME MEAN exclusion vs raw_ge10 : "
          f"{summary['per_frame_mean_exclusion']['vs_raw_ge10'] * 100:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
