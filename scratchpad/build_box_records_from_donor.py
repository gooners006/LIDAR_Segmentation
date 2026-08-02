"""Build completion_box_eval Step-1 records from cached donor-metric pairs.

The #29 box eval (`completion_box_eval.py`) re-runs live detection and stores
only fitted boxes. But the donor-metric pairs already cache the raw cluster +
recomputed completed cloud + world transform for the SAME production detection
(promoted config, stage_b_scratch). So we can fit the exact #29 oriented boxes
(`world_box`, fit_oriented_box_xz minmax) from those clouds and emit a
step1_records JSON that `completion_box_eval_step2.py` consumes unchanged — no
detection sweep, and the length-prior A/B is perfectly paired (same clusters).

Run:
    .venv\\Scripts\\python.exe scratchpad\\build_box_records_from_donor.py \
        --dir output/experiments/donor_perf_lenon \
        --out output/experiments/completion_box_eval/step1_records_08_lenon.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from completion_box_eval import world_box  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--dir", required=True, help="Donor recompute dir (index + pairs)")
    ap.add_argument("--out", required=True, help="Output step1_records JSON")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = args.dir if os.path.isabs(args.dir) else os.path.join(PROJECT_ROOT, args.dir)
    out = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it.")

    with open(os.path.join(src, f"step1_index_{args.seq}.json")) as f:
        index = json.load(f)
    pairs_dir = os.path.join(src, f"pairs_{args.seq}")

    records = []
    t0 = time.time()
    for k, p in enumerate(index["pairs"]):
        data = np.load(os.path.join(pairs_dir, p["file"]))
        T = data["T"]; R, t = T[:3, :3], T[:3, 3]
        raw_box = world_box(data["raw"].astype(np.float64) @ R.T + t)
        comp_box = (world_box(data["completed"].astype(np.float64) @ R.T + t)
                    if p["skip_reason"] is None else None)
        records.append({
            "frame": p["frame"], "inst_id": p["inst_id"], "iou": p.get("iou"),
            "n_raw_pts": p["n_raw_pts"], "skip_reason": p["skip_reason"],
            "raw_box": raw_box, "comp_box": comp_box,
        })
        if (k + 1) % 500 == 0 or k == len(index["pairs"]) - 1:
            print(f"  {k + 1}/{len(index['pairs'])} ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "seq": args.seq,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            **index.get("config", {}),
            "box_fit": "fit_oriented_box_xz extent=minmax, world frame via poses @ Tr",
            "built_from": os.path.basename(src),
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f)
    n_comp = sum(1 for r in records if r["skip_reason"] is None)
    print(f"\nWrote {len(records)} records ({n_comp} completed) -> {out}")


if __name__ == "__main__":
    main()
