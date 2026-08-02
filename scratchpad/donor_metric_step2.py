"""Direction 1 — Step 2: donor-frame occluded-side metric computation.

Reads Step 1's cached pairs (raw cluster, completed cloud, canonical frame,
world transform) and, per pair:

  1. accumulates the car's GT points from all *other* observation frames
     (donor cloud, world frame, 3 cm voxel — amodal_gt.py machinery),
  2. builds the visibility-mask novel set: donor points >= tau from every
     input point (the surfaces the input frame never saw),
  3. scores three method clouds against the novel set — raw partial,
     mirrored partial (symmetry union across complete()'s own estimated
     symmetry plane), and the PCN completion:
       - novel-coverage distance (median, m) and coverage fraction @ 0.1 m,
       - hallucination guard: fraction of method points outside the amodal
         GT box + 0.2 m margin,
  4. region breakdowns of the novel set at the primary tau (far side / far
     end relative to ego, top) and a symmetry self-consistency CD (secondary).

The raw partial scores zero coverage at thresholds < tau *by construction* —
that is the point: the metric isolates added surface. Records ->
output/experiments/donor_metric/donor_metric_records_<seq>.json (Step 3 does
the per-car statistics).

Run:
    .venv\\Scripts\\python.exe scratchpad\\donor_metric_step2.py --seq 08
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amodal_gt import accumulate_instances, voxel_downsample  # noqa: E402
from completion import chamfer_distance  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TAUS = [0.10, 0.15, 0.20]   # visibility-mask thresholds (m); middle = primary
TAU_PRIMARY = 0.15
COV_THRESH = 0.10           # coverage counts novel points within this (m)
BOX_MARGIN = 0.20           # hallucination guard margin around the GT box (m)
DONOR_VOXEL = 0.03          # m, donor accumulation voxel (matches amodal_gt)
METHODS = ["raw", "mirrored", "completed"]


def box_coords(pts_world: np.ndarray, rec: dict):
    """World points -> (length, width, y) coords in the amodal GT box frame.

    World is cam0: X right, Y down, Z forward; box yaw is the length-axis
    angle from +X toward +Z in the X-Z ground plane (amodal_gt convention).
    """
    cx, cy, cz = rec["center_world"]
    yaw = np.deg2rad(rec["yaw_deg"])
    dx, dy, dz = pts_world[:, 0] - cx, pts_world[:, 1] - cy, pts_world[:, 2] - cz
    c, s = np.cos(yaw), np.sin(yaw)
    len_c = dx * c + dz * s
    wid_c = -dx * s + dz * c
    return len_c, wid_c, dy


def out_of_box_frac(pts_world: np.ndarray, rec: dict, margin: float) -> float:
    L, W, H = rec["dims_lwh"]
    len_c, wid_c, dy = box_coords(pts_world, rec)
    inside = ((np.abs(len_c) <= L / 2 + margin)
              & (np.abs(wid_c) <= W / 2 + margin)
              & (np.abs(dy) <= H / 2 + margin))
    return float(1.0 - inside.mean())


def coverage(novel_world: np.ndarray, method_world: np.ndarray):
    """(median novel->method distance, coverage fraction @ COV_THRESH)."""
    d, _ = cKDTree(method_world).query(novel_world)
    return float(np.median(d)), float((d < COV_THRESH).mean()), d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--dir", default=None,
                    help="Experiment dir (default output/experiments/donor_metric)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    exp_dir = args.dir or os.path.join(
        PROJECT_ROOT, "output", "experiments", "donor_metric")
    index_path = os.path.join(exp_dir, f"step1_index_{args.seq}.json")
    pairs_dir = os.path.join(exp_dir, f"pairs_{args.seq}")
    out_path = os.path.join(exp_dir, f"donor_metric_records_{args.seq}.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    with open(index_path) as f:
        index = json.load(f)
    amodal_path = os.path.join(PROJECT_ROOT, "output", args.seq, "amodal_gt.json")
    with open(amodal_path) as f:
        amodal = json.load(f)
    well = {int(k): r for k, r in amodal["instances"].items()
            if r.get("well_observed")}

    pairs = [p for p in index["pairs"] if p["skip_reason"] is None]
    insts_needed = sorted({p["inst_id"] for p in pairs})
    print(f"{len(pairs)} gate-passed pairs on {len(insts_needed)} instances "
          f"(of {len(index['pairs'])} total pairs)")

    # --- Donor point source: per-frame world-frame GT clusters, cached ---
    cache_path = os.path.join(exp_dir, f"accum_cache_{args.seq}.npz")
    if os.path.exists(cache_path):
        print(f"Loading donor accumulation cache: {cache_path}")
        cache = np.load(cache_path)
        per_frame = {}
        for key in cache.files:
            inst_s, frame_s = key.split("_")
            per_frame.setdefault(int(inst_s), {})[int(frame_s)] = cache[key]
    else:
        seq_dir = os.path.join(PROJECT_ROOT, "dataset", "sequences", args.seq)
        print("Accumulating GT instance points (one full-sequence scan) ...")
        buffers, obs, _ = accumulate_instances(seq_dir, 10**9)
        per_frame = {}
        for inst in insts_needed:
            frames = [fi for fi, _, _ in obs[inst]]
            per_frame[inst] = dict(zip(frames, buffers[inst]))
        np.savez_compressed(cache_path, **{
            f"{inst}_{fi}": arr
            for inst, fr in per_frame.items() for fi, arr in fr.items()})
        print(f"Cached donor frames -> {cache_path}")

    # --- Per-pair metric ---
    records = []
    t0 = time.time()
    for k, p in enumerate(pairs):
        inst, fi = p["inst_id"], p["frame"]
        rec_gt = well[inst]
        data = np.load(os.path.join(pairs_dir, p["file"]))
        raw = data["raw"].astype(np.float64)
        completed = data["completed"].astype(np.float64)
        basis, center = data["basis"], data["center"]
        T = data["T"]
        R, t = T[:3, :3], T[:3, 3]

        donor_arrays = [arr for fj, arr in per_frame[inst].items() if fj != fi]
        if not donor_arrays:
            continue
        donor = voxel_downsample(
            np.vstack(donor_arrays).astype(np.float64), DONOR_VOXEL)

        raw_w = raw @ R.T + t
        pts_c = raw @ basis
        mir_c = pts_c.copy()
        mir_c[:, 0] = 2.0 * center[0] - pts_c[:, 0]
        mirrored_w = np.vstack([raw_w, (mir_c @ basis.T) @ R.T + t])
        comp_w = completed @ R.T + t
        method_clouds = {"raw": raw_w, "mirrored": mirrored_w, "completed": comp_w}

        d_in, _ = cKDTree(raw_w).query(donor)

        rec = {
            "frame": fi, "inst_id": inst,
            "n_raw_pts": int(len(raw)), "n_donor_pts": int(len(donor)),
            "taus": {},
        }
        d_primary = {}
        for tau in TAUS:
            novel = donor[d_in >= tau]
            entry = {"n_novel": int(len(novel))}
            if len(novel):
                for m, cloud in method_clouds.items():
                    med, cov, d = coverage(novel, cloud)
                    entry[m] = {"med_dist": round(med, 4), "cov": round(cov, 4)}
                    if tau == TAU_PRIMARY:
                        d_primary[m] = d
            rec["taus"][f"{tau:.2f}"] = entry

        rec["out_of_box"] = {
            m: round(out_of_box_frac(cloud, rec_gt, BOX_MARGIN), 4)
            for m, cloud in method_clouds.items()}

        # Region breakdown of the primary-tau novel set, relative to ego.
        novel = donor[d_in >= TAU_PRIMARY]
        if len(novel) and d_primary:
            len_c, wid_c, dy = box_coords(novel, rec_gt)
            e_len, e_wid, _ = box_coords(t[None, :], rec_gt)
            L = rec_gt["dims_lwh"][0]
            H = rec_gt["dims_lwh"][2]
            subsets = {
                "far_side": wid_c * e_wid[0] < 0,
                "far_end": (len_c * e_len[0] < 0) & (np.abs(len_c) > L / 4),
                "top": dy < -H / 4,   # world up = -Y
            }
            rec["regions"] = {}
            for name, mask in subsets.items():
                if mask.sum() < 20:
                    continue
                rec["regions"][name] = {
                    "n": int(mask.sum()),
                    **{m: round(float((d_primary[m][mask] < COV_THRESH).mean()), 4)
                       for m in METHODS},
                }

        # Symmetry self-consistency (secondary): completed vs its own mirror.
        comp_c = completed @ basis
        comp_mir = comp_c.copy()
        comp_mir[:, 0] = 2.0 * center[0] - comp_c[:, 0]
        rec["sym_self_cd"] = round(chamfer_distance(comp_c, comp_mir), 4)

        records.append(rec)
        if (k + 1) % 100 == 0 or k == len(pairs) - 1:
            el = time.time() - t0
            print(f"  pair {k + 1}/{len(pairs)}  ({el:.0f}s)", flush=True)

    payload = {
        "seq": args.seq,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "taus": TAUS, "tau_primary": TAU_PRIMARY,
            "cov_thresh": COV_THRESH, "box_margin": BOX_MARGIN,
            "donor_voxel": DONOR_VOXEL,
            "step1_index": os.path.basename(index_path),
            "novel_definition": "donor pts >= tau from every input pt "
                                "(visibility mask); donor = GT sem=10 points "
                                "of the instance, all frames except the input "
                                "frame, world frame, 3 cm voxel",
        },
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved {len(records)} records -> {out_path}")

    # Quick pooled preview (Step 3 does the real per-car statistics).
    key = f"{TAU_PRIMARY:.2f}"
    ok = [r for r in records if r["taus"][key]["n_novel"] >= 100]
    print(f"\nPreview (pairs with >= 100 novel pts @ tau={TAU_PRIMARY}: {len(ok)}):")
    for m in METHODS:
        med = np.median([r["taus"][key][m]["med_dist"] for r in ok])
        cov = np.median([r["taus"][key][m]["cov"] for r in ok])
        oob = np.median([r["out_of_box"][m] for r in ok])
        print(f"  {m:>9}: med novel-dist {med:.3f} m   cov@0.1 {cov:.3f}   "
              f"out-of-box {oob:.3f}")


if __name__ == "__main__":
    main()
