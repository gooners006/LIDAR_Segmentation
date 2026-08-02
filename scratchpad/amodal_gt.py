"""Direction 4a — Step 0: amodal GT box builder for static cars.

Accumulates SemanticKITTI static-car (sem=10) instance points across all frames
of a sequence into the global frame (poses @ Tr; cam0 frame: X right, Y down,
Z forward — ground plane is X-Z, up is -Y), fits an oriented box per instance
(L-shape angle search in X-Z + Y-extent height), computes viewpoint-azimuth
coverage, and caches everything to output/<seq>/amodal_gt.json.

Moving cars are sem=252 in SemanticKITTI, so sem=10 alone selects parked cars;
a global center-spread guard additionally rejects mislabeled movers / ID reuse.

Run:
    .venv\\Scripts\\python.exe scratchpad\\amodal_gt.py --seq 08
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import load_calib, load_poses  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "car_sem_id": 10,           # static car class
    "moving_sem_id": 252,       # moving-car: instances ever seen moving are rejected
    "min_frame_points": 10,     # min points for a per-frame observation to count
    "voxel": 0.03,              # m, accumulation voxel size
    "fit_max_points": 6000,     # subsample cap for the box angle search
    "pct_lo": 0.5,              # robust extent percentiles
    "pct_hi": 99.5,
    "end_slab": 0.35,           # m, slab inside each box face for support counts
    "end_support_min": 40,      # min points at BOTH length ends / width sides
    # truncation detector: points > overhang_margin beyond a box face mean the
    # percentile trim cut a real (sparsely observed) tail -> extent_uncertain
    "overhang_margin": 0.15,    # m beyond the face
    # points allowed beyond margin per face: max(overhang_max, frac * n) —
    # absolute floor for sparse clouds, relative for dense ones where a few
    # dozen label-bleed points are noise, not a truncated car end
    "overhang_max": 15,
    "overhang_frac": 0.001,
    # well_observed criteria
    "min_frames": 5,
    "min_acc_points": 1500,
    "min_azimuth_bins": 3,      # of 8 x 45-degree bins
    "width_view_max_deg": 30.0,  # a view this close to the length axis sees full W
    "center_spread_max_m": 2.0,  # static sanity: per-frame bbox-center spread (X-Z)
}


# --------------------------------------------------------------------------
# Geometry helpers (Step 1 imports fit_oriented_box_xz from here)
# --------------------------------------------------------------------------

def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one point per voxel (deterministic: lowest original index)."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _closeness_score(c1: np.ndarray, c2: np.ndarray) -> float:
    """Zhang et al. L-shape closeness criterion (higher = better fit)."""
    d1 = np.minimum(c1.max() - c1, c1 - c1.min())
    d2 = np.minimum(c2.max() - c2, c2 - c2.min())
    d = np.maximum(np.minimum(d1, d2), 0.01)
    return float(np.sum(1.0 / d))


def fit_oriented_box_xz(pts_xz: np.ndarray, *, step_deg: float = 0.5,
                        max_points: int = 6000, extent: str = "percentile",
                        pct_lo: float = 0.5, pct_hi: float = 99.5,
                        seed: int = 0):
    """Search-based oriented rectangle fit in the ground (X-Z) plane.

    extent: "percentile" (robust, for accumulated clouds with label noise) or
    "minmax" (for small single-frame partials where trimming costs real edge).

    Returns dict with yaw_deg (length-axis direction from +X toward +Z, in
    [0, 180)), length, width, center_xz, and the length/width axis coords
    (c_len, c_wid, centered on the box) for support/overhang diagnostics.
    """
    fit_pts = pts_xz
    if len(fit_pts) > max_points:
        rng = np.random.default_rng(seed)
        fit_pts = fit_pts[rng.choice(len(fit_pts), max_points, replace=False)]

    best_theta, best_score = 0.0, -np.inf
    for theta in np.deg2rad(np.arange(0.0, 90.0, step_deg)):
        e1 = np.array([np.cos(theta), np.sin(theta)])
        e2 = np.array([-np.sin(theta), np.cos(theta)])
        score = _closeness_score(fit_pts @ e1, fit_pts @ e2)
        if score > best_score:
            best_score, best_theta = score, theta

    e1 = np.array([np.cos(best_theta), np.sin(best_theta)])
    e2 = np.array([-np.sin(best_theta), np.cos(best_theta)])
    c1 = pts_xz @ e1
    c2 = pts_xz @ e2
    if extent == "percentile":
        lo1, hi1 = np.percentile(c1, [pct_lo, pct_hi])
        lo2, hi2 = np.percentile(c2, [pct_lo, pct_hi])
    else:
        lo1, hi1 = float(c1.min()), float(c1.max())
        lo2, hi2 = float(c2.min()), float(c2.max())
    ext1, ext2 = hi1 - lo1, hi2 - lo2
    center = 0.5 * (lo1 + hi1) * e1 + 0.5 * (lo2 + hi2) * e2

    if ext1 >= ext2:
        length, width = ext1, ext2
        yaw = np.rad2deg(best_theta)
        c_len, c_wid = c1 - 0.5 * (lo1 + hi1), c2 - 0.5 * (lo2 + hi2)
    else:
        length, width = ext2, ext1
        yaw = np.rad2deg(best_theta) + 90.0
        c_len, c_wid = c2 - 0.5 * (lo2 + hi2), c1 - 0.5 * (lo1 + hi1)
    return {
        "yaw_deg": float(yaw % 180.0),
        "length": float(length),
        "width": float(width),
        "center_xz": [float(center[0]), float(center[1])],
        "c_len": c_len,
        "c_wid": c_wid,
    }


def _fold_90(a: np.ndarray) -> np.ndarray:
    """Fold angles (deg) to distance-from-axis in [0, 90]."""
    a = np.abs(a) % 180.0
    return np.minimum(a, 180.0 - a)


def _circular_span_deg(az: np.ndarray) -> float:
    """Angular span covered by azimuths = 360 minus the largest gap."""
    if len(az) < 2:
        return 0.0
    s = np.sort(az % 360.0)
    gaps = np.diff(np.concatenate([s, [s[0] + 360.0]]))
    return float(360.0 - gaps.max())


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------

def accumulate_instances(seq_dir: str, n_frames: int):
    """Scan the sequence, accumulating world-frame points per car instance.

    Returns (buffers, obs, moving_ids) where buffers[inst] is a list of
    per-frame point arrays (voxel-downsampled), obs[inst] is a list of
    (frame_idx, n_raw_points, ego_xz), and moving_ids is the set of instance
    IDs ever labeled moving-car (a stop-and-go car keeps its ID across the
    10 <-> 252 class switch, so its sem=10 accumulation would smear).
    """
    cfg = CONFIG
    bin_paths = sorted(glob.glob(os.path.join(seq_dir, "velodyne", "*.bin")))
    label_paths = sorted(glob.glob(os.path.join(seq_dir, "labels", "*.label")))
    n = min(len(bin_paths), len(label_paths), n_frames)
    poses = load_poses(os.path.join(seq_dir, "poses.txt"))
    Tr = load_calib(os.path.join(seq_dir, "calib.txt"))["Tr"]

    buffers: dict[int, list] = {}
    obs: dict[int, list] = {}
    moving_ids: set[int] = set()

    t0 = time.time()
    for fi in range(n):
        xyz = np.fromfile(bin_paths[fi], dtype=np.float32).reshape(-1, 4)[:, :3]
        raw = np.fromfile(label_paths[fi], dtype=np.uint32)
        sem = raw & 0xFFFF
        inst = raw >> 16

        moving_mask = (sem == cfg["moving_sem_id"]) & (inst > 0)
        if moving_mask.any():
            moving_ids.update(int(g) for g in np.unique(inst[moving_mask]))

        car_mask = (sem == cfg["car_sem_id"]) & (inst > 0)
        if not car_mask.any():
            continue

        T = poses[fi] @ Tr
        R, t = T[:3, :3], T[:3, 3]
        ego_xz = np.array([t[0], t[2]])

        car_xyz = xyz[car_mask].astype(np.float64)
        car_inst = inst[car_mask]
        world = car_xyz @ R.T + t

        for gi in np.unique(car_inst):
            m = car_inst == gi
            n_raw = int(m.sum())
            if n_raw < cfg["min_frame_points"]:
                continue
            pts = voxel_downsample(world[m], cfg["voxel"]).astype(np.float32)
            buffers.setdefault(int(gi), []).append(pts)
            obs.setdefault(int(gi), []).append((fi, n_raw, ego_xz))

        if (fi + 1) % 200 == 0 or fi == n - 1:
            el = time.time() - t0
            print(f"  frame {fi + 1}/{n}  ({el:.0f}s, "
                  f"{len(buffers)} instances so far)", flush=True)
    return buffers, obs, moving_ids


# --------------------------------------------------------------------------
# Per-instance box fit + viewpoint coverage
# --------------------------------------------------------------------------

def build_record(inst_id: int, frames_pts: list, observations: list,
                 also_moving: bool):
    cfg = CONFIG
    acc = voxel_downsample(np.vstack(frames_pts).astype(np.float64), cfg["voxel"])
    frames = [int(fi) for fi, _, _ in observations]
    n_frames = len(frames)

    rec = {
        "inst_id": inst_id,
        "n_frames": n_frames,
        "first_frame": min(frames),
        "last_frame": max(frames),
        "n_points_accumulated": int(len(acc)),
        "also_moving": also_moving,
        "frames": [[int(fi), int(nr)] for fi, nr, _ in observations],
    }

    if len(acc) < 100:
        rec["well_observed"] = False
        rec["reject_reason"] = "too_few_points_for_fit"
        return rec

    # Static sanity: spread of per-frame bbox centers in the ground plane.
    frame_centers = np.array([
        0.5 * (p[:, [0, 2]].min(0) + p[:, [0, 2]].max(0)) for p in frames_pts
    ])
    spread = float(np.linalg.norm(frame_centers.max(0) - frame_centers.min(0)))
    rec["center_spread_xz_m"] = round(spread, 3)

    # Oriented box: L-shape fit in X-Z, robust Y extent for height (Y is down).
    box = fit_oriented_box_xz(acc[:, [0, 2]], max_points=cfg["fit_max_points"],
                              pct_lo=cfg["pct_lo"], pct_hi=cfg["pct_hi"])
    y_lo, y_hi = np.percentile(acc[:, 1], [cfg["pct_lo"], cfg["pct_hi"]])
    rec.update({
        "center_world": [box["center_xz"][0],
                         float(0.5 * (y_lo + y_hi)),
                         box["center_xz"][1]],
        "yaw_deg": box["yaw_deg"],
        "dims_lwh": [round(box["length"], 3), round(box["width"], 3),
                     round(float(y_hi - y_lo), 3)],
    })

    # Face support: were both length ends / width sides actually observed?
    # A box face that was truly seen has many points in its slab; a face
    # extrapolated from nothing has (near) none.
    c_len, c_wid = box["c_len"], box["c_wid"]
    hl, hw = box["length"] / 2, box["width"] / 2
    slab = cfg["end_slab"]
    end_support = [int((c_len > hl - slab).sum()), int((c_len < -hl + slab).sum())]
    side_support = [int((c_wid > hw - slab).sum()), int((c_wid < -hw + slab).sum())]

    # Truncation: a well-covered car has its 0.5% percentile tail hugging the
    # face (few cm); a sparsely observed far end extends far beyond it.
    m = cfg["overhang_margin"]
    omax = max(cfg["overhang_max"], int(cfg["overhang_frac"] * len(acc)))
    overhang = [int((c_len > hl + m).sum()), int((c_len < -hl - m).sum()),
                int((c_wid > hw + m).sum()), int((c_wid < -hw - m).sum())]

    rec.update({
        "length_end_support": end_support,
        "width_side_support": side_support,
        "ends_seen": bool(min(end_support) >= cfg["end_support_min"]),
        "sides_seen": bool(min(side_support) >= cfg["end_support_min"]),
        "face_overhang": overhang,
        "extent_uncertain": bool(max(overhang) > omax),
    })

    # Viewpoint-azimuth coverage (ego position relative to the box center).
    center_xz = np.array(box["center_xz"])
    egos = np.array([e for _, _, e in observations])
    v = egos - center_xz
    az = np.rad2deg(np.arctan2(v[:, 1], v[:, 0]))  # deg from +X toward +Z
    bins_occupied = int(len(np.unique((np.floor(az / 45.0).astype(int)) % 8)))
    rel = az - box["yaw_deg"]
    ang_len = _fold_90(rel)          # 0 = viewing along length axis (sees W face)
    ang_wid = _fold_90(rel - 90.0)   # 0 = viewing broadside (sees L face)
    sin_rel = np.sin(np.deg2rad(rel))
    both_sides = bool((sin_rel > 0.2).any() and (sin_rel < -0.2).any())
    width_evidence = bool(ang_len.min() <= cfg["width_view_max_deg"]) or both_sides

    rec.update({
        "azimuth_bins_occupied": bins_occupied,
        "azimuth_span_deg": round(_circular_span_deg(az), 1),
        "min_view_angle_to_length_axis_deg": round(float(ang_len.min()), 1),
        "min_view_angle_to_width_axis_deg": round(float(ang_wid.min()), 1),
        "both_sides_seen": both_sides,
        "width_evidence": width_evidence,
    })

    rec["well_observed"] = bool(
        not also_moving
        and n_frames >= cfg["min_frames"]
        and len(acc) >= cfg["min_acc_points"]
        and bins_occupied >= cfg["min_azimuth_bins"]
        and width_evidence
        and rec["ends_seen"]
        and rec["sides_seen"]
        and not rec["extent_uncertain"]
        and spread <= cfg["center_spread_max_m"]
    )
    return rec


# --------------------------------------------------------------------------
# Validation report
# --------------------------------------------------------------------------

SANITY = {"L": (3.5, 5.0), "W": (1.6, 2.0), "H": (1.4, 1.6)}


def report(records: list):
    fitted = [r for r in records if "dims_lwh" in r]
    well = [r for r in fitted if r["well_observed"]]
    print("\n==== Amodal GT summary ====")
    print(f"Instances observed (>= {CONFIG['min_frame_points']} pts in >= 1 frame): "
          f"{len(records)}")
    print(f"Instances with box fit (>= 100 acc points): {len(fitted)}")
    print(f"Well-observed: {len(well)}")

    cfg = CONFIG
    checks = {
        "also_moving": lambda r: r["also_moving"],
        "too_few_frames": lambda r: r["n_frames"] < cfg["min_frames"],
        "too_few_points": lambda r: r["n_points_accumulated"] < cfg["min_acc_points"],
        "azimuth_bins": lambda r: r["azimuth_bins_occupied"] < cfg["min_azimuth_bins"],
        "no_width_evidence": lambda r: not r["width_evidence"],
        "ends_not_seen": lambda r: not r["ends_seen"],
        "sides_not_seen": lambda r: not r["sides_seen"],
        "extent_uncertain": lambda r: r["extent_uncertain"],
        "center_spread": lambda r: r["center_spread_xz_m"] > cfg["center_spread_max_m"],
    }
    rej = [r for r in fitted if not r["well_observed"]]
    if rej:
        print("\nRejection criteria hit (non-exclusive), among fitted:")
        for name, fn in checks.items():
            print(f"  {name:>18}: {sum(1 for r in rej if fn(r))}")

    for name, group in [("ALL FITTED", fitted), ("WELL-OBSERVED", well)]:
        if not group:
            continue
        dims = np.array([r["dims_lwh"] for r in group])
        print(f"\n-- {name} (n={len(group)}) dims (m):")
        print(f"{'':>4} {'mean':>6} {'med':>6} {'p10':>6} {'p90':>6} "
              f"{'sane-range':>12} {'in-range':>9}")
        for i, key in enumerate("LWH"):
            lo, hi = SANITY[key]
            in_range = int(((dims[:, i] >= lo) & (dims[:, i] <= hi)).sum())
            print(f"{key:>4} {dims[:, i].mean():>6.2f} "
                  f"{np.median(dims[:, i]):>6.2f} "
                  f"{np.percentile(dims[:, i], 10):>6.2f} "
                  f"{np.percentile(dims[:, i], 90):>6.2f} "
                  f"{f'{lo}-{hi}':>12} "
                  f"{in_range:>4}/{len(group)}")

    if well:
        dims = np.array([r["dims_lwh"] for r in well])
        off = np.abs(dims - [[4.2, 1.8, 1.5]]).max(axis=1)
        worst = np.argsort(off)[::-1][:8]
        print("\n-- Largest well-observed outliers vs nominal 4.2x1.8x1.5:")
        for i in worst:
            r = well[i]
            print(f"  inst {r['inst_id']:>6}: LWH={r['dims_lwh']}  "
                  f"frames={r['n_frames']}  pts={r['n_points_accumulated']}  "
                  f"spread={r['center_spread_xz_m']}m  "
                  f"span={r['azimuth_span_deg']}deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=10**9,
                    help="Max frames to scan (default: all)")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default output/<seq>/amodal_gt.json)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        PROJECT_ROOT, "output", args.seq, "amodal_gt.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to replace it.")

    seq_dir = os.path.join(PROJECT_ROOT, "dataset", "sequences", args.seq)
    print(f"Accumulating static-car (sem={CONFIG['car_sem_id']}) instances "
          f"from seq {args.seq} ...")
    buffers, obs, moving_ids = accumulate_instances(seq_dir, args.frames)
    n_also_moving = len(moving_ids & set(buffers))
    print(f"{len(moving_ids)} instance IDs ever labeled moving; "
          f"{n_also_moving} overlap the sem=10 set (excluded from well_observed)")

    print(f"\nFitting boxes for {len(buffers)} instances ...")
    records = [build_record(gi, buffers[gi], obs[gi], gi in moving_ids)
               for gi in sorted(buffers)]

    report(records)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "seq": args.seq,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_convention": "world = cam0 frame via poses @ Tr; X right, "
                            "Y down, Z forward; ground plane X-Z; up = -Y; "
                            "yaw_deg = length-axis angle from +X toward +Z, "
                            "in [0, 180)",
        "config": CONFIG,
        "instances": {str(r["inst_id"]): r for r in records},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved {len(records)} instance records -> {out_path}")


if __name__ == "__main__":
    main()
