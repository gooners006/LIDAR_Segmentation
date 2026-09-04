"""Analyze the completion input-gate ablation (Finding #27 hardening, all 11 seqs).

Reads the GATE-OFF renders in output/experiments/gate_ablation_v2/<seq>/ (every
car track completed) and reports, per sequence and pooled:

  - gate-OFF plausible-completion rate = plausible / all completed
  - gate-ON  plausible-completion rate = plausible among gate-passing tracks,
    where the gate is reconstructed post-hoc from the recorded per-track
    ref_fit_length/ref_fit_width (fit_length >= FRAG_MIN and fit_width <= MERGE_MAX).

A completion is "plausible" if its completed-cloud footprint L / width W / height H
fall in the car box L in [3.3,4.9], W in [1.5,2.1], H in [1.1,1.7] (Finding #27).
Footprint (L,W) is the L-shape fit on the global-frame horizontal plane (X-Z);
height H is the vertical (Y) extent (global frame is Y-vertical; #27 line 676).

Validation: the frozen output/08 (shipped gate-ON render) is scored with the SAME
metric and compared to the post-hoc gate-ON rate derived from the seq-08 gate-OFF
render. If they agree (within a few points / tracks), the one-render-per-sequence
design is validated; otherwise explicit gate-ON renders are needed.

Freeze-safe: reads frozen artifacts, writes only under output/experiments/.
Run: .venv\\Scripts\\python.exe scratchpad/gate_ablation_analyze.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from completion import (  # noqa: E402
    COMPLETION_FRAGMENT_MIN_LENGTH,
    COMPLETION_MERGE_MAX_WIDTH,
    PointCloudCompleter,
)

ABL_DIR = os.path.join(ROOT, "output", "experiments", "gate_ablation_v2")
FROZEN_08 = os.path.join(ROOT, "output", "08")  # shipped gate-ON, cross-check
SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# Plausibility box (Finding #27).
L_LO, L_HI = 3.3, 4.9
W_LO, W_HI = 1.5, 2.1
H_LO, H_HI = 1.1, 1.7

OUT_JSON = os.path.join(ABL_DIR, "gate_ablation_summary.json")

_fitter = PointCloudCompleter(seed=0)  # no model needed for _lshape_axes


def completed_dims(ply_path: str):
    """Return (L, W, H) of a completed car cloud, or None if unreadable.

    L, W from the L-shape footprint on the horizontal X-Z plane (same fitter the
    gate uses); H from the vertical Y extent (global frame is Y-vertical, #27).
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) < 16:
        return None
    xz = pts[:, [0, 2]] - pts[:, [0, 2]].mean(0)
    _, _, fit_length, fit_width = _fitter._lshape_axes(xz)
    height = float(pts[:, 1].max() - pts[:, 1].min())
    return float(fit_length), float(fit_width), height


def is_plausible(dims) -> bool:
    L, W, H = dims
    return (L_LO <= L <= L_HI) and (W_LO <= W <= W_HI) and (H_LO <= H <= H_HI)


def gate_passes(track: dict) -> bool | None:
    """Post-hoc gate decision from the recorded ref-frame footprint.

    None when the track carries no ref_fit_* (e.g. the frozen output/08, whose
    completed tracks are gate-passers by construction).
    """
    if "ref_fit_length" not in track:
        return None
    return (
        track["ref_fit_length"] >= COMPLETION_FRAGMENT_MIN_LENGTH
        and track["ref_fit_width"] <= COMPLETION_MERGE_MAX_WIDTH
    )


def score_sequence(tracks_json: str, objects_dir: str):
    """Score one gate-OFF render into the full keep/drop x plausible/junk 2x2.

    Every car track completed (gate off), so plausibility is known even for the
    tracks the gate would DROP -- which is what lets us measure the cost side
    (false rejections), not just the benefit (precision up).
    """
    with open(tracks_json) as f:
        meta = json.load(f)

    # 2x2: gate keep/drop x plausible/junk.
    keep_plaus = keep_junk = drop_plaus = drop_junk = 0
    n_no_reffit = 0
    for tr in meta["tracks"]:
        if tr.get("class") != "car" or not tr.get("completed"):
            continue
        ply = os.path.join(objects_dir, f"{tr['track_id']}.ply")
        dims = completed_dims(ply)
        if dims is None:
            continue
        plausible = is_plausible(dims)

        gp = gate_passes(tr)
        if gp is None:
            # No recorded footprint (frozen 08): its completed tracks are already
            # gate-passers, so they count as "keep".
            n_no_reffit += 1
            gp = True
        if gp:
            keep_plaus += int(plausible); keep_junk += int(not plausible)
        else:
            drop_plaus += int(plausible); drop_junk += int(not plausible)

    n_completed = keep_plaus + keep_junk + drop_plaus + drop_junk   # gate-OFF universe
    n_plausible = keep_plaus + drop_plaus
    n_gate_on = keep_plaus + keep_junk
    return {
        "keep_plaus": keep_plaus, "keep_junk": keep_junk,
        "drop_plaus": drop_plaus, "drop_junk": drop_junk,   # drop_plaus = false rejections
        "n_completed": n_completed,
        "n_plausible": n_plausible,
        "off_rate": (n_plausible / n_completed) if n_completed else None,
        "n_gate_on": n_gate_on,
        "n_gate_on_plausible": keep_plaus,
        "on_rate": (keep_plaus / n_gate_on) if n_gate_on else None,
        # false-rejection rate = plausible cars dropped / all plausible cars.
        "false_rej_rate": (drop_plaus / n_plausible) if n_plausible else None,
        "n_no_reffit": n_no_reffit,
    }


def _fmt(rate):
    return f"{rate:.3f}" if rate is not None else "  -  "


def main():
    rows = {}
    pooled = dict(keep_plaus=0, keep_junk=0, drop_plaus=0, drop_junk=0)

    for seq in SEQS:
        tj = os.path.join(ABL_DIR, seq, "tracks.json")
        od = os.path.join(ABL_DIR, seq, "objects")
        if not os.path.isfile(tj):
            print(f"[skip] {seq}: no render at {tj}")
            continue
        r = score_sequence(tj, od)
        rows[seq] = r
        for k in pooled:
            pooled[k] += r[k]

    # Cross-check: frozen output/08 (shipped gate-ON) scored with the same metric.
    frozen08 = None
    f08_tj = os.path.join(FROZEN_08, "tracks.json")
    if os.path.isfile(f08_tj):
        frozen08 = score_sequence(f08_tj, os.path.join(FROZEN_08, "objects"))

    print("\n=== Completion input-gate ablation (Finding #27 hardening) ===")
    print(f"gate: fit_length >= {COMPLETION_FRAGMENT_MIN_LENGTH}, "
          f"fit_width <= {COMPLETION_MERGE_MAX_WIDTH} | "
          f"plausible box L[{L_LO},{L_HI}] W[{W_LO},{W_HI}] H[{H_LO},{H_HI}]\n")
    hdr = (f"{'seq':>4} | {'n_off':>6} {'off_rate':>8} | {'n_on':>5} {'on_rate':>8} | "
           f"{'delta':>7} | {'falseRej':>8}")
    print(hdr)
    print("-" * len(hdr))

    def _fr(r):
        # "drop_plaus / n_plausible  (count)" -- the cost side.
        return (f"{r['false_rej_rate']:.3f}({r['drop_plaus']})"
                if r["false_rej_rate"] is not None else "   -   ")

    for seq in SEQS:
        if seq not in rows:
            continue
        r = rows[seq]
        delta = (r["on_rate"] - r["off_rate"]) if (r["on_rate"] is not None and r["off_rate"] is not None) else None
        print(f"{seq:>4} | {r['n_completed']:>6} {_fmt(r['off_rate']):>8} | "
              f"{r['n_gate_on']:>5} {_fmt(r['on_rate']):>8} | "
              f"{('+' + format(delta, '.3f')) if delta is not None else '   -   ':>7} | "
              f"{_fr(r):>8}")

    kp, kj = pooled["keep_plaus"], pooled["keep_junk"]
    dp, dj = pooled["drop_plaus"], pooled["drop_junk"]
    n_completed = kp + kj + dp + dj
    n_plausible = kp + dp
    n_gate_on = kp + kj
    off_rate = n_plausible / n_completed if n_completed else None
    on_rate = kp / n_gate_on if n_gate_on else None
    false_rej_rate = dp / n_plausible if n_plausible else None
    print("-" * len(hdr))
    delta = (on_rate - off_rate) if (on_rate is not None and off_rate is not None) else None
    print(f"{'POOL':>4} | {n_completed:>6} {_fmt(off_rate):>8} | "
          f"{n_gate_on:>5} {_fmt(on_rate):>8} | "
          f"{('+' + format(delta, '.3f')) if delta is not None else '   -   ':>7} | "
          f"{(format(false_rej_rate, '.3f') + '(' + str(dp) + ')') if false_rej_rate is not None else '   -   ':>8}")
    print("\n2x2 (pooled): keep&plausible=%d  keep&junk=%d  DROP&plausible=%d (false rej)  drop&junk=%d"
          % (kp, kj, dp, dj))

    if frozen08 is not None and "08" in rows:
        print("\n=== seq-08 validation (post-hoc gate-ON vs frozen output/08) ===")
        posthoc = rows["08"]["on_rate"]
        frozen = frozen08["off_rate"]  # every frozen-08 completed track is a gate-passer
        print(f"  post-hoc gate-ON (from gate-OFF render): {_fmt(posthoc)} "
              f"(n={rows['08']['n_gate_on']})")
        print(f"  frozen output/08 gate-ON (shipped)      : {_fmt(frozen)} "
              f"(n={frozen08['n_completed']})")
        if posthoc is not None and frozen is not None:
            print(f"  |delta| = {abs(posthoc - frozen):.3f}  "
                  f"(accept < ~0.05 -> one-render design validated)")

    summary = {
        "gate": {"fragment_min_length": COMPLETION_FRAGMENT_MIN_LENGTH,
                 "merge_max_width": COMPLETION_MERGE_MAX_WIDTH},
        "plausible_box": {"L": [L_LO, L_HI], "W": [W_LO, W_HI], "H": [H_LO, H_HI]},
        "per_sequence": rows,
        "pooled": {**pooled, "off_rate": off_rate, "on_rate": on_rate},
        "frozen08_crosscheck": frozen08,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
