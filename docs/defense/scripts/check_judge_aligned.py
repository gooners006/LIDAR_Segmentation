"""Private check (qa_backup, NOT a thesis number): re-score the plausibility judge with every
raw/completed pair turned so the car's length lies on the judge's length axis (Z).

Why: harmonize.py only flips partial/completed clouds 180 deg about Z (fixes up); they keep their
scene heading (check_completion_heading.py: 37.6% within 20 deg of Z). The thesis says they are
harmonised into (width, up, length).

Alignment: rotate each pair about the cam0 up axis (Y) so the completed cloud's main horizontal
PCA axis (X-Z plane) lies on Z; the same rotation is applied to its raw partial, so the pair stays
paired. Then the unchanged harmonize.to_judge_frame (flip, FPS 1024, unit sphere) and judge.

Pre-registered reading (written before the run):
  aligned completed acceptance >= 0.75 and raw <= 0.35 -> headline holds under true alignment;
  aligned completed acceptance <  0.70                 -> 0.785 depends on the unaligned frame;
  in between                                           -> report as is.
Also reported: the original (unaligned) acceptance per heading bin, to show how orientation
sensitive the judge is on these clouds.

Reads output/experiments/plausibility_gen (read-only); writes docs/defense/qa_backup/judge_aligned_rescore.json.
Run:  .venv\\Scripts\\python.exe docs/defense/scripts/check_judge_aligned.py   (~5-10 min, GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scratchpad" / "plausibility"))
from harmonize import rot_y, to_judge_frame  # noqa: E402
from score import load_judge, mcnemar, score_batch, SEQS  # noqa: E402

GEN = REPO / "output" / "experiments" / "plausibility_gen"
OUT = REPO / "docs" / "defense" / "qa_backup" / "judge_aligned_rescore.json"


def main_axis_deg(pts):
    """Angle of the main horizontal (X-Z) PCA axis from +Z, in degrees, signed (atan2(x, z))."""
    xz = pts[:, [0, 2]] - pts[:, [0, 2]].mean(0)
    _, v = np.linalg.eigh(np.cov(xz.T))
    mx, mz = v[:, 1]
    return np.degrees(np.arctan2(mx, mz))


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_judge(device)
rows = []
for seq in SEQS:
    for t in json.load(open(GEN / seq / "tracks.json"))["tracks"]:
        tid = t["track_id"]
        cp, pp = GEN / seq / "objects" / f"{tid}.ply", GEN / seq / "objects" / f"{tid}_partial.ply"
        if not t.get("completed") or not cp.exists() or not pp.exists():
            continue
        comp = np.asarray(o3d.io.read_point_cloud(str(cp)).points)
        part = np.asarray(o3d.io.read_point_cloud(str(pp)).points)
        a = main_axis_deg(comp)
        R = rot_y(-a)                                  # pts @ R.T turns the main axis onto +Z
        c0 = comp.mean(0)                              # rotate both about the completed centre
        comp_al = (comp - c0) @ R.T + c0
        part_al = (part - c0) @ R.T + c0
        rows.append({"seq": seq, "tid": tid, "deg_from_z": float(min(abs(a) % 180, 180 - abs(a) % 180)),
                     "raw": to_judge_frame(part, "partial", seed=tid),
                     "comp": to_judge_frame(comp, "completed", seed=tid),
                     "raw_al": to_judge_frame(part_al, "partial", seed=tid),
                     "comp_al": to_judge_frame(comp_al, "completed", seed=tid)})
    print(f"[seq {seq}] {len(rows)} pairs so far", flush=True)

res = {}
for k in ("raw", "comp", "raw_al", "comp_al"):
    p, acc = score_batch(model, [r[k] for r in rows], device)
    res[k] = (p, acc)
ang = np.array([r["deg_from_z"] for r in rows])
# sanity: must reproduce the thesis run (summary.json: raw 0.253, completed 0.785)
summary = {
    "n": len(rows),
    "unaligned": {"raw_accept": float(res["raw"][1].mean()), "completed_accept": float(res["comp"][1].mean()),
                  "raw_carprob": float(res["raw"][0].mean()), "completed_carprob": float(res["comp"][0].mean())},
    "aligned": {"raw_accept": float(res["raw_al"][1].mean()), "completed_accept": float(res["comp_al"][1].mean()),
                "raw_carprob": float(res["raw_al"][0].mean()), "completed_carprob": float(res["comp_al"][0].mean())},
    "aligned_mcnemar": dict(zip(("b_raw_only", "c_completed_only", "p"), mcnemar(res["raw_al"][1], res["comp_al"][1]))),
    "unaligned_by_heading_bin": {},
    "aligned_by_heading_bin": {},
}
for name, m in (("within_20_of_Z", ang <= 20), ("20_to_70", (ang > 20) & (ang < 70)), ("within_20_of_X", ang >= 70)):
    summary["unaligned_by_heading_bin"][name] = {"n": int(m.sum()), "raw_accept": float(res["raw"][1][m].mean()),
                                                 "completed_accept": float(res["comp"][1][m].mean())}
    summary["aligned_by_heading_bin"][name] = {"n": int(m.sum()), "raw_accept": float(res["raw_al"][1][m].mean()),
                                               "completed_accept": float(res["comp_al"][1][m].mean())}
summary["per_seq_aligned_completed_accept"] = {
    s: float(res["comp_al"][1][np.array([r["seq"] == s for r in rows])].mean()) for s in SEQS}
OUT.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(f"wrote {OUT}")
