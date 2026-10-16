"""PRIVATE Q&A BACKUP follow-up to ransac_convergence.py (not a thesis result).

Question: would more RANSAC tries change WHAT is removed as ground? For a sample of frames
(every 20th) plus every frame whose kept plane was rejected at 300 tries, replays the seeded loop
to 1000 tries and compares the best plane at 300 vs 1000: inlier gain (%), angle between normals,
and whether the near-vertical check passes. Also confirms with remove_ground() itself that the
"rejected" frames really return no plane.

Pre-registered reading (2026-10-16, written before running): if the best at 300 is within ~2% of
the best at 1000 and the planes differ by < 1 degree in most frames, late "improvements" are
near-identical road planes and 300 is enough in practice; if many frames gain > 5% or the plane
tilts noticeably, 300 is genuinely tight.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import PIPELINE_CONFIG as CFG, preprocess_frame, remove_ground  # noqa: E402

SEQ = "08"
CONV = ROOT / f"docs/defense/qa_backup/ransac_convergence_seq{SEQ}.json"
OUT = ROOT / f"docs/defense/qa_backup/ransac_300_vs_1000_seq{SEQ}.json"
MINZ = CFG["ransac_min_normal_z"]


def run(frame):
    xyz = np.fromfile(ROOT / f"dataset/sequences/{SEQ}/velodyne/{frame:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    pcd = preprocess_frame(xyz, CFG)
    P = np.asarray(pcd.points)
    rng = np.random.default_rng(42)
    best, bn, snap = 0, None, {}
    for it in range(1, 1001):
        idx = rng.choice(len(P), size=3, replace=False)
        p0, p1, p2 = P[idx]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) >= 1e-10:   # same skip rule as remove_ground, without skipping snapshots
            n /= np.linalg.norm(n)
            c = int((np.abs(P @ n - n.dot(p0)) < CFG["ransac_distance_threshold"]).sum())
            if c > best:
                best, bn = c, (n if n[2] >= 0 else -n)
        if it in (100, 300, 1000):
            snap[it] = (best, bn.copy())
    (b3, n3), (b10, n10) = snap[300], snap[1000]
    ang = float(np.degrees(np.arccos(np.clip(abs(n3 @ n10), -1, 1))))
    plane_none = remove_ground(pcd, CFG)[2] is None
    return dict(frame=frame, n_points=len(P), best100=snap[100][0], best300=b3, best1000=b10,
                gain_pct=100 * (b10 - b3) / b3, angle_deg=ang,
                acc300=bool(n3[2] >= MINZ), acc1000=bool(n10[2] >= MINZ),
                n300_z=float(n3[2]), pipeline_plane_none=plane_none)


if __name__ == "__main__":
    conv = json.loads(CONV.read_text())["per_frame"]
    rejected = [r["frame"] for r in conv if not r["accepted"]]
    frames = sorted(set(range(0, len(conv), 20)) | set(rejected))
    with Pool(6) as pool:
        rows = pool.map(run, frames, chunksize=8)
    samp = [r for r in rows if r["frame"] % 20 == 0]
    rej = [r for r in rows if r["frame"] in set(rejected)]
    g = np.array([r["gain_pct"] for r in samp]); a = np.array([r["angle_deg"] for r in samp])
    summary = dict(
        sample=dict(frames=len(samp), gain_pct_median=float(np.median(g)), gain_pct_p90=float(np.percentile(g, 90)),
                    gain_pct_max=float(g.max()), frames_gain_over_2pct=int((g > 2).sum()),
                    frames_gain_over_5pct=int((g > 5).sum()), angle_median=float(np.median(a)),
                    angle_p90=float(np.percentile(a, 90)), angle_max=float(a.max()),
                    frames_angle_over_1deg=int((a > 1).sum())),
        rejected_at_300=dict(frames=len(rej),
                             pipeline_confirms_no_plane=int(sum(r["pipeline_plane_none"] for r in rej)),
                             still_rejected_at_1000=int(sum(not r["acc1000"] for r in rej)),
                             n300_z_median=float(np.median([r["n300_z"] for r in rej])) if rej else None),
    )
    OUT.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
