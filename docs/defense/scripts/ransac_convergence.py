"""PRIVATE Q&A BACKUP (not a thesis result, not for slides): does 300 RANSAC tries suffice?

For every frame of a sequence, replays remove_ground()'s seeded loop (src/pipeline.py) on the
preprocessed scan and records when the final best plane was first found, the road fraction
w = best inliers / points, and whether the kept plane passes the near-vertical check. Every
200th frame, the replay is asserted to match remove_ground() itself.

Pre-registered reading (2026-10-16): if every frame's best is found well before try 300
(worst <= ~100), 300 is comfortably enough; if frames are still improving near 300, 300 is tight
for those frames, and that's a weakness to prepare for.

Usage: .venv/Scripts/python.exe docs/defense/scripts/ransac_convergence.py [seq]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import PIPELINE_CONFIG as CFG, preprocess_frame, remove_ground  # noqa: E402

SEQ = sys.argv[1] if len(sys.argv) > 1 else "08"
OUT = ROOT / f"docs/defense/qa_backup/ransac_convergence_seq{SEQ}.json"


def run(frame):
    xyz = np.fromfile(ROOT / f"dataset/sequences/{SEQ}/velodyne/{frame:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    pcd = preprocess_frame(xyz, CFG)
    P = np.asarray(pcd.points)
    rng = np.random.default_rng(42)
    best, best_it, best_n = 0, 0, None
    for it in range(1, CFG["ransac_iterations"] + 1):
        idx = rng.choice(len(P), size=CFG["ransac_n"], replace=False)
        p0, p1, p2 = P[idx]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-10:
            continue
        n /= np.linalg.norm(n)
        c = int((np.abs(P @ n + -n.dot(p0)) < CFG["ransac_distance_threshold"]).sum())
        if c > best:
            best, best_it, best_n = c, it, n
    if frame % 200 == 0:
        assert len(remove_ground(pcd, CFG)[3]) == best, f"replay diverged on frame {frame}"
    return dict(frame=frame, n_points=len(P), best=best, best_it=best_it,
                w=best / len(P), accepted=bool(abs(best_n[2]) >= CFG["ransac_min_normal_z"]))


if __name__ == "__main__":
    frames = sorted(int(p.stem) for p in (ROOT / f"dataset/sequences/{SEQ}/velodyne").glob("*.bin"))
    with Pool(6) as pool:
        rows = pool.map(run, frames, chunksize=20)
    it = np.array([r["best_it"] for r in rows]); w = np.array([r["w"] for r in rows])
    summary = dict(
        seq=SEQ, frames=len(rows), iterations=CFG["ransac_iterations"],
        best_found_at=dict(median=float(np.median(it)), p90=float(np.percentile(it, 90)),
                           p99=float(np.percentile(it, 99)), max=int(it.max()),
                           frames_after_100=int((it > 100).sum()), frames_after_200=int((it > 200).sum())),
        road_fraction=dict(min=float(w.min()), p1=float(np.percentile(w, 1)),
                           median=float(np.median(w)), frames_below_0_25=int((w < 0.25).sum())),
        frames_plane_rejected=int(sum(not r["accepted"] for r in rows)),
        worst_frames=sorted(rows, key=lambda r: -r["best_it"])[:5],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(summary=summary, per_frame=rows), indent=1))
    print(json.dumps(summary, indent=1))
