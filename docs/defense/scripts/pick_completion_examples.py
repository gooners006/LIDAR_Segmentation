"""Pick real-car completion examples for the defense deck by a fixed rule (not by eye).

Re-scores the thesis plausibility set (output/experiments/plausibility_gen, N=2822 pairs) with
the same judge and harmonisation as scratchpad/plausibility/score.py, because per-car verdicts
were never saved (only summary.json / per_seq.json).

Sanity gate (pre-registered): pooled raw / completed acceptance and the McNemar counts must
reproduce tab:plausibility (0.253 / 0.785, 115 vs 1615). If not, stop: the candidates would not
come from the population the thesis reports.

Selection rule (pre-registered, fixed before looking at any candidate):
  1. judge flip: raw partial rejected, completion accepted (the 1615 cars behind the headline);
  2. typical density: raw partial point count within the 25th-75th percentile of all 2822 pairs;
  3. 12 drawn at random (seed 0) from the cars meeting 1-2, any sequence.
The user then picks one from the contact sheet; the slide notes state the rule.

Outputs (no existing results touched):
  docs/defense/figures/completion_candidates.png   contact sheet, BEV + side view per car
  docs/defense/figures/completion_candidates.json  per-candidate seq / track / counts / car-prob

Run:  .venv\\Scripts\\python.exe docs/defense/scripts/pick_completion_examples.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scratchpad" / "plausibility"))
import score  # noqa: E402  (load_judge, load_pairs, score_batch, mcnemar, SEQS, GEN)

FIG = REPO / "docs" / "defense" / "figures"
N_PICK, SEED = 12, 0
BLUE, GREEN = "#1F6FEB", "#12A150"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = score.load_judge(device)

    valid = []
    for seq in score.SEQS:
        for p in score.load_pairs(seq):
            if p["raw"] is not None:
                p["seq"] = seq
                valid.append(p)
    raw_p, raw_a = score.score_batch(model, [p["raw"] for p in valid], device)
    comp_p, comp_a = score.score_batch(model, [p["completed"] for p in valid], device)
    b, c, _ = score.mcnemar(raw_a, comp_a)
    print(f"[check] N={len(valid)} raw {raw_a.mean():.3f} completed {comp_a.mean():.3f} "
          f"McNemar {b} vs {c}  (thesis: 2822, 0.253, 0.785, 115 vs 1615)")
    if not (len(valid) == 2822 and round(raw_a.mean(), 3) == 0.253
            and round(comp_a.mean(), 3) == 0.785 and (b, c) == (115, 1615)):
        sys.exit("[check] FAILED: does not reproduce tab:plausibility; not picking examples.")

    def cloud(p, suffix):
        f = score.GEN + f"/{p['seq']}/objects/{p['tid']}{suffix}.ply"
        return np.asarray(o3d.io.read_point_cloud(f).points)

    n_raw = np.array([len(cloud(p, "_partial")) for p in valid])
    lo, hi = np.percentile(n_raw, [25, 75])
    pool = np.flatnonzero(~raw_a & comp_a & (n_raw >= lo) & (n_raw <= hi))
    print(f"[rule] raw points P25-P75 = {lo:.0f}-{hi:.0f}; flips in band: {len(pool)} "
          f"of {c}")
    pick = np.random.default_rng(SEED).choice(pool, N_PICK, replace=False)

    rows = []
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(6, 4, hspace=0.45, wspace=0.18)
    for k, i in enumerate(pick):
        p = valid[i]
        part, comp = cloud(p, "_partial"), cloud(p, "")
        rows.append({"label": chr(65 + k), "seq": p["seq"], "track": int(p["tid"]),
                     "raw_points": int(n_raw[i]), "mover": bool(p["mover"]),
                     "raw_carprob": round(float(raw_p[i]), 3),
                     "completed_carprob": round(float(comp_p[i]), 3)})
        r, col = 2 * (k // 4), k % 4
        # Saved clouds are in the canonical car frame: X width, Y height (down), Z length.
        for j, (u, v, name) in enumerate([(2, 0, "top"), (2, 1, "side")]):
            ax = fig.add_subplot(gs[r + j, col])
            sgn = -1 if v == 1 else 1
            ax.scatter(comp[:, u], sgn * comp[:, v], s=1, c=GREEN, alpha=0.45, lw=0)
            ax.scatter(part[:, u], sgn * part[:, v], s=2, c=BLUE, lw=0)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_title(f"{chr(65 + k)}  seq {p['seq']} / {p['tid']}  ·  {n_raw[i]} pts  ·  "
                             f"car-prob {raw_p[i]:.2f} → {comp_p[i]:.2f}", fontsize=11, loc="left")
            ax.text(0.01, 0.02, name, transform=ax.transAxes, fontsize=9, color="#6B7280")
    fig.suptitle("Completion candidates (judge flips, typical density, random seed 0) — "
                 "blue = single-frame scan, green = completed", fontsize=14)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "completion_candidates.png", dpi=110, bbox_inches="tight")
    json.dump({"rule": "raw rejected & completed accepted; raw points in "
                       f"[{lo:.0f}, {hi:.0f}] (P25-P75 of N=2822); seed {SEED}",
               "pool_size": int(len(pool)), "candidates": rows},
              open(FIG / "completion_candidates.json", "w"), indent=1)
    print(f"wrote {FIG / 'completion_candidates.png'}")


if __name__ == "__main__":
    main()
