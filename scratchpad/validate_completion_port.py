"""Validate the completion.py complete() port against the validated step-2 path.

The corrected normalization was validated in scratchpad/verify_pcn_step2.py
(complete_corrected). This checks that the ported PointCloudCompleter.complete()
reproduces it numerically on the same real seq-08 partials. With the RNG re-seeded
identically before each call, the only randomness (_fix_size padding) matches, so
outputs should be bit-for-bit identical -> the port is faithful and inherits the
step-2 validation.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcn import PCN  # noqa: E402
from completion import PointCloudCompleter  # noqa: E402

# Reuse the validated reference implementation and the real-data miner from step 2.
sys.path.insert(0, os.path.dirname(__file__))
from verify_pcn_step2 import (  # noqa: E402
    complete_corrected, mine_static_cars, PROJECT_ROOT,
)

SEED = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kitti_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")

    # Reference model (same weights complete() loads).
    ck = torch.load(kitti_ckpt, map_location=device, weights_only=False)
    ref_model = PCN(num_coarse=1024, grid_size=2).to(device)
    ref_model.load_state_dict(ck["model_state_dict"]); ref_model.eval()

    completer = PointCloudCompleter(model_path=kitti_ckpt, seed=SEED)

    # A handful of real single-frame car partials (velodyne frame).
    static = mine_static_cars("08", n_frames=400, min_frames=6, static_thresh=2.0)
    partials = []
    for tid, entries, _ in static:
        ref_idx = int(np.argmax([len(e[1]) for e in entries]))
        p = entries[ref_idx][1]
        if 40 <= len(p) <= 300:
            partials.append((tid, p))
        if len(partials) >= 8:
            break

    print(f"\n==== Equivalence: complete() vs validated complete_corrected ====")
    print(f"{'track':>6} {'pts':>5} {'out':>5} {'max|delta|(m)':>14} {'skip':>20}")
    max_overall = 0.0
    for tid, partial in partials:
        # Match the RNG state seen by each path's _fix_size call.
        ref = complete_corrected(ref_model, device, partial,
                                 np.random.default_rng(SEED))
        completer._rng = np.random.default_rng(SEED)
        out, skip = completer.complete(partial, "car")

        if ref is None or skip is not None:
            print(f"{tid:>6} {len(partial):>5} {'-':>5} {'-':>14} {str(skip):>20}")
            continue
        delta = float(np.abs(out - ref).max())
        max_overall = max(max_overall, delta)
        print(f"{tid:>6} {len(partial):>5} {len(out):>5} {delta:>14.2e} {str(skip):>20}")

    print("-" * 56)
    print(f"max |delta| across all tracks: {max_overall:.2e} m")
    print("PASS (port is faithful)" if max_overall < 1e-5
          else "MISMATCH — investigate")


if __name__ == "__main__":
    main()
