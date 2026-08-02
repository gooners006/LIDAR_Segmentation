"""Invariant tests locking down conventions/behaviors this repo has violated
before (see docs/plans/delegate_brief_2026_08_02.md, T4):

- sensor<->global round-trip (Finding #39 frame-convention trap).
- estimate_canonical_frame() purity and extend-only length push (#35/#36).
- track_length_estimate() fallback/quantile branches (#36).
- complete() order-independence under sample_seed (Finding #38).

Run with: .venv\\Scripts\\python.exe -m pytest src/test_invariants.py -v
"""

import math
import os

import numpy as np
import pytest

from completion import (
    COMPLETION_CAR_LENGTH_PRIOR,
    COMPLETION_LENGTH_TRACK_OFFSET,
    COMPLETION_LENGTH_TRACK_QUANTILE,
    PointCloudCompleter,
    track_length_estimate,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCN_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "pcn_kitti_best.pth")


# ---------------------------------------------------------------------------
# Sensor <-> global round-trip (CLAUDE.md frame conventions; Finding #39)
# ---------------------------------------------------------------------------

def _random_orthonormal(rng: np.random.Generator) -> np.ndarray:
    """Random rotation matrix via QR decomposition of a random matrix."""
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    # Fix sign ambiguity from QR so det(q) == 1 (proper rotation).
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


@pytest.mark.parametrize("trial", range(5))
def test_sensor_global_round_trip(trial):
    """sensor->global: s @ R.T + t; global->sensor: (g - t) @ R (CLAUDE.md)."""
    rng = np.random.default_rng(trial)
    R = _random_orthonormal(rng)
    t = rng.uniform(-50, 50, size=3)
    s = rng.uniform(-20, 20, size=(200, 3))

    g = s @ R.T + t
    s_recovered = (g - t) @ R

    np.testing.assert_allclose(s_recovered, s, atol=1e-9)


# ---------------------------------------------------------------------------
# estimate_canonical_frame(): purity + extend-only length push
# ---------------------------------------------------------------------------

def _synthetic_partial(n=300, x0=5.0, y0=0.0, length=3.5, width=1.8, seed=0):
    """Axis-aligned rectangular footprint standing in for a partial car
    cluster, offset from the ego origin so the length/width priors have a
    well-defined push direction.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform([x0, y0], [x0 + length, y0 + width], size=(n, 2))
    z = rng.uniform(0.0, 1.4, size=n)
    return np.column_stack([xy, z])


def test_estimate_canonical_frame_is_pure():
    completer = PointCloudCompleter(model_path=None, length_prior=None)
    pts = _synthetic_partial()

    frame_a, skip_a = completer.estimate_canonical_frame(pts, length_estimate=4.5)
    frame_b, skip_b = completer.estimate_canonical_frame(pts, length_estimate=4.5)

    assert skip_a is None and skip_b is None
    np.testing.assert_array_equal(frame_a["basis"], frame_b["basis"])
    np.testing.assert_array_equal(frame_a["center"], frame_b["center"])
    assert frame_a["radius"] == frame_b["radius"]
    assert frame_a["fit_length"] == frame_b["fit_length"]
    assert frame_a["fit_width"] == frame_b["fit_width"]


def test_estimate_canonical_frame_length_push_extend_only():
    # length_prior=None on the completer so the "no push" baseline is a
    # simple explicit length_estimate=None call.
    completer = PointCloudCompleter(model_path=None, length_prior=None)
    pts = _synthetic_partial(length=3.5)  # observed length ~3.5 m

    frame_no_push, skip = completer.estimate_canonical_frame(pts, length_estimate=None)
    assert skip is None
    center0 = frame_no_push["center"]
    assert abs(center0[2]) > 1e-6, "test fixture must give a non-degenerate sign"

    # Target well below the observed length -> push must clip to zero
    # (extend-only: never retracts).
    frame_no_op, skip = completer.estimate_canonical_frame(pts, length_estimate=0.01)
    assert skip is None
    assert frame_no_op["center"][2] == pytest.approx(center0[2])

    # Target well above the observed length -> push must fire, and its
    # displacement must point the same way as the pre-push sign(center[2]).
    frame_push, skip = completer.estimate_canonical_frame(pts, length_estimate=100.0)
    assert skip is None
    displacement = frame_push["center"][2] - center0[2]
    assert displacement != 0.0
    assert np.sign(displacement) == np.sign(center0[2])


# ---------------------------------------------------------------------------
# track_length_estimate(): fallback / quantile branches
# ---------------------------------------------------------------------------

def test_track_length_estimate_empty_returns_fallback():
    assert track_length_estimate([]) == COMPLETION_CAR_LENGTH_PRIOR


def test_track_length_estimate_below_min_frames_returns_fallback():
    vals = [3.5, 4.0, 4.2]  # 3 < COMPLETION_LENGTH_MIN_FRAMES (5)
    assert track_length_estimate(vals) == COMPLETION_CAR_LENGTH_PRIOR


def test_track_length_estimate_fallback_none_disables_push():
    assert track_length_estimate([], fallback=None) is None
    assert track_length_estimate([3.5, 4.0, 4.2], fallback=None) is None


def test_track_length_estimate_at_min_frames_uses_quantile():
    vals = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]  # 6 >= min_frames (5)
    expected = float(np.percentile(vals, COMPLETION_LENGTH_TRACK_QUANTILE)
                      + COMPLETION_LENGTH_TRACK_OFFSET)
    result = track_length_estimate(vals)
    assert result == pytest.approx(expected)


def test_track_length_estimate_ignores_none_entries():
    vals = [3.0, None, 3.5, 4.0, None, 4.5, 5.0, 5.5]
    filtered = [v for v in vals if v is not None]
    expected = float(np.percentile(filtered, COMPLETION_LENGTH_TRACK_QUANTILE)
                      + COMPLETION_LENGTH_TRACK_OFFSET)
    assert track_length_estimate(vals) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# complete(): order-independence under sample_seed (Finding #38)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(PCN_CHECKPOINT),
    reason=f"PCN checkpoint not found at {PCN_CHECKPOINT}",
)
def test_complete_order_independence_with_sample_seed():
    pts = _synthetic_partial(length=3.8, width=1.8, seed=1)

    fresh = PointCloudCompleter(model_path=PCN_CHECKPOINT, seed=0)
    out_fresh, skip_fresh = fresh.complete(pts, "car", sample_seed=42)

    warmed = PointCloudCompleter(model_path=PCN_CHECKPOINT, seed=0)
    for i in range(5):
        # Advance warmed._rng with unrelated completions (no sample_seed) so
        # its internal RNG state differs from a freshly constructed instance.
        warmed.complete(_synthetic_partial(length=3.8, width=1.8, seed=100 + i), "car")
    out_warmed, skip_warmed = warmed.complete(pts, "car", sample_seed=42)

    assert skip_fresh is None and skip_warmed is None
    np.testing.assert_allclose(out_fresh, out_warmed, atol=1e-5)
