"""
tests/test_assumption1.py: Unit tests for Assumption 1 regularity check (Task 2).
"""

import pytest
import numpy as np
from underspec_sim.verifications.verify_assumption1 import (
    solve_optimal_m,
    run_verification,
)


def test_solve_optimal_m_bounds():
    """Verify that optimal specification m* stays within [0, k]."""
    for k in [5.0, 10.0, 15.0]:
        for q in [0.70, 0.85, 0.95]:
            m = solve_optimal_m(k=k, q=q, kappa=0.5, V=100.0)
            assert 0.0 <= m <= k


def test_assumption1_fails_for_low_cost_users():
    """When kappa is low, user specifies nearly all attributes (m* -> k), so (k - m*)*ln(q) > -1."""
    k = 10.0
    q = 0.85
    kappa = 0.20  # Low specification cost
    m = solve_optimal_m(k=k, q=q, kappa=kappa, V=100.0)
    lhs = (k - m) * np.log(q)
    # Since m* is close to k, lhs is near 0 > -1
    assert lhs > -1.0


def test_assumption1_holds_for_high_cost_and_noisy_guessing():
    """When kappa is high and guessing is noisy, user leaves enough attributes unspecified that (k - m*)*ln(q) <= -1."""
    k = 15.0
    q = 0.70
    kappa = 3.0  # High specification cost
    m = solve_optimal_m(k=k, q=q, kappa=kappa, V=100.0)
    lhs = (k - m) * np.log(q)
    assert lhs <= -1.0


def test_assumption1_canonical_grid_exact_fraction():
    """Regression test locking in exact canonical v4 audit results (653/726 = 89.94% corrected, 73/726 = 10.06% legacy)."""
    res = run_verification(output_dir="outputs")
    assert res["verdict"] in ("PASS", "CAUTION")
    assert res["total_points"] == 726
    assert res["holds_count"] == 653
    assert abs(res["holds_fraction"] - 653.0 / 726.0) < 1e-6
    assert res["legacy_holds_count"] == 73
    assert abs(res["legacy_holds_fraction"] - 73.0 / 726.0) < 1e-6
