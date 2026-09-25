"""
tests/test_monotonicity_survival.py: Unit tests for Monotonicity Survival (Task 1).
"""

import pytest
import numpy as np
from underspec_sim.verifications.verify_monotonicity_survival import (
    exact_user_utility,
    solve_optimal_m,
    run_verification,
)


def test_solve_optimal_m_decreases_with_q_outside_assumption1():
    """Verify that when (k - m)*ln(q) >= -1 (outside Ass1), m* is non-increasing in q."""
    k = 10.0
    kappa = 1.0
    # At q=0.85, 0.90, 0.95, user cost of effort outweighs guessing loss as guessing improves
    m_85 = solve_optimal_m(k=k, q=0.85, kappa=kappa)
    m_90 = solve_optimal_m(k=k, q=0.90, kappa=kappa)
    m_95 = solve_optimal_m(k=k, q=0.95, kappa=kappa)

    assert m_90 <= m_85 + 1e-3
    assert m_95 <= m_90 + 1e-3


def test_cross_partial_mathematical_sign():
    """
    Verify the sign of d^2 U / (dm dq):
    d^2 U / (dm dq) = - V * q^{k-m-1} * [1 + (k - m) ln(q)].
    When (k - m) ln(q) > -1, cross-partial is strictly negative (decreasing differences).
    When (k - m) ln(q) < -1, cross-partial is strictly positive (increasing differences).
    """
    V = 100.0
    q = 0.8
    k = 10.0

    # Case A: m near k, so (k - m)*ln(q) is near 0 > -1
    m_A = 9.0
    lhs_A = (k - m_A) * np.log(q)
    cross_partial_A = -V * (q ** (k - m_A - 1)) * (1.0 + lhs_A)
    assert lhs_A > -1.0
    assert cross_partial_A < 0.0  # Decreasing differences!

    # Case B: m small, so (k - m)*ln(q) < -1
    m_B = 1.0
    lhs_B = (k - m_B) * np.log(q)
    cross_partial_B = -V * (q ** (k - m_B - 1)) * (1.0 + lhs_B)
    assert lhs_B < -1.0
    assert cross_partial_B > 0.0  # Increasing differences!


def test_proposition2_naive_welfare_weakly_worse():
    """Verify naive users (who assume a^dagger > a_true) are weakly worse off than sophisticated users."""
    k = 10.0
    kappa = 0.80
    q_true = 0.80
    a_true = 0.0
    a_dagger = 0.50

    q_naive = a_dagger + (1.0 - a_dagger) * q_true
    m_soph = solve_optimal_m(k=k, q=q_true, kappa=kappa, a=a_true)
    m_naive = solve_optimal_m(k=k, q=q_naive, kappa=kappa, a=a_dagger)

    u_soph = exact_user_utility(m_soph, a=a_true, kappa=kappa, g=q_true, k=k)
    u_naive = exact_user_utility(m_naive, a=a_true, kappa=kappa, g=q_true, k=k)

    assert u_naive <= u_soph + 1e-5


def test_monotonicity_survival_runner():
    """Verify Task 1 runner executes and reports 100% survival outside Assumption 1."""
    res = run_verification(output_dir="outputs")
    assert res["verdict"] in ("PASS", "CAUTION")
    assert res["ass1_fails_mono_fails"] == 0
    assert res["pct_ass1_fails_mono_survives"] == pytest.approx(100.0, rel=1e-3)
