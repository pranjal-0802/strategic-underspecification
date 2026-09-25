"""
tests/test_exact_conjunctive.py: Unit tests for exact conjunctive payoffs (Task 1).
"""

import pytest
import numpy as np
from underspec_sim.verifications.verify_exact_conjunctive import (
    exact_q,
    exact_user_utility,
    exact_user_best_response,
    solve_exact_pooling_optimum,
    solve_exact_screening_menu,
)


def test_exact_user_utility_algebra():
    """Checks basic properties of exact conjunctive utility."""
    V = 100.0
    c_Q = 2.0
    kappa = 0.5
    g = 0.5
    k = 10.0

    # At m = k: q^{k-m} = q^0 = 1, clarify cost = 0 => U = V - 0.5 * kappa * k^2
    u_full = exact_user_utility(m=k, a=0.5, kappa=kappa, g=g, k=k, c_Q=c_Q, V=V)
    expected_full = V - 0.5 * kappa * (k ** 2)
    assert u_full == pytest.approx(expected_full)

    # At a = 1: q = 1, so success prob = 1 regardless of m
    u_a1 = exact_user_utility(m=5.0, a=1.0, kappa=kappa, g=g, k=k, c_Q=c_Q, V=V)
    expected_a1 = V - 0.5 * kappa * (5.0 ** 2) - 1.0 * (k - 5.0) * c_Q
    assert u_a1 == pytest.approx(expected_a1)


def test_exact_best_response_bounds():
    """Checks that user best response under exact conjunctive payoff lands within [0, k]."""
    V = 100.0
    c_Q = 2.0
    k = 10.0
    g = 0.6

    for kappa in [0.2, 0.5, 1.0]:
        for a in [0.0, 0.5, 1.0]:
            m_star = exact_user_best_response(kappa=kappa, a=a, g=g, k=k, c_Q=c_Q, V=V)
            assert 0.0 <= m_star <= k


def test_unbiased_pooling_corner_solution_under_exact_form():
    """Verifies that unbiased pooling optimum lands at a corner {0, 1} for exact conjunctive form."""
    F_samples = np.linspace(0.2, 0.6, 20)
    c_Q = 2.0
    V = 100.0

    # For k=5 and g=0.7
    best_a, _, regime, _, _ = solve_exact_pooling_optimum(
        F_samples=F_samples,
        g=0.7,
        k=5,
        c_Q=c_Q,
        V=V,
        mu_A=1.0,
        lambda_A=c_Q,
    )
    assert regime in ("corner_0", "corner_1")
    assert best_a in (0.0, 1.0)


def test_exact_screening_downward_distortion():
    """Verifies downward distortion and slack IR_H under exact conjunctive screening."""
    res = solve_exact_screening_menu(
        kappa_L=0.3,
        kappa_H=0.5,
        g=0.7,
        k=10,
        c_Q=2.0,
        V=100.0,
        mu_A=1.0,
        lambda_A=4.0,
    )
    assert res["downward_distortion_holds"] is True
    assert res["a_H"] <= res["a_B_H"]
    assert res["IR_H_slack"] > 0.0
