"""
tests/test_screening.py: Tests for Model II screening menu optimization (Section 6, Props 5 & 6).
"""

import pytest
from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import user_utility
from underspec_sim.model2_screening.solver import solve_menu
from underspec_sim.model2_screening.properties import (
    test_no_distortion_without_bias as check_no_distortion_without_bias,
    test_distortion_under_bias as check_distortion_under_bias,
)


def test_single_crossing_rent_formula():
    r"""
    Proof sketch Proposition 6:
    U(m_H, a_H; kappa_L) - U(m_H, a_H; kappa_H) = 0.5 * (kappa_H - kappa_L) * m_H^2
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    kappa_L = 0.3
    kappa_H = 0.6
    m_H = 4.0
    a_H = 0.8

    u_L = user_utility(m_H, a_H, kappa_L, p)
    u_H = user_utility(m_H, a_H, kappa_H, p)
    actual_rent = u_L - u_H
    predicted_rent = 0.5 * (kappa_H - kappa_L) * (m_H ** 2)
    assert actual_rent == pytest.approx(predicted_rent, rel=1e-7)


def test_prop5_unbiased_menu():
    """Proposition 5: Unbiased leader recovers First Best and IC is slack."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    res = check_no_distortion_without_bias(kappa_L=0.3, kappa_H=0.5, params=p, unconstrained_m=True)
    assert res.passed is True
    assert res.IC_L_slack > 0.0
    assert res.IC_H_slack > 0.0


def test_prop6_biased_menu_distortion():
    """Proposition 6/7: Downward distortion of high-cost type under under-asking bias."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=4.0)
    res = check_distortion_under_bias(kappa_L=0.3, kappa_H=0.5, lambda_A=4.0, params=p, unconstrained_m=False)
    assert res.passed is True
    assert res.a_L_at_corner is True
    assert res.a_H_strictly_below_biased_fb is True
    assert "IC_L" in res.active_constraints


def test_prop7_boundary_saturation_narrow_heterogeneity():
    """Proposition 7 Claim 4: When m_H^* > k (narrow heterogeneity), menu collapses to boundary pooling at (k, 0)."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=3.0)
    res = solve_menu(kappa_L=0.26, kappa_H=0.31, f_L=0.5, f_H=0.5, mu_A=1.0, lambda_A=3.0, c_Q=2.0, params=p, unconstrained_m=False)
    assert res.m_L == pytest.approx(10.0, abs=1e-5)
    assert res.m_H == pytest.approx(10.0, abs=1e-5)
    assert res.a_L == pytest.approx(0.0, abs=1e-5)
    assert res.a_H == pytest.approx(0.0, abs=1e-5)


def test_prop7_effort_distortion_regime():
    """Proposition 7 Claim 2: Intermediate complexity k_slack < k <= k_crit yields pure effort distortion (a_H = 1, m_H < m_H^B)."""
    p = ModelParams(k=7.5, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=3.0)
    # k_slack = 4 / 0.3 - 3 / 0.5 = 7.333
    # m_H^* = 5 / (3 * 0.2 + 0.3) = 5.556
    # k_crit = 4 / 0.3 - 5.556 = 7.778
    # k = 7.5 lies strictly in (k_slack, k_crit]
    res = solve_menu(kappa_L=0.3, kappa_H=0.5, f_L=0.5, f_H=0.5, mu_A=1.0, lambda_A=3.0, c_Q=2.0, params=p, unconstrained_m=False)
    assert "IC_L" in res.active_constraints
    assert res.a_H == pytest.approx(1.0, abs=1e-4)
    expected_m_H = 2.0 * 2.0 / 0.3 - 7.5  # 5.8333...
    assert res.m_H == pytest.approx(expected_m_H, abs=1e-4)
    # Compare with unconstrained biased benchmark m_H^B = c_Q^eff / kappa_H = 3.0 / 0.5 = 6.0
    assert res.m_H < 6.0

