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
    """Proposition 6: Downward distortion of high-cost type under under-asking bias."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=4.0)
    res = check_distortion_under_bias(kappa_L=0.3, kappa_H=0.5, lambda_A=4.0, params=p, unconstrained_m=False)
    assert res.passed is True
    assert res.a_L_at_corner is True
    assert res.a_H_strictly_below_biased_fb is True
    assert "IC_L" in res.active_constraints
