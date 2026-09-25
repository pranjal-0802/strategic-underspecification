"""
tests/test_pooling.py: Direct algebraic tests of Model I pooling formulas (Section 5, Prop 4).
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.model1_pooling.payoff import leader_payoff_pooling
from underspec_sim.model1_pooling.solver import solve_pooling_equilibrium


def test_pooling_quadratic_coefficients():
    r"""
    Verify coefficients of Pi(a) = mu*V - C0*R0_bar - a*(C0*gamma_bar + Delta*R0_bar) - a^2 * Delta*gamma_bar
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, mu_A=1.0, lambda_A=3.0, V=100.0)
    F = [0.4, 0.5]  # two points
    # Lambda = 5.0, c_Q = 2.0
    # beta(0.4) = 5/0.4 = 12.5; beta(0.5) = 5/0.5 = 10.0 => mean beta = 11.25
    # gamma(0.4) = 3/0.4 = 7.5; gamma(0.5) = 3/0.5 = 6.0 => mean gamma = 6.75
    # R0_bar = 10 - 11.25 = -1.25
    # C0 = 1.0 * 5.0 = 5.0
    # Delta = 3.0 - 5.0 = -2.0
    # linear coeff: C0 * gamma_bar + Delta * R0_bar = 5 * 6.75 + (-2) * (-1.25) = 33.75 + 2.5 = 36.25
    # quad coeff: Delta * gamma_bar = -2.0 * 6.75 = -13.5
    # constant: mu*V - C0*R0_bar = 100 - 5 * (-1.25) = 100 + 6.25 = 106.25

    # At a = 0: Pi = 106.25
    assert leader_payoff_pooling(0.0, F, params=p) == pytest.approx(106.25)
    # At a = 1: Pi = 106.25 - 36.25 - (-13.5) = 70.0 + 13.5 = 83.5
    assert leader_payoff_pooling(1.0, F, params=p) == pytest.approx(83.5)


def test_prop4_closed_form_matches_exact_quadratic_argmax():
    r"""
    Proposition 4: For any quadratic Pi(a) = - Q*a^2 - B*a + C with Q > 0,
    the unconstrained argmax is a* = - B / (2Q).
    """
    # Use parameters where Q = Delta * gamma_bar > 0
    k = 5.0
    c_Q = 2.0
    Lambda = 4.0
    F_samples = np.linspace(0.25, 0.35, 100)
    inv_kap = np.mean(1.0 / F_samples)
    gamma_b = (Lambda - c_Q) * inv_kap
    r0_b = k - Lambda * inv_kap
    c0 = Lambda
    delta_target = - c0 * gamma_b / (r0_b + gamma_b)
    p = ModelParams(k=k, g=0.5, L=8.0, c_Q=c_Q, mu_A=1.0, lambda_A=c0 + delta_target, V=100.0)

    res = solve_pooling_equilibrium(F_samples, params=p)
    assert res.is_concave is True
    assert res.matches is True
    assert res.a_SE_closed == pytest.approx(0.5, abs=1e-3)
    assert res.a_SE_grid == pytest.approx(0.5, abs=1e-2)
