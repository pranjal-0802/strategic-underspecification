"""
tests/test_primitives.py: Direct algebraic tests of core symbols and payoffs against .tex formulas.
No optimization or numerical solvers, just plugging numbers directly into equations.
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import (
    resolution_probability,
    user_utility,
    user_utility_exact_conjunctive,
    leader_payoff_per_type,
)


def test_model_params_primitives():
    """Verify default values and computed properties."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    assert p.k == 10.0
    assert p.g == 0.5
    assert p.L == 10.0
    assert p.c_Q == 2.0
    assert p.V == 100.0

    # Lambda = L * (1 - g) = 10 * 0.5 = 5.0
    assert p.Lambda == 5.0

    # kappa* = (Lambda + c_Q) / (2k) = (5 + 2) / 20 = 0.35
    assert p.kappa_star == 0.35

    # Unbiased check
    assert p.is_unbiased is True

    # Biased copy
    p_bias = p.with_bias(lambda_A=4.0)
    assert p_bias.lambda_A == 4.0
    assert p_bias.is_unbiased is False
    # kappa^{*B} = (Lambda + lambda_A) / (2k) = (5 + 4) / 20 = 0.45
    assert p_bias.kappa_star_biased == 0.45


def test_resolution_probability():
    r"""q(a, g) = a + (1 - a)g."""
    assert resolution_probability(0.0, 0.5) == 0.5
    assert resolution_probability(1.0, 0.5) == 1.0
    assert resolution_probability(0.5, 0.5) == 0.75


def test_user_utility_algebra():
    r"""
    eq (210): U(m, a; kappa) = V - Lambda*(1-a)*(k-m) - 0.5*kappa*m^2 - a*c_Q*(k-m)
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    kappa = 0.4

    # Case 1: m=0, a=0
    # U = 100 - 5*(1)*(10) - 0 - 0 = 50.0
    u1 = user_utility(m=0.0, a=0.0, kappa=kappa, params=p)
    assert u1 == pytest.approx(50.0)

    # Case 2: m=10, a=0
    # U = 100 - 0 - 0.5*0.4*100 - 0 = 100 - 20 = 80.0
    u2 = user_utility(m=10.0, a=0.0, kappa=kappa, params=p)
    assert u2 == pytest.approx(80.0)

    # Case 3: m=0, a=1
    # U = 100 - 0 - 0 - 1*2*(10) = 80.0
    u3 = user_utility(m=0.0, a=1.0, kappa=kappa, params=p)
    assert u3 == pytest.approx(80.0)

    # Case 4: m=4, a=0.5
    # k-m = 6
    # loss_guess = 5 * 0.5 * 6 = 15.0
    # effort = 0.5 * 0.4 * 16 = 3.2
    # loss_ask = 0.5 * 2.0 * 6 = 6.0
    # U = 100 - 15 - 3.2 - 6 = 75.8
    u4 = user_utility(m=4.0, a=0.5, kappa=kappa, params=p)
    assert u4 == pytest.approx(75.8)


def test_leader_payoff_per_type_algebra():
    r"""
    eq (179): Pi_kappa(m, a) = mu_A * U - (lambda_A - c_Q)*a*(k-m)
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=2.0)
    kappa = 0.5
    # Unbiased: lambda_A = c_Q, so Pi == U
    u = user_utility(5.0, 0.5, kappa, p)
    pi = leader_payoff_per_type(5.0, 0.5, kappa, p)
    assert pi == pytest.approx(u)

    # Biased: lambda_A = 4.0 > c_Q = 2.0
    p_bias = p.with_bias(lambda_A=4.0)
    # Pi = 1.0 * u - (4.0 - 2.0) * 0.5 * (10 - 5) = u - 2.0 * 0.5 * 5 = u - 5.0
    pi_bias = leader_payoff_per_type(5.0, 0.5, kappa, p_bias)
    assert pi_bias == pytest.approx(u - 5.0)
