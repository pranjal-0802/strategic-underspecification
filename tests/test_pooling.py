r"""
tests/test_pooling.py: Direct algebraic tests of Model I pooling formulas (Section 5, Prop 4, Cor 3).
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
    assert res.a_SE == pytest.approx(0.5, abs=1e-2)
    assert res.a_SE_grid == pytest.approx(0.5, abs=1e-2)
    assert res.regime == "interior"


def test_unbiased_pooling_always_corner_matching_formula():
    r"""
    Corollary 3 (Corrected): In the unbiased case (mu_A=1, lambda_A=c_Q),
    Pi(a) is weakly convex on [0, 1] because Delta*gamma_bar <= 0 always.
    The solver must ALWAYS return regime: 'corner' with a_SE in {0, 1},
    matching Pi(1) - Pi(0) = (Lambda - c_Q)*(R0_bar - c_Q*E[1/kappa]).
    Tested across a grid of (Lambda, c_Q, k, and F distributions).
    """
    Lambda_values = [3.0, 5.0, 7.0]
    c_Q_values = [1.5, 2.0, 4.0]
    k_values = [5.0, 10.0, 15.0]
    distributions = [
        np.linspace(0.2, 0.6, 200),
        np.linspace(0.1, 0.3, 200),
        np.linspace(0.4, 0.9, 200),
    ]

    for lam in Lambda_values:
        for cq in c_Q_values:
            if abs(lam - cq) < 1e-4:
                continue
            for k_val in k_values:
                for F in distributions:
                    # Construct params: L = lam / (1 - g), choose g = 0.5 => L = 2 * lam
                    p = ModelParams(
                        k=k_val,
                        g=0.5,
                        L=2.0 * lam,
                        c_Q=cq,
                        mu_A=1.0,
                        lambda_A=cq,  # Unbiased
                        V=100.0,
                    )
                    assert p.Lambda == pytest.approx(lam)
                    assert p.is_unbiased is True

                    res = solve_pooling_equilibrium(F, params=p)

                    # 1. Must be corner regime
                    assert res.regime == "corner"
                    assert res.a_SE in (0.0, 1.0)

                    # 2. SOC must be non-positive
                    assert res.soc_value <= 1e-9

                    # 3. Predict corner via Corollary 3 formula:
                    inv_kap = float(np.mean(1.0 / F))
                    R0_bar = k_val - lam * inv_kap
                    pi_diff_formula = (lam - cq) * (R0_bar - cq * inv_kap)
                    expected_corner = 1.0 if pi_diff_formula >= 0.0 else 0.0

                    assert res.a_SE == expected_corner
                    assert res.a_SE == pytest.approx(res.a_SE_grid, abs=1e-2)


def test_cor2_derivative_sign_on_interior_branch():
    r"""
    Regression Test for Corollary 2 (Corrected in v4):
    On the interior branch (\Delta \bar\gamma > 0),
      \partial a^{SE} / \partial \lambda_A = \mu_A \Lambda / [2 (\mu_A \Lambda - \lambda_A)^2] > 0
    strictly. Tests that:
    1. The analytical derivative is strictly positive for all valid \lambda_A.
    2. Empirical finite-difference slopes \Delta a^{SE} / \Delta \lambda_A are strictly positive.
    3. Empirical slopes match the analytical formula within numerical tolerance.
    """
    from underspec_sim.model1_pooling.properties import test_bias_direction

    res = test_bias_direction()
    assert res.passed is True
    assert all(r == "interior" for r in res.regimes)
    assert all(s > 0.0 for s in res.soc_values)
    assert all(s > 0.0 for s in res.empirical_slopes)
    assert all(d > 0.0 for d in res.analytical_derivatives)
    assert res.max_derivative_error < 0.06
