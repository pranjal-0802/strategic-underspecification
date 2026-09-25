r"""
tests/test_pooling.py: Direct algebraic tests of Model I pooling formulas (Section 5, Prop 4, Cor 3).
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import leader_payoff_per_type
from underspec_sim.core.best_response import user_best_response
from underspec_sim.model1_pooling.payoff import leader_payoff_pooling
from underspec_sim.model1_pooling.solver import solve_pooling_equilibrium


def test_pooling_quadratic_coefficients():
    r"""
    Verify exact quadratic coefficients of Pi(a) = const + L_bar * a + Q_bar * a^2
    derived from primitives:
      s = Lambda - c_Q
      b = lambda_A - c_Q
      Q_bar = (mu_A * s - 2 * b) * gamma_bar / 2
      L_bar = (mu_A * s - b) * R0_bar
      const = mu_A * (V - Lambda * k) + mu_A * Lambda^2 * E[1/kappa] / 2
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, mu_A=1.0, lambda_A=3.0, V=100.0)
    F = [0.4, 0.5]  # two points
    # Lambda = 5.0, c_Q = 2.0 => s = 3.0
    # lambda_A = 3.0 => b = 1.0
    # E[1/kappa] = (2.5 + 2.0)/2 = 2.25
    # R0_bar = 10 - 5 * 2.25 = -1.25
    # gamma_bar = 3.0 * 2.25 = 6.75
    # Q_bar = (1.0 * 3.0 - 2.0 * 1.0) * 6.75 / 2 = 1.0 * 3.375 = 3.375
    # L_bar = (1.0 * 3.0 - 1.0) * (-1.25) = -2.5
    # const = 1.0 * (100 - 50) + 1.0 * 25 * 2.25 / 2 = 50 + 28.125 = 78.125

    # At a = 0: Pi = 78.125
    assert leader_payoff_pooling(0.0, F, params=p) == pytest.approx(78.125)
    # At a = 1: Pi = 78.125 - 2.5 + 3.375 = 79.0
    assert leader_payoff_pooling(1.0, F, params=p) == pytest.approx(79.0)

    # Cross-check directly against primitives (leader_payoff_per_type)
    from underspec_sim.core.best_response import user_best_response
    for a_val in [0.0, 0.3, 0.7, 1.0]:
        expected_from_primitives = float(np.mean([
            leader_payoff_per_type(user_best_response(kap, a_val, p, clip=False), a_val, kap, p)
            for kap in F
        ]))
        assert leader_payoff_pooling(a_val, F, params=p) == pytest.approx(expected_from_primitives, abs=1e-12)


def test_prop4_closed_form_matches_exact_quadratic_argmax():
    r"""
    Proposition 4: For quadratic Pi(a) = const + L_bar * a + Q_bar * a^2 with Q_bar < 0,
    the unconstrained argmax is a* = - L_bar / (2 * Q_bar).
    """
    k = 5.0
    c_Q = 2.0
    Lambda = 4.0
    F_samples = np.linspace(0.25, 0.35, 100)
    inv_kap = float(np.mean(1.0 / F_samples))
    s = Lambda - c_Q
    gamma_b = s * inv_kap
    r0_b = k - Lambda * inv_kap

    # Target interior peak a* = 0.5:
    # - L_bar = Q_bar  <=>  - (s - b) * r0_b = (s - 2b) * gamma_b / 2
    # Solving for b: b = s * (r0_b + gamma_b/2) / (r0_b + gamma_b)
    b_target = s * (r0_b + 0.5 * gamma_b) / (r0_b + gamma_b)
    lam_target = c_Q + b_target

    p = ModelParams(k=k, g=0.5, L=8.0, c_Q=c_Q, mu_A=1.0, lambda_A=lam_target, V=100.0)

    res = solve_pooling_equilibrium(F_samples, params=p)
    assert res.is_concave is True
    assert res.matches is True
    assert res.a_SE_closed == pytest.approx(0.5, abs=1e-3)
    assert res.a_SE == pytest.approx(0.5, abs=1e-2)
    assert res.a_SE_grid == pytest.approx(0.5, abs=1e-2)
    assert res.regime == "interior"


def test_unbiased_pooling_always_corner_matching_formula():
    r"""
    Corollary 3 (Corrected): In the unbiased case (mu_A=1, lambda_A=c_Q => b=0),
    Pi(a) is weakly convex on [0, 1] because Q_bar = s^2 * E[1/kappa] / 2 >= 0 always.
    The solver must ALWAYS return regime: 'corner' with a_SE in {0, 1},
    matching Pi(1) - Pi(0) = (Lambda - c_Q) * [R0_bar + gamma_bar / 2]
                           = (Lambda - c_Q) * [k - (Lambda + c_Q)/2 * E[1/kappa]].
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

                    # 2. SOC must be weakly positive (convex)
                    assert res.soc_value >= -1e-9

                    # 3. Predict corner via Corollary 3 formula:
                    inv_kap = float(np.mean(1.0 / F))
                    R0_bar = k_val - lam * inv_kap
                    gamma_bar = (lam - cq) * inv_kap
                    pi_diff_formula = (lam - cq) * (R0_bar + 0.5 * gamma_bar)
                    expected_corner = 1.0 if pi_diff_formula >= 0.0 else 0.0

                    assert res.a_SE == expected_corner
                    assert res.a_SE == pytest.approx(res.a_SE_grid, abs=1e-2)


def test_cor2_derivative_sign_on_interior_branch():
    r"""
    Regression Test for Corollary 2 (Corrected):
    On the interior branch (Q_bar < 0),
      \partial a^{SE} / \partial \lambda_A = - (R0_bar / gamma_bar) * (mu_A * s) / (mu_A * s - 2b)^2 > 0
    strictly when R0_bar < 0. Tests that:
    1. The analytical derivative is strictly positive for all valid \lambda_A.
    2. Empirical finite-difference slopes \Delta a^{SE} / \Delta \lambda_A are strictly positive.
    3. Empirical slopes match the analytical formula within numerical tolerance.
    """
    from underspec_sim.model1_pooling.properties import test_bias_direction

    res = test_bias_direction()
    assert res.passed is True
    assert all(r == "interior" for r in res.regimes)
    assert all(s < 0.0 for s in res.soc_values)
    assert all(s > 0.0 for s in res.empirical_slopes)
    assert all(d > 0.0 for d in res.analytical_derivatives)
    assert res.max_derivative_error < 0.06


def test_ground_truth_regression_against_primitives():
    r"""
    P0 Regression Test: Verify leader_payoff_pooling against ground truth primitives
    (leader_payoff_per_type) to machine precision across diverse randomized parameter sets.
    """
    rng = np.random.default_rng(42)
    for _ in range(5):
        k = float(rng.uniform(4.0, 15.0))
        g = float(rng.uniform(0.2, 0.8))
        L = float(rng.uniform(5.0, 15.0))
        c_Q = float(rng.uniform(0.5, 3.0))
        mu_A = float(rng.uniform(0.3, 1.0))
        lambda_A = float(rng.uniform(0.5, 6.0))
        F_samples = rng.uniform(0.2, 0.8, size=150)

        p = ModelParams(k=k, g=g, L=L, c_Q=c_Q, mu_A=mu_A, lambda_A=lambda_A, V=100.0)

        for a_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fast_val = leader_payoff_pooling(a_val, F_samples, params=p)
            prim_val = float(np.mean([
                leader_payoff_per_type(user_best_response(kap, a_val, p, clip=False), a_val, kap, p)
                for kap in F_samples
            ]))
            assert abs(fast_val - prim_val) < 2e-13


def test_ground_truth_argmax_grid_search_against_primitives():
    r"""
    P0 Regression Test (Audit §2): Grid-search leader_payoff_per_type (primitives)
    averaged over F_samples as ground truth, and assert that solve_pooling_equilibrium
    (and the Proposition 4 closed-form formula) matches the empirical argmax.
    Tests:
    1. Unbiased benchmark (k=10, k* approx 9.61) -> argmax is 1.0 (always ask).
    2. Unbiased below-threshold (k=5, k* approx 9.61) -> argmax is 0.0 (never ask).
    3. Concave interior regime -> argmax is interior (0.500) matching closed form.
    """
    # 1. Unbiased benchmark (k=10)
    p_unb = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, mu_A=1.0, lambda_A=2.0, V=100.0)
    F_unb = np.linspace(0.2, 0.6, 200)
    a_grid = np.linspace(0.0, 1.0, 501)

    prim_vals_unb = [
        float(np.mean([leader_payoff_per_type(user_best_response(kap, a, p_unb, clip=False), a, kap, p_unb) for kap in F_unb]))
        for a in a_grid
    ]
    best_a_prim_unb = a_grid[np.argmax(prim_vals_unb)]
    res_unb = solve_pooling_equilibrium(F_unb, params=p_unb)
    assert best_a_prim_unb == 1.0
    assert res_unb.a_SE == 1.0

    # 2. Unbiased below-threshold (k=5)
    p_k5 = ModelParams(k=5.0, g=0.5, L=10.0, c_Q=2.0, mu_A=1.0, lambda_A=2.0, V=100.0)
    prim_vals_k5 = [
        float(np.mean([leader_payoff_per_type(user_best_response(kap, a, p_k5, clip=False), a, kap, p_k5) for kap in F_unb]))
        for a in a_grid
    ]
    best_a_prim_k5 = a_grid[np.argmax(prim_vals_k5)]
    res_k5 = solve_pooling_equilibrium(F_unb, params=p_k5)
    assert best_a_prim_k5 == 0.0
    assert res_k5.a_SE == 0.0

    # 3. Concave interior regime targeting a* = 0.500
    k = 5.0
    c_Q = 2.0
    Lambda = 4.0
    F_int = np.linspace(0.25, 0.35, 200)
    inv_kap = float(np.mean(1.0 / F_int))
    s = Lambda - c_Q
    gamma_b = s * inv_kap
    r0_b = k - Lambda * inv_kap
    b_target = s * (r0_b + 0.5 * gamma_b) / (r0_b + gamma_b)
    lam_target = float(c_Q + b_target)
    p_int = ModelParams(k=k, g=0.5, L=8.0, c_Q=c_Q, mu_A=1.0, lambda_A=lam_target, V=100.0)

    prim_vals_int = [
        float(np.mean([leader_payoff_per_type(user_best_response(kap, a, p_int, clip=False), a, kap, p_int) for kap in F_int]))
        for a in a_grid
    ]
    best_a_prim_int = a_grid[np.argmax(prim_vals_int)]
    res_int = solve_pooling_equilibrium(F_int, params=p_int)
    assert res_int.regime == "interior"
    assert res_int.is_concave is True
    assert abs(best_a_prim_int - 0.5) <= 0.005
    assert abs(res_int.a_SE - 0.5) <= 0.005
    assert abs(res_int.a_SE_closed - 0.5) <= 0.005
    assert abs(best_a_prim_int - res_int.a_SE) <= 0.005

