"""
tests/test_mu_lambda_corner.py: Unit tests for Ray-Invariance in Corner Regime (Task 2).
"""

import pytest
import numpy as np
from underspec_sim.verifications.verify_mu_lambda_corner import run_verification


def test_factored_diff_matches_raw_diff():
    """Verify Pi(1) - Pi(0) matches the factored formula mu_A * [ s * (R_0_bar + gamma_bar/2) - rho * (R_0_bar + gamma_bar) ]."""
    k = 10.0
    Lambda = 2.0
    c_Q = 1.0
    s = Lambda - c_Q
    gamma_bar = 0.9242
    R_0_bar = 8.1516

    for rho in [0.5, 0.8, 1.0, 1.5]:
        for mu_A in [0.2, 0.5, 1.0]:
            b = rho * mu_A
            lam_A = c_Q + b

            Q_bar = (mu_A * s - 2.0 * b) * gamma_bar / 2.0
            L_bar = (mu_A * s - b) * R_0_bar

            raw_diff = L_bar + Q_bar
            factored_diff = mu_A * (s * (R_0_bar + 0.5 * gamma_bar) - rho * (R_0_bar + gamma_bar))

            assert raw_diff == pytest.approx(factored_diff, abs=1e-10)


def test_corner_choice_invariant_along_rays():
    """Verify that along any ray rho = (lambda_A - c_Q) / mu_A, the corner decision never flips as mu_A scales."""
    k = 10.0
    Lambda = 2.0
    c_Q = 1.0
    s = Lambda - c_Q
    gamma_bar = 0.9242
    R_0_bar = 8.1516
    rho_star = s * (R_0_bar + 0.5 * gamma_bar) / (R_0_bar + gamma_bar)

    # Test ray below critical ratio -> corner 1
    for mu in [0.1, 0.4, 0.7, 1.0]:
        diff = mu * (s * (R_0_bar + 0.5 * gamma_bar) - 0.5 * (R_0_bar + gamma_bar))
        assert diff > 0.0

    # Test ray above critical ratio -> corner 0
    for mu in [0.1, 0.4, 0.7, 1.0]:
        diff = mu * (s * (R_0_bar + 0.5 * gamma_bar) - 1.5 * (R_0_bar + gamma_bar))
        assert diff < 0.0


def test_mu_lambda_corner_runner():
    """Verify Task 2 runner executes with 0 ray invariance failures."""
    res = run_verification(output_dir="outputs")
    assert res["verdict"] == "PASS"
    assert res["ray_invariance_failures"] == 0
    assert res["rho_star"] > 0.0
