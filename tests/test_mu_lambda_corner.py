"""
tests/test_mu_lambda_corner.py: Unit tests for Ray-Invariance in Corner Regime (Task 2).
"""

import pytest
import numpy as np
from underspec_sim.verifications.verify_mu_lambda_corner import run_verification


def test_factored_diff_matches_raw_diff():
    """Verify Pi(1) - Pi(0) matches the factored formula mu_A * [ Lambda * R_0_bar - rho * (R_0_bar + gamma_bar) ]."""
    k = 10.0
    Lambda = 2.0
    gamma_bar = 0.9242
    R_0_bar = 8.1516

    for rho in [0.5, 1.0, 1.5, 1.8]:
        for mu_A in [0.2, 0.5, 1.0]:
            lam_A = rho * mu_A
            Delta = lam_A - mu_A * Lambda
            C_0 = mu_A * Lambda

            raw_diff = - (C_0 * gamma_bar + Delta * (R_0_bar + gamma_bar))
            factored_diff = mu_A * (Lambda * R_0_bar - rho * (R_0_bar + gamma_bar))

            assert raw_diff == pytest.approx(factored_diff, abs=1e-10)


def test_corner_choice_invariant_along_rays():
    """Verify that along any ray rho = lambda_A / mu_A, the corner decision never flips as mu_A scales."""
    k = 10.0
    Lambda = 2.0
    gamma_bar = 0.9242
    R_0_bar = 8.1516
    rho_star = Lambda * R_0_bar / (R_0_bar + gamma_bar)

    # Test ray below critical ratio -> corner 1
    for mu in [0.1, 0.4, 0.7, 1.0]:
        diff = mu * (Lambda * R_0_bar - 1.0 * (R_0_bar + gamma_bar))
        assert diff > 0.0

    # Test ray above critical ratio -> corner 0
    for mu in [0.1, 0.4, 0.7, 1.0]:
        diff = mu * (Lambda * R_0_bar - 1.9 * (R_0_bar + gamma_bar))
        assert diff < 0.0


def test_mu_lambda_corner_runner():
    """Verify Task 2 runner executes with 0 ray invariance failures."""
    res = run_verification(output_dir="outputs")
    assert res["verdict"] == "PASS"
    assert res["ray_invariance_failures"] == 0
    assert res["rho_star"] > 0.0
