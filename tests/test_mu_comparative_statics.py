"""
tests/test_mu_comparative_statics.py: Unit tests for mu_A comparative statics (Task 3).
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.verifications.verify_mu_comparative_statics import (
    compute_pooling_derivatives,
    run_verification,
)


def test_derivative_ratio_matches_analytical_formula():
    """Verify that (d a_SE / d mu_A) / (d a_SE / d lambda_A) == - lambda_A / mu_A."""
    k = 5.0
    c_Q = 2.0
    Lambda = 4.0
    F_samples = np.linspace(0.25, 0.35, 100)
    inv_kap = float(np.mean(1.0 / F_samples))
    gamma_b = (Lambda - c_Q) * inv_kap
    r0_b = k - Lambda * inv_kap
    c0 = Lambda
    delta_target = -c0 * gamma_b / (r0_b + gamma_b)
    lam_anchor = c0 + delta_target

    p = ModelParams(k=k, g=0.5, L=8.0, c_Q=c_Q, mu_A=1.0, lambda_A=lam_anchor, V=100.0)

    for mu in [0.6, 0.8, 1.0]:
        da_dmu, da_dlam, ratio = compute_pooling_derivatives(F_samples, mu, lam_anchor, c_Q, p)
        expected_ratio = - lam_anchor / mu
        assert ratio == pytest.approx(expected_ratio, rel=1e-3)
        # Check opposite signs
        assert (da_dmu * da_dlam) < 0.0


def test_mu_comparative_statics_runner():
    """Verify runner executes and passes."""
    res = run_verification(output_dir="outputs")
    assert res["verdict"] == "PASS"
    assert res["passed"] is True
    assert res["max_ratio_error"] < 1e-2
