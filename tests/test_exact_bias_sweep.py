"""
tests/test_exact_bias_sweep.py: Unit tests for Exact Bias Sweep & Constraint Binding (Task 3).
"""

import pytest
from underspec_sim.verifications.verify_exact_bias_sweep import (
    solve_exact_screening_menu_point,
    run_verification,
)


def test_solve_exact_screening_unbiased_point():
    """Verify that under no bias (mu_A=1.0, lambda_A=c_Q=2.0), no IC/IR constraints bind."""
    pt = solve_exact_screening_menu_point(mu_A=1.0, lambda_A=2.0)
    assert pt["IC_L_slack"] > 0.1
    assert pt["IC_H_slack"] > 0.1
    assert pt["active_constraints"] == "None"


def test_solve_exact_screening_moderate_bias():
    """Verify that under moderate bias (mu_A=1.0, lambda_A=4.0), IC_L binds alone."""
    pt = solve_exact_screening_menu_point(mu_A=1.0, lambda_A=4.0)
    assert pt["IC_L_active"] is True
    assert pt["IC_H_active"] is False
    assert "IC_L" in pt["active_constraints"]


def test_solve_exact_screening_extreme_bias():
    """Verify that under extreme bias (mu_A=1.0, lambda_A=12.0), IC_H also binds."""
    pt = solve_exact_screening_menu_point(mu_A=1.0, lambda_A=12.0)
    assert pt["IC_H_active"] is True
    assert "IC_H" in pt["active_constraints"]


def test_exact_bias_sweep_runner():
    """Verify Task 3 runner executes and confirms IC_H binding under extreme bias."""
    res = run_verification(output_dir="outputs")
    assert res["verdict"] == "PASS"
    assert res["total_points"] == 100
    assert res["both_ic_count"] > 0  # Confirms IC_H binds under extreme bias
