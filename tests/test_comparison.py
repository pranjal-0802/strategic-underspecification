"""
tests/test_comparison.py: Tests for Section 7 regime comparison (dominance of menus).
"""

import pytest
from underspec_sim.core.params import ModelParams
from underspec_sim.comparison.compare import compare_regimes


def test_menus_dominate_pooling():
    """Corollary 4: Menus weakly dominate pooling for any parameter setting."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)

    # Test unbiased
    comp_unbiased = compare_regimes((0.3, 0.5, 0.5, 0.5), mu_A=1.0, lambda_A=2.0, c_Q=2.0, params=p)
    assert comp_unbiased.dominance_holds is True
    assert comp_unbiased.payoff_gap >= -1e-4

    # Test biased
    comp_biased = compare_regimes((0.3, 0.5, 0.5, 0.5), mu_A=1.0, lambda_A=4.0, c_Q=2.0, params=p)
    assert comp_biased.dominance_holds is True
    assert comp_biased.payoff_gap >= -1e-4
