r"""
tests/test_best_response.py: Direct algebraic tests of best-response formulas (Section 5.1).
m^*(\kappa; a) = \beta(\kappa) - a * \gamma(\kappa)
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.best_response import (
    beta_func,
    gamma_func,
    user_best_response,
    user_best_response_with_diagnostics,
)


def test_beta_and_gamma_algebra():
    r"""\beta = \Lambda / \kappa, \gamma = (\Lambda - c_Q) / \kappa."""
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0)  # Lambda = 5.0
    kappa = 0.5
    assert beta_func(kappa, p) == pytest.approx(10.0)
    assert gamma_func(kappa, p) == pytest.approx(6.0)


def test_best_response_unconstrained():
    r"""
    m*(kappa; a) = (Lambda*(1-a) + a*c_Q) / kappa
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0)  # Lambda = 5.0
    kappa = 0.5
    # a = 0 => m* = 5.0 / 0.5 = 10.0
    assert user_best_response(kappa, a=0.0, params=p, clip=False) == pytest.approx(10.0)
    # a = 1 => m* = 2.0 / 0.5 = 4.0
    assert user_best_response(kappa, a=1.0, params=p, clip=False) == pytest.approx(4.0)
    # a = 0.5 => m* = (5*0.5 + 2*0.5) / 0.5 = 3.5 / 0.5 = 7.0
    assert user_best_response(kappa, a=0.5, params=p, clip=False) == pytest.approx(7.0)


def test_clipping_bounds():
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0)
    # For small kappa = 0.1, unconstrained m*(a=0) = 5.0 / 0.1 = 50.0 > 10.0
    m_clip, m_uncon, is_clipped = user_best_response_with_diagnostics(kappa=0.1, a=0.0, params=p)
    assert m_uncon == pytest.approx(50.0)
    assert m_clip == pytest.approx(10.0)
    assert is_clipped is True

    # Vectorized inputs
    kappas = np.array([0.1, 0.5, 1.0])
    m_vec = user_best_response(kappas, a=0.0, params=p, clip=True)
    # 5.0/0.1=50->10; 5.0/0.5=10; 5.0/1.0=5
    np.testing.assert_allclose(m_vec, [10.0, 10.0, 5.0])
