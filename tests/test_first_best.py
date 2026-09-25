"""
tests/test_first_best.py: Direct algebraic tests of Proposition 3 first-best formulas.
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.first_best import first_best, biased_first_best
from underspec_sim.core.payoffs import user_utility


def test_first_best_threshold_formula():
    r"""
    Proposition 3: kappa* = (Lambda + c_Q) / (2k).
    If kappa > kappa*: a_FB = 1, m_FB = c_Q / kappa
    If kappa < kappa*: a_FB = 0, m_FB = Lambda / kappa
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    # Lambda = 5.0, c_Q = 2.0, k = 10.0 => kappa* = 7 / 20 = 0.35
    assert p.kappa_star == 0.35

    # Below threshold: kappa = 0.2 < 0.35
    m_fb_low, a_fb_low = first_best(0.2, p, clip=False)
    assert a_fb_low == 0.0
    assert m_fb_low == pytest.approx(5.0 / 0.2)  # 25.0

    # Above threshold: kappa = 0.5 > 0.35
    m_fb_high, a_fb_high = first_best(0.5, p, clip=False)
    assert a_fb_high == 1.0
    assert m_fb_high == pytest.approx(2.0 / 0.5)  # 4.0


def test_first_best_payoff_difference_formula():
    r"""
    Proof of Proposition 3:
    U(a=1) - U(a=0) = (Lambda - c_Q) * (k - (Lambda + c_Q)/(2 * kappa))
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    Lambda = p.Lambda  # 5.0
    c_Q = p.c_Q        # 2.0
    k = p.k            # 10.0

    for kap in [0.2, 0.35, 0.5, 0.8]:
        m1 = c_Q / kap
        m0 = Lambda / kap
        u1 = user_utility(m1, 1.0, kap, p)
        u0 = user_utility(m0, 0.0, kap, p)
        actual_diff = u1 - u0
        predicted_diff = (Lambda - c_Q) * (k - (Lambda + c_Q) / (2.0 * kap))
        assert actual_diff == pytest.approx(predicted_diff, rel=1e-7)
