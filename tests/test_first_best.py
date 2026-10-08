"""
tests/test_first_best.py: Tests of First-Best Benchmark under feasible domain m in [0, k]
and exact conjunctive specification.
"""

import pytest
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.first_best import first_best, first_best_exact
from underspec_sim.core.payoffs import user_utility


def test_first_best_feasible_domain_dominance():
    r"""
    On the feasible domain m in [0, k], for any Lambda > c_Q,
    dU/da = (Lambda - c_Q)(k - m) >= 0 everywhere, so a_FB = 1
    weakly dominates guessing for all types kappa.
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    assert p.Lambda == 5.0
    assert p.c_Q == 2.0

    kappas = [0.1, 0.2, 0.3, 0.35, 0.5, 0.8]
    for kap in kappas:
        m_fb, a_fb = first_best(kap, p, clip=True)
        assert a_fb == 1.0
        assert 0.0 <= m_fb <= p.k

        u_fb = user_utility(m_fb, a_fb, kap, p)
        # Compare against grid of alternative (m, a) in [0, k] x [0, 1]
        for m_alt in np.linspace(0.0, p.k, 21):
            for a_alt in [0.0, 0.5, 1.0]:
                u_alt = user_utility(m_alt, a_alt, kap, p)
                assert u_alt <= u_fb + 1e-7


def test_first_best_unclipped_algebraic_formula():
    r"""
    The unconstrained algebraic difference formula:
    U(a=1) - U(a=0) = (Lambda - c_Q) * (k - (Lambda + c_Q)/(2 * kappa))
    operates when m is permitted to exceed k (unclipped benchmark).
    """
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    for kap in [0.2, 0.35, 0.5, 0.8]:
        m1 = p.c_Q / kap
        m0 = p.Lambda / kap
        u1 = user_utility(m1, 1.0, kap, p)
        u0 = user_utility(m0, 0.0, kap, p)
        actual_diff = u1 - u0
        predicted_diff = (p.Lambda - p.c_Q) * (p.k - (p.Lambda + p.c_Q) / (2.0 * kap))
        assert actual_diff == pytest.approx(predicted_diff, rel=1e-7)


def test_first_best_exact_conjunctive_threshold():
    r"""
    In the exact conjunctive model U_{conj}(m, a) = V q^{k-m} - 0.5 kappa m^2 - a c_Q (k-m),
    a genuine threshold in kappa emerges where asking vs guessing trades off.
    """
    p = ModelParams(k=5.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    m_low, a_low, u_low = first_best_exact(0.1, p)
    m_high, a_high, u_high = first_best_exact(1.5, p)
    assert 0.0 <= m_low <= p.k
    assert 0.0 <= m_high <= p.k
    assert a_high == 1.0
