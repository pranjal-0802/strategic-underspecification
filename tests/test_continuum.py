"""Tests for Section 8 (continuum screening) and Section 9 (competition and disclosure)."""
import numpy as np
import pytest

from underspec_sim.model3_continuum import (
    TypeDistribution, standard_threshold, tail_condition, solve_discrete_menu, recover_wedge,
)

V, K, LAM, CQ = 100.0, 10.0, 5.0, 2.0
S = LAM - CQ
UNIFORM = TypeDistribution(0.3, 0.6, lambda t: np.ones_like(t))


def test_standard_first_order_condition():
    """Theorem 1(i): mbar = c_Q^eff / E[kappa | kappa >= kappa_b]."""
    for rho in (0.3, 0.6, 1.2):
        st = standard_threshold(UNIFORM, V, LAM, CQ, K, rho)
        assert st["mbar"] == pytest.approx(st["foc_mbar"], abs=1e-4)


def test_all_pooled_closed_form():
    """With the whole support pooled, mbar = c_Q^eff / E[kappa] (7.111 for rho = 1.2 on U[0.3, 0.6])."""
    st = standard_threshold(UNIFORM, V, LAM, CQ, K, 1.2)
    assert st["kappa_b"] == pytest.approx(0.3)
    assert st["mbar"] == pytest.approx(3.2 / 0.45, abs=1e-4)


def test_standard_nondecreasing_in_bias():
    mbars = [standard_threshold(UNIFORM, V, LAM, CQ, K, rho)["mbar"] for rho in (0.15, 0.3, 0.6, 0.9, 1.2)]
    assert np.all(np.diff(mbars) > 0)


def test_uniform_tail_condition_equivalence():
    """Theorem 1(iv): for uniform types the tail condition holds iff mbar <= Lambda / kbar."""
    for r in (0.2, 0.4, 0.5, 0.6, 0.7):
        st = standard_threshold(UNIFORM, V, LAM, CQ, K, r * S)
        tc = tail_condition(UNIFORM, st["mbar"], LAM, CQ, r)
        assert (tc >= -1e-6) == (st["mbar"] <= LAM / UNIFORM.hi + 1e-9)


def test_discrete_optimum_matches_standard_when_tail_holds():
    """With every type pooled (r = 0.4), the 30-type optimum is exactly the standard."""
    r = 0.4
    st = standard_threshold(UNIFORM, V, LAM, CQ, K, r * S)
    kaps, p = UNIFORM.discrete(30)
    sol = solve_discrete_menu(kaps, p, V, LAM, CQ, K, r)
    assert np.allclose(sol["m"], st["mbar"], atol=1e-3)
    assert np.all(sol["a"] > 1 - 1e-4)
    assert sol["value"] == pytest.approx(st["value"], abs=1e-4)


def test_discrete_optimum_beats_standard_when_tail_fails():
    """r = 0.6 on U[0.3, 0.6]: mbar = 8.44 > Lambda / kbar = 8.33; tersest users get partial clarification."""
    r = 0.6
    st = standard_threshold(UNIFORM, V, LAM, CQ, K, r * S)
    assert st["mbar"] > LAM / UNIFORM.hi
    kaps, p = UNIFORM.discrete(40)
    sol = solve_discrete_menu(kaps, p, V, LAM, CQ, K, r)
    assert sol["value"] > st["value"] + 1e-5
    assert sol["a"][-1] < 0.99


def test_wedge_recovered_from_bunching_point():
    """Corollary 8: rho = mbar E[kappa | pooled] - c_Q."""
    for rho in (0.3, 0.9):
        st = standard_threshold(UNIFORM, V, LAM, CQ, K, rho)
        assert recover_wedge(st["mbar"], st["cond_mean"], CQ) == pytest.approx(rho, abs=1e-3)


def test_welfare_cost_of_bias_is_third_order():
    """Corollary 7: W^FB - W_std = (4/3) f(kbar) rho^3 / c_Q + O(rho^4)."""
    dist = TypeDistribution(0.22, 0.8, lambda t: 0.2 + (t - 0.22), n_quad=100001)
    t, f, F = dist.grid()
    u1 = lambda m: V - CQ * (K - m) - 0.5 * t * m ** 2
    w_fb = np.trapezoid(f * u1(np.minimum(K, CQ / t)), t)
    ratios = []
    for rho in (0.1, 0.05):
        st = standard_threshold(dist, V, LAM, CQ, K, rho)
        loss = w_fb - np.trapezoid(f * u1(np.maximum(st["mbar"], np.minimum(K, CQ / t))), t)
        ratios.append(loss / (4 / 3 * f[-1] * rho ** 3 / CQ))
    assert abs(ratios[1] - 1) < abs(ratios[0] - 1) < 0.2


def test_competition_with_disclosure_is_pooling_with_real_cost():
    """Proposition 10(i): the disclosed competitive ask rate maximizes U*(a) - c_A a R(a); with c_A = 0 it is a = 1."""
    kap = 0.5
    a = np.linspace(0, 1, 1001)
    m = np.minimum(K, (LAM * (1 - a) + a * CQ) / kap)
    u = V - LAM * (1 - a) * (K - m) - 0.5 * kap * m ** 2 - a * CQ * (K - m)
    assert a[np.argmax(u)] == 1.0
    rates = [a[np.argmax(u - c_A * a * (K - m))] for c_A in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)]
    assert np.all(np.diff(rates) <= 0)   # the competitive rate falls with the real cost of questions
    assert rates[-1] == 0.0               # c_A >= s: asking is never worth its real cost
