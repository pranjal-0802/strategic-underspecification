"""Tests for results added or corrected in the October 2026 review of the manuscript."""
import numpy as np
import pytest

from underspec_sim.verifications.verify_prop7_regimes import closed_form, solve_menu_exact, _leader, _U


def _clipped_pool(k, Lam, c_Q, lam, mu, kappas, weights, a_grid):
    pi = np.zeros_like(a_grid)
    for kap, w in zip(kappas, weights):
        m = np.minimum(k, (Lam * (1 - a_grid) + a_grid * c_Q) / kap)
        u = 100 - Lam * (1 - a_grid) * (k - m) - kap / 2 * m * m - a_grid * c_Q * (k - m)
        pi += w * (mu * u - (lam - c_Q) * a_grid * (k - m))
    return pi


def test_unbiased_clipped_pooling_payoff_is_convex():
    """Corollary 4: max_m U(m, a) is convex in a even with clipping, so unbiased pooling is a corner."""
    rng = np.random.default_rng(0)
    a = np.linspace(0, 1, 401)
    for _ in range(50):
        k = rng.uniform(3, 15); Lam = rng.uniform(1, 10); c_Q = rng.uniform(0.2, 5)
        kappas = rng.uniform(0.05, 2.0, size=5); weights = np.ones(5) / 5
        pi = _clipped_pool(k, Lam, c_Q, c_Q, 1.0, kappas, weights, a)
        assert np.all(np.diff(pi, 2) >= -1e-9)
        assert int(np.argmax(pi)) in (0, len(a) - 1) or np.isclose(pi.max(), max(pi[0], pi[-1]))


def test_pooling_rate_weakly_decreasing_in_friction_under_clipping():
    """Corollary 2 (clipping-robust): decreasing differences in (a, lambda_A) => a_SE weakly decreasing."""
    rng = np.random.default_rng(1)
    a = np.linspace(0, 1, 2001)
    for _ in range(25):
        k = rng.uniform(5, 12); Lam = rng.uniform(3, 9); c_Q = rng.uniform(0.5, 0.9 * Lam)
        kappas = rng.uniform(0.1, 1.5, size=4); weights = np.ones(4) / 4
        prev = 1.0
        for lam in np.linspace(c_Q, c_Q + 2 * (Lam - c_Q), 40):
            a_se = a[int(np.argmax(_clipped_pool(k, Lam, c_Q, lam, 1.0, kappas, weights, a)))]
            assert a_se <= prev + 1e-9
            prev = a_se


def test_welfare_ordering_pooling_breakpoints():
    """Remark (welfare ordering): pooling a_SE = 1 up to lambda_A = 3.0, then 2/3, then 0 from 3.5."""
    a = np.linspace(0, 1, 30001)
    def a_se(lam):
        return a[int(np.argmax(_clipped_pool(10, 5, 2, lam, 1.0, (0.3, 0.5), (0.5, 0.5), a)))]
    assert a_se(2.99) == pytest.approx(1.0)
    assert a_se(3.01) == pytest.approx(2 / 3, abs=1e-3)
    assert a_se(3.49) == pytest.approx(2 / 3, abs=1e-3)
    assert a_se(3.51) == pytest.approx(0.0)


def test_asking_distortion_bounded_by_relative_bias():
    """Proposition 7 regime 3: 0 < 1 - a_H^SB < rho/s."""
    rng = np.random.default_rng(2)
    n = 0
    while n < 300:
        k = 10.0; Lam = rng.uniform(3, 12); c_Q = rng.uniform(0.5, 3)
        if Lam < 1.15 * c_Q:
            continue
        s = Lam - c_Q; rho = rng.uniform(0.05, 0.95) * s; ceff = c_Q + rho
        kL = rng.uniform(c_Q / k, ceff / k); kH = kL * rng.uniform(1.02, 8)
        if kH <= ceff / k:
            continue
        reg, sat, cand = closed_form(k, Lam, c_Q, 1.0, c_Q + rho, kL, kH, 0.5)
        if reg != "asking":
            continue
        n += 1
        assert 0 < 1 - cand[3] < rho / s


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_prop7_closed_form_and_implementability(seed):
    """Closed forms match an exact grid optimum under saturation, and no type prefers an off-menu prompt."""
    rng = np.random.default_rng(seed)
    checked = 0
    while checked < 4:
        k = 10.0; Lam = rng.uniform(4, 10); c_Q = rng.uniform(0.5, 2.5)
        if Lam < 1.2 * c_Q:
            continue
        s = Lam - c_Q; mu = 1.0; rho = rng.uniform(0.1, 0.9) * s; ceff = c_Q + rho
        kL = rng.uniform(c_Q / k, ceff / k); kH = kL * rng.uniform(1.05, 6)
        if kH <= ceff / k:
            continue
        p = (k, Lam, c_Q, mu, c_Q + rho, kL, kH, 0.5)
        reg, sat, cand = closed_form(*p)
        if not sat:
            continue
        checked += 1
        val, x = solve_menu_exact(*p, nm=501, na=11)
        assert val - _leader(cand, *p) <= 2e-3
        assert _U(x[2], x[3], kH, k, Lam, c_Q, 100.0) >= _U(min(k, Lam / kH), 0.0, kH, k, Lam, c_Q, 100.0) - 1e-6


def test_effort_dictation_gap_is_second_order():
    """Corollary 6: leader-preferred interior bundles are optimal and IC; W_screen = W_FB - rho^2 E[1/kappa]/2."""
    rng = np.random.default_rng(5)
    checked = 0
    while checked < 6:
        k = 10.0; Lam = rng.uniform(4, 10); c_Q = rng.uniform(0.5, 2.5)
        s = Lam - c_Q
        if s < 0.2 * c_Q:
            continue
        kL = rng.uniform(0.2, 0.8); kH = kL * rng.uniform(1.2, 4)
        rho = rng.uniform(0.05, 1.0) * min(0.9 * s, c_Q * (kH - kL) / (kH + kL))
        ceff = c_Q + rho
        if kL < ceff / k:
            continue
        checked += 1
        p = (k, Lam, c_Q, 1.0, c_Q + rho, kL, kH, 0.5)
        menu = (ceff / kL, 1.0, ceff / kH, 1.0)
        U = lambda m, a, kap: _U(m, a, kap, k, Lam, c_Q, 100.0)
        assert U(menu[0], 1, kL) >= U(menu[2], 1, kL) - 1e-9
        assert U(menu[2], 1, kH) >= U(menu[0], 1, kH) - 1e-9
        val, _ = solve_menu_exact(*p, nm=801, na=11)
        assert val - _leader(menu, *p) <= 2e-3
        gap = 0.5 * sum(U(c_Q / kap, 1, kap) - U(ceff / kap, 1, kap) for kap in (kL, kH))
        assert gap == pytest.approx(rho ** 2 / 2 * 0.5 * (1 / kL + 1 / kH), rel=1e-9)


def test_exact_model_small_task_is_a_tie_not_guessing():
    """At k = 3 every type saturates under both policies, so the exact pooling payoff is flat in a."""
    from underspec_sim.verifications.verify_exact_conjunctive import solve_exact_pooling_optimum
    _, _, regime, _, pi = solve_exact_pooling_optimum(np.linspace(0.2, 0.6, 20), g=0.5, k=3, c_Q=2.0, V=100.0)
    assert regime == "tie"
    assert float(np.ptp(pi)) < 1e-9


def test_exact_firstbest_gap_terse_type_guesses(tmp_path):
    """Where the exact model has a zero-bias pooling gap, the terse type guesses and the diligent type asks."""
    from underspec_sim.verifications.verify_exact_firstbest_gap import run_verification
    res = run_verification(output_dir=str(tmp_path), n_draws=400, seed=11)
    assert res["n_pos"] > 0
    assert res["diligent_guesses"] == 0
    assert res["terse_guesses"] == res["n_pos"]
    assert res["n_pos_short"] == 0


def test_self_confirming_learning_path():
    """Lemma 1: the naive learning path falls monotonically to a fixed point below m*(a), ordered in d."""
    k, Lam, c_Q, kap, a = 10.0, 5.0, 2.0, 0.6, 0.3
    m_star = min(k, (Lam * (1 - a) + a * c_Q) / kap)

    def path(d0):
        d = lambda m: d0 + (1 - d0) * m / k
        T = lambda m: min(k, (Lam * d(m) * (1 - a) + a * c_Q) / kap)
        ms = [m_star]
        for _ in range(500):
            ms.append(T(ms[-1]))
        return np.array(ms), T

    lo, T_lo = path(0.3)
    hi, T_hi = path(0.7)
    for ms, T in ((lo, T_lo), (hi, T_hi)):
        assert np.all(np.diff(ms) <= 1e-12)
        assert abs(T(ms[-1]) - ms[-1]) < 1e-9
        assert ms[-1] < m_star
    assert lo[-1] <= hi[-1]


def test_continuous_schedule_implements_dictation_menu_but_threshold_does_not():
    """Lemma 4 on the Corollary 6 menu: alpha(m) implements it; a minimum-effort threshold lets type L shorten."""
    from underspec_sim.verifications.verify_continuous_implementation import audit, C_Q
    kL, kH, ce = 0.3, 0.5, 2.5
    res = audit(((ce / kL, 1.0), (ce / kH, 1.0)), kL, kH)
    assert res["gain_L"] <= 1e-9 and res["gain_H"] <= 1e-9
    assert res["attain"] >= -1e-9
    # Under the threshold, type L picks c_Q / kappa_L with a = 1 and gains rho^2 / (2 kappa_L).
    assert res["thr_gain_L"] == pytest.approx((ce - C_Q) ** 2 / (2 * kL), abs=1e-4)


def test_continuous_implementation_audit(tmp_path):
    """Lemma 4 (i)-(v) on solver menus, including menus with binding IC_L."""
    from underspec_sim.verifications.verify_continuous_implementation import run_verification
    res = run_verification(output_dir=str(tmp_path), n_draws=60, seed=7)
    assert res["passed"]
    assert res["n_binding"] > 0
    assert res["thr_fail_binding"] == res["n_binding"]


def test_heterogeneous_stakes_threshold(tmp_path):
    """Proposition 8: diligent users guess, terse users ask, with a single sign change of G at kappa_hat."""
    from underspec_sim.verifications.verify_heterogeneous_stakes import run_verification
    res = run_verification(output_dir=str(tmp_path), n_profiles=10, seed=13)
    assert res["passed"]
    assert res["n_ok"] == res["n_profiles"]
    assert 0 < res["gap"] < res["target_gain"]


def test_biased_first_best_on_feasible_domain_has_no_spurious_threshold():
    """With clip=True the leader's preferred bundle is (min{k, c_eff/kappa}, 1) whenever c_eff < Lambda."""
    from underspec_sim.core.params import ModelParams
    from underspec_sim.core.first_best import biased_first_best
    p = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=3.0)  # c_eff = 3 < Lambda = 5
    for kap in (0.31, 0.35, 0.39, 0.5, 1.0):  # 0.35 lies below the unclipped threshold (Lambda + c_eff)/(2k) = 0.4
        m, a = biased_first_best(kap, p, clip=True)
        assert a == 1.0 and m == pytest.approx(3.0 / kap)
    m, a = biased_first_best(0.25, p, clip=True)  # c_eff / kappa >= k: full specification, asking irrelevant
    assert m == 10.0 and a == 0.0
    m, a = biased_first_best(np.array([0.25, 0.35]), p, clip=True)
    assert np.allclose(m, [10.0, 3.0 / 0.35]) and np.allclose(a, [0.0, 1.0])
