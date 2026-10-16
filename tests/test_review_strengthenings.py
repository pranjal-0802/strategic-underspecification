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
