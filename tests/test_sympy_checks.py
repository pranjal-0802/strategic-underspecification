"""
tests/test_sympy_checks.py: Symbolic cross-checks of all FOCs and Hessians using SymPy.
"""

import sympy as sp


def test_sympy_user_best_response_foc():
    r"""Verify dU/dm = 0 gives m* = beta - a * gamma."""
    m, a, kappa, Lambda, c_Q, V, k = sp.symbols('m a kappa Lambda c_Q V k', positive=True)
    U = V - Lambda * (1 - a) * (k - m) - (kappa / 2) * m**2 - a * c_Q * (k - m)

    # dU/dm
    dU_dm = sp.diff(U, m)
    # Solve dU/dm = 0 for m
    m_sol = sp.solve(dU_dm, m)[0]

    beta = Lambda / kappa
    gamma = (Lambda - c_Q) / kappa
    m_paper = beta - a * gamma

    assert sp.simplify(m_sol - m_paper) == 0

    # dm*/da
    dm_da = sp.diff(m_paper, a)
    assert sp.simplify(dm_da - (c_Q - Lambda) / kappa) == 0


def test_sympy_prop3_first_best_threshold():
    r"""Verify U(m*(1), 1) - U(m*(0), 0) = (Lambda - c_Q) * (k - (Lambda + c_Q)/(2*kappa))."""
    kappa, Lambda, c_Q, V, k = sp.symbols('kappa Lambda c_Q V k', positive=True)

    # At a=1, m = c_Q / kappa
    U_1 = V - c_Q * (k - c_Q / kappa) - (kappa / 2) * (c_Q / kappa)**2
    # At a=0, m = Lambda / kappa
    U_0 = V - Lambda * (k - Lambda / kappa) - (kappa / 2) * (Lambda / kappa)**2

    diff = sp.simplify(U_1 - U_0)
    expected_diff = (Lambda - c_Q) * (k - (Lambda + c_Q) / (2 * kappa))

    assert sp.simplify(diff - expected_diff) == 0


def test_sympy_prop4_pooling_foc_and_soc():
    r"""Verify dPi/da = 0 gives Proposition 4's a_SE and second derivative is -2*Delta*gamma."""
    a, mu_A, V, C0, Delta, R0_bar, gamma_bar = sp.symbols('a mu_A V C0 Delta R0_bar gamma_bar')
    Pi = mu_A * V - C0 * R0_bar - a * (C0 * gamma_bar + Delta * R0_bar) - a**2 * (Delta * gamma_bar)

    dPi_da = sp.diff(Pi, a)
    a_SE_sol = sp.solve(dPi_da, a)[0]

    expected_a_SE = - (C0 * gamma_bar + Delta * R0_bar) / (2 * Delta * gamma_bar)
    assert sp.simplify(a_SE_sol - expected_a_SE) == 0

    d2Pi_da2 = sp.diff(dPi_da, a)
    assert sp.simplify(d2Pi_da2 - (-2 * Delta * gamma_bar)) == 0


def test_sympy_unbiased_pooling_convexity():
    r"""Prove symbolically that in the unbiased case, Delta * gamma_bar <= 0 always."""
    Lambda, c_Q, inv_kappa = sp.symbols('Lambda c_Q inv_kappa', positive=True)
    # Unbiased: mu_A = 1, lambda_A = c_Q
    Delta = c_Q - Lambda
    gamma_bar = (Lambda - c_Q) * inv_kappa

    soc_val = Delta * gamma_bar
    # soc_val = (c_Q - Lambda) * (Lambda - c_Q) * inv_kappa = - (Lambda - c_Q)^2 * inv_kappa
    expected = - (Lambda - c_Q)**2 * inv_kappa

    assert sp.simplify(soc_val - expected) == 0
    # Second derivative of Pi with respect to a:
    d2Pi_da2 = - 2 * soc_val
    # d2Pi_da2 = + 2 * (Lambda - c_Q)^2 * inv_kappa > 0!
    assert sp.simplify(d2Pi_da2 - 2 * (Lambda - c_Q)**2 * inv_kappa) == 0


def test_sympy_single_crossing_rent():
    r"""Prove symbolically that rent = 0.5 * (kappa_H - kappa_L) * m_H^2."""
    m_H, a_H, kappa_L, kappa_H, Lambda, c_Q, V, k = sp.symbols('m_H a_H kappa_L kappa_H Lambda c_Q V k', positive=True)

    U_L = V - Lambda * (1 - a_H) * (k - m_H) - (kappa_L / 2) * m_H**2 - a_H * c_Q * (k - m_H)
    U_H = V - Lambda * (1 - a_H) * (k - m_H) - (kappa_H / 2) * m_H**2 - a_H * c_Q * (k - m_H)

    rent = sp.simplify(U_L - U_H)
    expected_rent = (kappa_H - kappa_L) * m_H**2 / 2

    assert sp.simplify(rent - expected_rent) == 0
