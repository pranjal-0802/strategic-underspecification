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
    r"""
    Derive the leader's pooling objective directly from primitives using SymPy:
    Pi_kappa = mu_A * U(m*, a; kappa) - (lambda_A - c_Q) * a * (k - m*)
    where m* = beta - a * gamma.
    Verify that expanding in a yields Pi = const + L * a + Q * a^2,
    with dPi/da = L + 2 * Q * a = 0 giving a* = - L / (2 * Q),
    and d2Pi/da2 = 2 * Q.
    """
    a, kappa, Lambda, c_Q, V, k, mu_A, lambda_A = sp.symbols(
        'a kappa Lambda c_Q V k mu_A lambda_A', positive=True
    )
    m_star = (Lambda - a * (Lambda - c_Q)) / kappa
    U = V - Lambda * (1 - a) * (k - m_star) - (kappa / 2) * m_star**2 - a * c_Q * (k - m_star)
    Pi_kappa = mu_A * U - (lambda_A - c_Q) * a * (k - m_star)

    # Collect powers of a:
    poly_a = sp.collect(sp.expand(Pi_kappa), a)
    Q_kappa = poly_a.coeff(a, 2)
    L_kappa = poly_a.coeff(a, 1)

    s = Lambda - c_Q
    b = lambda_A - c_Q
    gamma = s / kappa
    R_0 = k - Lambda / kappa

    expected_Q = (mu_A * s - 2 * b) * gamma / 2
    expected_L = (mu_A * s - b) * R_0

    assert sp.simplify(Q_kappa - expected_Q) == 0
    assert sp.simplify(L_kappa - expected_L) == 0

    # FOC and SOC:
    a_SE_sol = sp.solve(sp.diff(poly_a, a), a)[0]
    expected_a_SE = - expected_L / (2 * expected_Q)
    assert sp.simplify(a_SE_sol - expected_a_SE) == 0
    assert sp.simplify(sp.diff(poly_a, a, 2) - 2 * expected_Q) == 0


def test_sympy_unbiased_pooling_convexity():
    r"""
    Prove symbolically from primitives that in the unbiased case (mu_A=1, lambda_A=c_Q => b=0),
    Q_kappa = s^2 / (2 * kappa) >= 0 ALWAYS, so Pi(a) is weakly convex on [0, 1]
    and the optimum is always a corner solution.
    Furthermore, Pi(1) - Pi(0) = s * (R_0 + gamma / 2).
    """
    a, kappa, Lambda, c_Q, V, k = sp.symbols('a kappa Lambda c_Q V k', positive=True)
    s = Lambda - c_Q
    m_star = (Lambda - a * s) / kappa
    U = V - Lambda * (1 - a) * (k - m_star) - (kappa / 2) * m_star**2 - a * c_Q * (k - m_star)

    d2U_da2 = sp.diff(U, a, 2)
    expected_d2 = s**2 / kappa
    assert sp.simplify(d2U_da2 - expected_d2) == 0

    # Corner difference Pi(1) - Pi(0)
    U_1 = U.subs(a, 1)
    U_0 = U.subs(a, 0)
    diff = sp.simplify(U_1 - U_0)
    gamma = s / kappa
    R_0 = k - Lambda / kappa
    expected_diff = s * (R_0 + gamma / 2)
    assert sp.simplify(diff - expected_diff) == 0


def test_sympy_single_crossing_rent():
    r"""Prove symbolically that rent = 0.5 * (kappa_H - kappa_L) * m_H^2."""
    m_H, a_H, kappa_L, kappa_H, Lambda, c_Q, V, k = sp.symbols('m_H a_H kappa_L kappa_H Lambda c_Q V k', positive=True)

    U_L = V - Lambda * (1 - a_H) * (k - m_H) - (kappa_L / 2) * m_H**2 - a_H * c_Q * (k - m_H)
    U_H = V - Lambda * (1 - a_H) * (k - m_H) - (kappa_H / 2) * m_H**2 - a_H * c_Q * (k - m_H)

    rent = sp.simplify(U_L - U_H)
    expected_rent = (kappa_H - kappa_L) * m_H**2 / 2

    assert sp.simplify(rent - expected_rent) == 0
