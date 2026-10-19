r"""
underspec_sim.model3_continuum.standard: screening a continuum of specification-cost types.

Writing x = a s (k - m) for the value of clarification, user utility is quasilinear,
    U = U0(m, kappa) + x,   U0(m, kappa) = V - Lambda (k - m) - kappa m^2 / 2,
with the cap 0 <= x <= s (k - m). By Lemma 2 the leader maximizes E[U0 + (1 - r) x],
r = rho / s in (0, 1): it values the transfer it gives, and only the cap limits it.

The minimum-specification standard asks about every open attribute iff m >= mbar.
Types with c_Q / kappa >= mbar keep their first-best prompt c_Q / kappa; the others
bunch at mbar. Theorem 1 states that the standard is optimal among all incentive-compatible
mechanisms iff the tail condition
    int_tau^kbar [ r f(t) (Lambda - t mbar) - (1 - r) (1 - F(t)) mbar ] dt >= 0
holds for every tau in the bunch.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar


@dataclass
class TypeDistribution:
    """Continuous type distribution on [lo, hi] given by an (unnormalized) density."""
    lo: float
    hi: float
    density: Callable[[np.ndarray], np.ndarray]
    n_quad: int = 20001

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.linspace(self.lo, self.hi, self.n_quad)
        w = np.maximum(self.density(t), 0.0)
        f = w / np.trapezoid(w, t)
        F = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(t))])
        F = F / F[-1]
        return t, f, F

    def discrete(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """n types at cell midpoints with probabilities equal to the cell masses."""
        edges = np.linspace(self.lo, self.hi, n + 1)
        t, f, F = self.grid()
        Fe = np.interp(edges, t, F)
        return 0.5 * (edges[1:] + edges[:-1]), np.diff(Fe)


def U0(m, kappa, V, Lam, k):
    return V - Lam * (k - m) - 0.5 * kappa * m ** 2


def U1(m, kappa, V, c_Q, k):
    return V - c_Q * (k - m) - 0.5 * kappa * m ** 2


def standard_value(mbar: float, t: np.ndarray, f: np.ndarray, V: float, Lam: float, c_Q: float,
                   k: float, rho: float) -> float:
    """Leader payoff (divided by mu_A) under a(m) = 1{m >= mbar}, every type meeting the standard."""
    m = np.maximum(mbar, np.minimum(k, c_Q / t))
    return float(np.trapezoid(f * (U1(m, t, V, c_Q, k) - rho * (k - m)), t))


def standard_threshold(dist: TypeDistribution, V: float, Lam: float, c_Q: float, k: float,
                       rho: float) -> Dict[str, float]:
    """Optimal standard mbar (maximizer of the leader payoff over standards) and its first-order check."""
    t, f, F = dist.grid()
    res = minimize_scalar(lambda mb: -standard_value(mb, t, f, V, Lam, c_Q, k, rho),
                          bounds=(0.0, k), method="bounded", options={"xatol": 1e-10})
    mbar = float(res.x)
    kb = max(dist.lo, c_Q / mbar)
    sel = t >= kb
    cond_mean = float(np.trapezoid(f[sel] * t[sel], t[sel]) / np.trapezoid(f[sel], t[sel]))
    return {"mbar": mbar, "kappa_b": kb, "cond_mean": cond_mean, "foc_mbar": (c_Q + rho) / cond_mean,
            "value": -float(res.fun)}


def tail_condition(dist: TypeDistribution, mbar: float, Lam: float, c_Q: float, r: float) -> float:
    """min over tau in the bunch of int_tau^kbar [r f (Lam - t mbar) - (1 - r)(1 - F) mbar] dt."""
    t, f, F = dist.grid()
    zeta = r * f * (Lam - t * mbar) - (1 - r) * (1 - F) * mbar
    seg = (zeta[1:] + zeta[:-1]) / 2 * np.diff(t)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    kb = max(dist.lo, c_Q / mbar)
    return float(tail[t >= kb].min())


def standard_on_types(kaps: np.ndarray, probs: np.ndarray, V: float, Lam: float, c_Q: float, k: float,
                      rho: float) -> Dict[str, object]:
    """Best standard for a finite set of types (same objects as the N-type optimum)."""
    def val(mb):
        m = np.maximum(mb, np.minimum(k, c_Q / kaps))
        return float(np.sum(probs * (U1(m, kaps, V, c_Q, k) - rho * (k - m))))
    res = minimize_scalar(lambda mb: -val(mb), bounds=(0.0, k), method="bounded", options={"xatol": 1e-10})
    mbar = float(res.x)
    return {"mbar": mbar, "m": np.maximum(mbar, np.minimum(k, c_Q / kaps)), "value": val(mbar)}


def recover_wedge(mbar: float, cond_mean: float, c_Q: float) -> float:
    """Corollary: rho = mbar * E[kappa | kappa >= kappa_b] - c_Q."""
    return mbar * cond_mean - c_Q


def solve_discrete_menu(kaps: np.ndarray, probs: np.ndarray, V: float, Lam: float, c_Q: float, k: float,
                        r: float, start: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Optimal direct mechanism for finitely many types (adjacent IC both ways, monotone m, 0 <= x <= s(k-m)).

    Adjacent IC with m nonincreasing is sufficient for global IC because U0 has decreasing
    differences in (m, kappa) and utility is quasilinear in x.
    """
    s = Lam - c_Q
    n = len(kaps)
    idx = np.arange(n - 1)

    def obj(z):
        m, x = z[:n], z[n:]
        return -np.sum(probs * (U0(m, kaps, V, Lam, k) + (1 - r) * x))

    def grad(z):
        m = z[:n]
        return -np.concatenate([probs * (Lam - kaps * m), probs * (1 - r)])

    def cons(z):
        m, x = z[:n], z[n:]
        own = U0(m, kaps, V, Lam, k) + x
        down = own[:-1] - (U0(m[1:], kaps[:-1], V, Lam, k) + x[1:])
        up = own[1:] - (U0(m[:-1], kaps[1:], V, Lam, k) + x[:-1])
        return np.concatenate([down, up, m[:-1] - m[1:], s * (k - m) - x])

    def cjac(z):
        m = z[:n]
        J = np.zeros((3 * (n - 1) + n, 2 * n))
        J[idx, idx] = Lam - kaps[:-1] * m[:-1]
        J[idx, idx + 1] = -(Lam - kaps[:-1] * m[1:])
        J[idx, n + idx] = 1.0
        J[idx, n + idx + 1] = -1.0
        o = n - 1
        J[o + idx, idx + 1] = Lam - kaps[1:] * m[1:]
        J[o + idx, idx] = -(Lam - kaps[1:] * m[:-1])
        J[o + idx, n + idx + 1] = 1.0
        J[o + idx, n + idx] = -1.0
        o += n - 1
        J[o + idx, idx] = 1.0
        J[o + idx, idx + 1] = -1.0
        o += n - 1
        J[o + np.arange(n), np.arange(n)] = -s
        J[o + np.arange(n), n + np.arange(n)] = -1.0
        return J

    m_fb = np.minimum(k, c_Q / kaps)
    starts = [np.concatenate([m_fb, s * (k - m_fb)])]
    if start is not None:
        starts.append(start)
    best = None
    for st in starts:
        res = minimize(obj, st, jac=grad, method="SLSQP", bounds=[(0, k)] * n + [(0, None)] * n,
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options={"maxiter": 5000, "ftol": 1e-13})
        if best is None or res.fun < best.fun:
            best = res
    m, x = best.x[:n], best.x[n:]
    a = np.where(k - m > 1e-9, x / (s * np.maximum(k - m, 1e-12)), 1.0)
    return {"m": m, "x": x, "a": np.clip(a, 0, 1), "value": -float(best.fun), "success": bool(best.success)}
