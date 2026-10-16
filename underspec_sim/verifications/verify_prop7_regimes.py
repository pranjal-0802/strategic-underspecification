r"""
underspec_sim.verifications.verify_prop7_regimes:
Exact-grid audit of Proposition 7 (screening under bias), independent of the SLSQP solver.

The solver exploits linearity in the ask rate: for fixed (m_L, a_L, m_H), the constraints IC_L and IC_H
and the leader's payoff from type H are affine in a_H, so the optimal a_H is an interval endpoint.
m_L and m_H are gridded on [0, k] and a_L on [0, 1].

Checks:
1. Regimes 1-4 closed forms (with the saturation condition of each regime) match the exact optimum.
2. When the saturation condition fails, the leader departs to (m_L < k, a_L = 1) (boundary relaxation).
3. Effort-regime saturation condition f_H (c_eff - kappa_H m_hat) <= f_L (c_eff - kappa_L k) is sufficient.
4. Asking distortion bound 0 < 1 - a_H^SB < rho/s.
5. Lemma (implementability): no type prefers an off-menu prompt answered with a = 0.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def _U(m, a, kap, k, Lam, c_Q, V):
    return V - Lam * (1 - a) * (k - m) - 0.5 * kap * m * m - a * c_Q * (k - m)


def solve_menu_exact(k, Lam, c_Q, mu, lam, kL, kH, fL, V=100.0, nm=801, na=21) -> Tuple[float, Tuple[float, float, float, float]]:
    s = Lam - c_Q
    b = lam - c_Q
    fH = 1.0 - fL
    grid = np.linspace(0.0, k, nm)
    mL = grid[:, None]
    mH = grid[None, :]
    slope = s * (k - mH)
    c0L = _U(mH, 0.0, kL, k, Lam, c_Q, V)
    c0H = _U(mH, 0.0, kH, k, Lam, c_Q, V)
    best_val, best_x = -np.inf, None
    for aL in np.linspace(0.0, 1.0, na):
        uLL = _U(mL, aL, kL, k, Lam, c_Q, V)
        uHL = _U(mL, aL, kH, k, Lam, c_Q, V)
        piL = mu * uLL - b * aL * (k - mL)
        with np.errstate(divide="ignore", invalid="ignore"):
            ub = np.where(slope > 0, (uLL - c0L) / slope, np.where(uLL - c0L >= -1e-12, 1.0, -1.0))
            lb = np.where(slope > 0, (uHL - c0H) / slope, np.where(c0H - uHL >= -1e-12, 0.0, 2.0))
        ub = np.minimum(ub, 1.0)
        lb = np.maximum(lb, 0.0)
        aH = np.where(mu * s - b >= 0, ub, lb)
        piH = mu * _U(mH, aH, kH, k, Lam, c_Q, V) - b * aH * (k - mH)
        tot = np.where(ub >= lb - 1e-12, fL * piL + fH * piH, -np.inf)
        i, j = np.unravel_index(np.argmax(tot), tot.shape)
        if tot[i, j] > best_val:
            best_val, best_x = float(tot[i, j]), (float(grid[i]), float(aL), float(grid[j]), float(aH[i, j]))
    return best_val, best_x


def closed_form(k, Lam, c_Q, mu, lam, kL, kH, fL):
    s = Lam - c_Q
    rho = (lam - c_Q) / mu
    ceff = c_Q + rho
    fH = 1.0 - fL
    mB = ceff / kH
    mstar = Lam * rho / (s * (kH - kL) + rho * kL)
    kslack = 2 * c_Q / kL - mB
    kcrit = 2 * c_Q / kL - mstar
    t = fH / fL * (s - rho) / s
    kbar = (ceff + t * c_Q) / (k * (1 + t))
    if k <= kslack:
        return "slack", True, (k, 0.0, mB, 1.0)
    if k <= kcrit:
        m_hat = 2 * c_Q / kL - k
        sat = fH * (ceff - kH * m_hat) <= fL * (ceff - kL * k)
        return "effort", sat, (k, 0.0, m_hat, 1.0)
    if k < mstar:
        return "full_spec", kL <= kbar, (k, 0.0, k, 0.0)
    return "asking", kL <= kbar, (k, 0.0, mstar, 1 - (kL * (k + mstar) / 2 - c_Q) / s)


def _leader(x, k, Lam, c_Q, mu, lam, kL, kH, fL, V=100.0):
    b = lam - c_Q
    mL, aL, mH, aH = x
    return (fL * (mu * _U(mL, aL, kL, k, Lam, c_Q, V) - b * aL * (k - mL))
            + (1 - fL) * (mu * _U(mH, aH, kH, k, Lam, c_Q, V) - b * aH * (k - mH)))


def _draw(rng, regime=None):
    while True:
        k = float(rng.choice([6.0, 10.0, 14.0]))
        Lam = rng.uniform(3, 12)
        c_Q = rng.uniform(0.5, 3)
        if Lam < 1.15 * c_Q:
            continue
        s = Lam - c_Q
        mu = rng.uniform(0.4, 1.0)
        rho = rng.uniform(0.05, 0.95) * s
        ceff = c_Q + rho
        kL = rng.uniform(c_Q / k, ceff / k)
        kH = kL * rng.uniform(1.02, 8.0)
        if kH <= ceff / k:
            continue
        fL = rng.uniform(0.1, 0.9)
        p = (k, Lam, c_Q, mu, c_Q + mu * rho, kL, kH, fL)
        reg, sat, cand = closed_form(*p)
        if regime is None or reg == regime:
            return p, reg, sat, cand


def run_verification(output_dir: str = "outputs", n_general: int = 150, n_effort: int = 120, seed: int = 7) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for phase, n, regime in (("general", n_general, None), ("effort", n_effort, "effort")):
        for _ in range(n):
            p, reg, sat, cand = _draw(rng, regime)
            k, Lam, c_Q, mu, lam, kL, kH, fL = p
            val, x = solve_menu_exact(*p)
            gap = val - _leader(cand, *p)
            uH = _U(x[2], x[3], kH, k, Lam, c_Q, 100.0)
            u_out_H = _U(min(k, Lam / kH), 0.0, kH, k, Lam, c_Q, 100.0)
            uL = _U(x[0], x[1], kL, k, Lam, c_Q, 100.0)
            u_out_L = _U(k, 0.0, kL, k, Lam, c_Q, 100.0)
            s = Lam - c_Q
            rho = (lam - c_Q) / mu
            rows.append({
                "phase": phase, "regime": reg, "saturation": sat, "k": k, "Lambda": Lam, "c_Q": c_Q, "mu_A": mu,
                "lambda_A": lam, "kappa_L": kL, "kappa_H": kH, "f_L": fL,
                "m_L": x[0], "a_L": x[1], "m_H": x[2], "a_H": x[3], "gap_vs_closed_form": gap,
                "closed_form_ok": (gap <= 1e-3) if sat else True,
                "departs_when_unsaturated": (x[1] > 0.5 and x[0] < k - 1e-9) if not sat else True,
                "distortion_bound_ok": (0 < 1 - cand[3] < rho / s) if (reg == "asking" and sat) else True,
                "implementable": (uH >= u_out_H - 1e-6) and (uL >= u_out_L - 1e-6),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "prop7_regimes.csv")
    df.to_csv(csv_path, index=False)

    sat = df[df["saturation"]]
    unsat = df[~df["saturation"]]
    eff = df[df["phase"] == "effort"]
    eff_sat = eff[eff["saturation"]]
    summary = {
        "closed_form_ok": int(sat["closed_form_ok"].sum()), "n_saturated": int(len(sat)),
        "departs": int(unsat["departs_when_unsaturated"].sum()), "n_unsaturated": int(len(unsat)),
        "effort_sufficient": int(eff_sat["closed_form_ok"].sum()), "n_effort_saturated": int(len(eff_sat)),
        "n_effort": int(len(eff)),
        "bound_ok": bool(df["distortion_bound_ok"].all()),
        "implementable": int(df["implementable"].sum()), "n": int(len(df)),
    }
    passed = (summary["closed_form_ok"] == summary["n_saturated"] and summary["bound_ok"]
              and summary["implementable"] == summary["n"])
    md_path = os.path.join(output_dir, "prop7_regimes.md")
    with open(md_path, "w") as f:
        f.write(
            "# Proposition 7 Regime Audit (exact grid solver)\n\n"
            f"**Status:** **{'PASS' if passed else 'FAIL'}**\n\n"
            f"- Closed forms (regimes 1-4) match the exact optimum whenever the saturation condition holds: "
            f"{summary['closed_form_ok']}/{summary['n_saturated']}\n"
            f"- Leader departs to (m_L < k, a_L = 1) when saturation fails: {summary['departs']}/{summary['n_unsaturated']}\n"
            f"- Effort regime: saturation condition sufficient in {summary['effort_sufficient']}/{summary['n_effort_saturated']} "
            f"saturated draws ({summary['n_effort']} effort-regime draws)\n"
            f"- Asking-distortion bound 0 < 1 - a_H < rho/s holds: {summary['bound_ok']}\n"
            f"- Lemma (implementability with off-menu a = 0): {summary['implementable']}/{summary['n']}\n"
        )
    return {"passed": bool(passed), "summary": summary, "df": df, "csv_path": csv_path, "md_path": md_path}


if __name__ == "__main__":
    res = run_verification()
    print("Prop 7 regime audit:", "PASS" if res["passed"] else "FAIL", res["summary"])
