r"""
underspec_sim.verifications.verify_guessing_trap_incidence:
Incidence of the Guessing Trap (Proposition 5) in the clipped linear-risk pooling model.

For each random draw of primitives, the leader's pooling rate a^{SE}(g) is re-solved on clipped
best responses m^*(\kappa; a) = min{k, (\Lambda(1-a) + a c_Q)/\kappa} at g and g + \Delta g, and we
record whether average prompt length rises AND user welfare falls (the guessing trap).

Sampling distribution (fixed seed, reported in the paper):
  k ~ U{6,...,15}, L ~ U[5, 20], g ~ U[0.50, 0.80], c_Q ~ U[0.5, 3.0] (redrawn unless L(1-g) > 1.1 c_Q),
  mu_A ~ U[0.5, 1.0], lambda_A = c_Q + mu_A * u * s with u ~ U[0, 1.5] and s = L(1-g) - c_Q,
  kappa ~ Uniform[lo, lo * r] with lo ~ U[0.1, 1.0], r ~ U[1.5, 4.0], V = 100, Delta g = 0.02.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd

A_GRID = np.linspace(0.0, 1.0, 1001)
N_KAPPA = 401


def pooling_equilibrium(k, L, g, c_Q, mu_A, lambda_A, lo, hi, V=100.0):
    """Return (a_SE, E[m], W) for the clipped pooling model with kappa ~ Uniform[lo, hi]."""
    Lam = L * (1.0 - g)
    kap = np.linspace(lo, hi, N_KAPPA)[None, :]
    a = A_GRID[:, None]
    m = np.minimum(k, (Lam * (1.0 - a) + a * c_Q) / kap)
    u = V - Lam * (1.0 - a) * (k - m) - 0.5 * kap * m * m - a * c_Q * (k - m)
    pi = (mu_A * u - (lambda_A - c_Q) * a * (k - m)).mean(axis=1)
    i = int(np.argmax(pi))
    return float(A_GRID[i]), float(m[i].mean()), float(u[i].mean())


def run_verification(output_dir: str = "outputs", n_draws: int = 2000, seed: int = 20251016) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    dg = 0.02
    while len(rows) < n_draws:
        k = float(rng.integers(6, 16))
        L = rng.uniform(5, 20)
        g = rng.uniform(0.50, 0.80)
        c_Q = rng.uniform(0.5, 3.0)
        s = L * (1 - g) - c_Q
        if L * (1 - g - dg) <= 1.1 * c_Q:
            continue
        mu_A = rng.uniform(0.5, 1.0)
        lambda_A = c_Q + mu_A * rng.uniform(0.0, 1.5) * s
        lo = rng.uniform(0.1, 1.0)
        hi = lo * rng.uniform(1.5, 4.0)
        a0, m0, w0 = pooling_equilibrium(k, L, g, c_Q, mu_A, lambda_A, lo, hi)
        a1, m1, w1 = pooling_equilibrium(k, L, g + dg, c_Q, mu_A, lambda_A, lo, hi)
        rows.append({
            "k": k, "L": L, "g": g, "c_Q": c_Q, "mu_A": mu_A, "lambda_A": lambda_A, "kappa_lo": lo, "kappa_hi": hi,
            "a_SE_g": a0, "a_SE_g_plus": a1, "E_m_g": m0, "E_m_g_plus": m1, "W_g": w0, "W_g_plus": w1,
            "interior": 0.0 < a0 < 1.0,
            "trap": (m1 > m0 + 1e-9) and (w1 < w0 - 1e-9),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "guessing_trap_incidence.csv")
    df.to_csv(csv_path, index=False)

    share_all = float(df["trap"].mean())
    interior = df[df["interior"]]
    share_interior = float(interior["trap"].mean()) if len(interior) else 0.0
    md_path = os.path.join(output_dir, "guessing_trap_incidence.md")
    with open(md_path, "w") as f:
        f.write(
            "# Guessing Trap Incidence (clipped pooling model)\n\n"
            f"- Draws: {len(df)} (seed {seed}), Delta g = {dg}\n"
            f"- Trap (E[m] rises and W falls): {int(df['trap'].sum())} draws = {100*share_all:.1f}%\n"
            f"- Draws with interior a_SE(g): {len(interior)}; trap among them: {int(interior['trap'].sum())} = {100*share_interior:.1f}%\n"
        )
    return {
        "passed": bool(df["trap"].sum() > 0),
        "share_all": share_all,
        "share_interior": share_interior,
        "n_interior": int(len(interior)),
        "df": df,
        "csv_path": csv_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Trap share: {100*res['share_all']:.1f}% overall, {100*res['share_interior']:.1f}% of {res['n_interior']} interior draws")
