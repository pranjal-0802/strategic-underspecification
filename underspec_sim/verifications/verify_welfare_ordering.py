r"""
underspec_sim.verifications.verify_welfare_ordering:
Verifies User Welfare Ordering across Pooling and Screening Regimes:
Evaluates user welfare W = \mathbb{E}[U] and leader payoff \Pi across bias bands:
Parameters: k=10, L=10, c_Q=2, g=0.5 (s=3), V=100, \kappa_L=0.3, \kappa_H=0.5, f_L=0.5, \mu_A=1.0.

Regimes:
1. Mild Bias (\lambda_A in [2.5, 3.0]):
   - Pooling maintains universal clarification (a^{SE} = 1.0, W_pool = 85.33).
   - Screening distorts a_H downward (a_H < 1.0, W_screen in [84.52, 85.00]).
   - Result: Pooling achieves HIGHER user welfare than screening (W_screen - W_pool < 0).
2. Moderate Bias (\lambda_A in [3.5, 4.5]):
   - Pooling collapses to zero asking (a^{SE} = 0.0, W_pool = 80.00).
   - Screening preserves partial clarification for high-cost types (a_H in [0.70, 0.81], W_screen in [80.71, 82.45]).
   - Result: Screening achieves HIGHER user welfare than pooling (W_screen - W_pool > 0).
3. Severe / Prohibitive Friction (\lambda_A >= 5.0, i.e. \tilde\rho >= s):
   - Asking is universally priced out by severe friction (c_Q^{eff} >= \Lambda).
   - Both pooling and screening shut down clarification (a = 0 everywhere, W_pool = W_screen = 80.00).
   - Result: Welfare gap is exactly zero (W_screen - W_pool = 0.00).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import user_utility
from underspec_sim.model2_screening.solver import solve_menu


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    k, c_Q, Lam = params.k, params.c_Q, params.Lambda
    kappa_L, kappa_H, f_L = 0.3, 0.5, 0.5
    f_H = 1.0 - f_L
    mu_A = 1.0

    def compute_pooling(lam_A):
        best = None
        for a in np.linspace(0, 1, 2001):
            pi_val = 0.0
            w_val = 0.0
            for x, f in ((kappa_L, f_L), (kappa_H, f_H)):
                m = min(k, (Lam * (1.0 - a) + a * c_Q) / x)
                u = float(user_utility(m, a, x, params))
                pi_val += f * (mu_A * u - (lam_A - c_Q) * a * (k - m))
                w_val += f * u
            if best is None or pi_val > best[0] + 1e-12:
                best = (pi_val, w_val, a)
        return best[0], best[1], best[2]

    lambda_vals = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    rows = []

    for lam_A in lambda_vals:
        pi_p, w_p, a_p = compute_pooling(lam_A)
        r = solve_menu(kappa_L, kappa_H, f_L, f_H, mu_A=mu_A, lambda_A=lam_A, c_Q=c_Q, params=params, unconstrained_m=False)
        w_s = f_L * float(user_utility(r.m_L, r.a_L, kappa_L, params)) + f_H * float(user_utility(r.m_H, r.a_H, kappa_H, params))
        pi_s = r.leader_payoff
        rows.append({
            "lambda_A": lam_A,
            "rho_tilde": (lam_A - c_Q) / mu_A,
            "Pi_pool": pi_p,
            "W_pool": w_p,
            "a_pool": a_p,
            "Pi_screen": pi_s,
            "W_screen": w_s,
            "a_L_screen": r.a_L,
            "a_H_screen": r.a_H,
            "W_gap": w_s - w_p,
            "Pi_gap": pi_s - pi_p,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "welfare_ordering.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # User welfare comparison
    ax1.plot(df["lambda_A"], df["W_pool"], "b-o", label="Pooling User Welfare ($W_{\\text{pool}}$)")
    ax1.plot(df["lambda_A"], df["W_screen"], "r--s", label="Screening User Welfare ($W_{\\text{screen}}$)")
    ax1.axvline(3.25, color="gray", linestyle=":", label="Regime Boundary")
    ax1.axvline(5.0, color="k", linestyle="--", label=r"Shutdown ($\tilde\rho \geq s$)")
    ax1.set_xlabel(r"Assistant Asking Friction $\lambda_A$")
    ax1.set_ylabel(r"User Welfare $W = \mathbb{E}[U]$")
    ax1.set_title(r"User Welfare: Non-Monotonic Regime Ranking")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Welfare gap
    ax2.plot(df["lambda_A"], df["W_gap"], "g-^", label=r"Welfare Gap ($W_{\mathrm{screen}} - W_{\mathrm{pool}}$)")
    ax2.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax2.axvline(5.0, color="k", linestyle="--", label=r"Shutdown ($\tilde\rho \geq s$)")
    ax2.set_xlabel("Assistant Asking Friction $\\lambda_A$")
    ax2.set_ylabel("Welfare Difference")
    ax2.set_title("Screening Welfare Premium ($W_{\\text{screen}} - W_{\\text{pool}}$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "welfare_ordering.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Markdown report
    md_path = os.path.join(output_dir, "welfare_ordering.md")
    passed = (
        df.loc[df["lambda_A"] == 2.5, "W_gap"].values[0] < -0.1
        and df.loc[df["lambda_A"] == 3.5, "W_gap"].values[0] > 0.5
        and abs(df.loc[df["lambda_A"] == 5.0, "W_gap"].values[0]) < 1e-4
    )

    with open(md_path, "w") as f:
        f.write(f"""# Verification Report: User Welfare Ordering Across Regimes

**Status:** **{'PASS' if passed else 'FAIL'}**

### Key Results Across Friction Bands:
1. **Mild Bias ($\\lambda_A \\in [2.5, 3.0]$):**
   - At $\\lambda_A = 2.5$: $W_{{pool}} = {df.loc[df['lambda_A']==2.5, 'W_pool'].values[0]:.2f}$, $W_{{screen}} = {df.loc[df['lambda_A']==2.5, 'W_screen'].values[0]:.2f}$ (Gap: ${df.loc[df['lambda_A']==2.5, 'W_gap'].values[0]:.2f}$). Pooling delivers higher user welfare because it preserves universal clarification ($a=1.0$).
   - At $\\lambda_A = 3.0$: $W_{{pool}} = {df.loc[df['lambda_A']==3.0, 'W_pool'].values[0]:.2f}$, $W_{{screen}} = {df.loc[df['lambda_A']==3.0, 'W_screen'].values[0]:.2f}$ (Gap: ${df.loc[df['lambda_A']==3.0, 'W_gap'].values[0]:.2f}$).
2. **Moderate Bias ($\\lambda_A \\in [3.5, 4.5]$):**
   - At $\\lambda_A = 3.5$: $W_{{pool}} = {df.loc[df['lambda_A']==3.5, 'W_pool'].values[0]:.2f}$, $W_{{screen}} = {df.loc[df['lambda_A']==3.5, 'W_screen'].values[0]:.2f}$ (Gap: $+{df.loc[df['lambda_A']==3.5, 'W_gap'].values[0]:.2f}$). Screening delivers higher user welfare because pooling collapses to zero asking ($a=0$), while screening preserves partial clarification for high-cost types ($a_H > 0$).
   - At $\\lambda_A = 4.0$: $W_{{pool}} = {df.loc[df['lambda_A']==4.0, 'W_pool'].values[0]:.2f}$, $W_{{screen}} = {df.loc[df['lambda_A']==4.0, 'W_screen'].values[0]:.2f}$ (Gap: $+{df.loc[df['lambda_A']==4.0, 'W_gap'].values[0]:.2f}$).
3. **Severe / Prohibitive Friction ($\\lambda_A \\ge 5.0$, where $\\tilde\\rho \\ge s = 3$):**
   - At $\\lambda_A = 5.0$: $W_{{pool}} = {df.loc[df['lambda_A']==5.0, 'W_pool'].values[0]:.2f}$, $W_{{screen}} = {df.loc[df['lambda_A']==5.0, 'W_screen'].values[0]:.2f}$ (Gap: ${df.loc[df['lambda_A']==5.0, 'W_gap'].values[0]:.2f}$). Both regimes shut down asking completely ($a = 0$), so user welfare ties at $80.00$.
""")

    return {
        "passed": bool(passed),
        "df": df,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print("User Welfare Ordering Verification:", "PASS" if res["passed"] else "FAIL")
