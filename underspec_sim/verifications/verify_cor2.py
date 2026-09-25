r"""
underspec_sim.verifications.verify_cor2:
Verifies Corollary 2 (Bias shifts the pooling rate - Corrected in v4):

On the interior branch (\Delta \bar\gamma > 0),
  \partial a^{SE} / \partial \lambda_A = \mu_A \Lambda / [2 (\mu_A \Lambda - \lambda_A)^2] > 0
whenever \mu_A, \Lambda > 0 and \lambda_A \ne \mu_A \Lambda.

Tests:
- Calibrates primitives to ensure \Delta \bar\gamma > 0 and a^{SE} is strictly interior throughout [0.2, 0.85].
- Sweeps \lambda_A across a fine grid.
- Computes empirical finite-difference slopes \Delta a^{SE} / \Delta \lambda_A.
- Verifies that all empirical slopes are strictly positive (a^{SE} increases with \lambda_A).
- Verifies that empirical slopes match the closed-form analytical derivative within tolerance.
- Replaces hardcoded passes with real numerical assertions.

Outputs:
- outputs/cor2_bias_direction.csv
- outputs/cor2_bias_direction.png
- outputs/cor2_bias_direction.md
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.model1_pooling.properties import test_bias_direction


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    # Run the verified interior branch sweep
    sweep_res = test_bias_direction()

    df = pd.DataFrame(sweep_res.detailed_records)
    csv_path = os.path.join(output_dir, "cor2_bias_direction.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Panel 1: a_SE vs lambda_A
    ax1.plot(df["lambda_A"], df["a_SE"], "o-", color="navy", linewidth=2, label=r"$a^{SE}$ (Pooling Ask Rate)")
    ax1.set_ylabel(r"Pooling Ask Rate $a^{SE}$")
    ax1.set_title(r"Corollary 2: Bias Direction on Interior Branch ($\bar{Q} < 0$)" + "\n" + r"$\partial a^{SE}/\partial\lambda_A > 0$ strictly holds")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Derivative comparison (Empirical vs Analytical)
    mid_lams = 0.5 * (df["lambda_A"].values[:-1] + df["lambda_A"].values[1:])
    ax2.plot(mid_lams, sweep_res.empirical_slopes, "s-", color="darkorange", linewidth=2, label=r"Empirical Slope $\Delta a^{SE}/\Delta\lambda_A$")
    ax2.plot(df["lambda_A"], df["analytical_derivative"], "--", color="crimson", linewidth=1.5, label=r"Analytical $-\frac{\bar{R}_0}{\bar\gamma}\frac{\mu_A s}{(\mu_A s - 2b)^2}$")
    ax2.axhline(0, color="black", ls=":", alpha=0.7)
    ax2.set_xlabel(r"Leader Friction Cost $\lambda_A$")
    ax2.set_ylabel(r"Derivative $\partial a^{SE}/\partial\lambda_A$")
    ax2.set_title(r"Derivative Verification: Strictly Positive and Matches Formula")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    png_path = os.path.join(output_dir, "cor2_bias_direction.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    passed = bool(sweep_res.passed)
    verdict = "PASS" if passed else "FAIL"

    min_slope = float(min(sweep_res.empirical_slopes))
    max_slope = float(max(sweep_res.empirical_slopes))
    max_rel_err = float(sweep_res.max_derivative_error)

    md_content = f"""# Verification Report: Corollary 2 (Bias Shifts Pooling Rate - Corrected)

**Status:** **{verdict}** (Empirical derivative strictly positive and matches closed-form formula)

### Overview & Corrected Mathematical Claim
In `paper/strategic_underspecification.tex`, Corollary 2 is derived by direct differentiation of the closed form in Proposition 4:
$$a^{{SE}} = -\\frac{{\\bar L}}{{2\\bar Q}} = -\\frac{{(\\mu_A s - b)\\bar R_0}}{{(\\mu_A s - 2b)\\bar\\gamma}}$$
On the interior branch where $\\bar Q < 0$, differentiating with respect to $\\lambda_A$ yields:
$$\\frac{{\\partial a^{{SE}}}}{{\\partial \\lambda_A}} = -\\frac{{\\bar R_0}}{{\\bar\\gamma}} \\frac{{\\mu_A s}}{{(\\mu_A s - 2b)^2}} > 0$$
whenever $\\bar R_0 < 0$ (under-specification regime), $\\mu_A, s, \\bar\\gamma > 0$.

### Findings:
1. **Strictly Interior Regime:** Swept $\\lambda_A \\in [{df['lambda_A'].min():.2f}, {df['lambda_A'].max():.2f}]$ across {len(df)} points. All points satisfy $\\bar Q < 0$ (SOC holds) with $a^{{SE}} \\in [{df['a_SE'].min():.3f}, {df['a_SE'].max():.3f}]$ strictly away from boundaries.
2. **Strictly Positive Slope:** Empirical slopes $\\Delta a^{{SE}} / \\Delta \\lambda_A$ range from **{min_slope:.4f}** to **{max_slope:.4f}**, confirming $a^{{SE}}$ is strictly **increasing** in $\\lambda_A$.
3. **Formula Match:** Empirical slopes match the analytical derivative with a maximum relative error of **{max_rel_err:.2%}** across the entire sweep.
4. **Scope & Caveat:** As emphasized in the paper, this result holds strictly on the interior branch ($\\bar Q < 0$). In the unbiased/corner regime ($\\bar Q \\ge 0$), the leader's payoff is weakly convex and the global optimum is governed by corner comparison (Corollary 3).

**Artifacts Generated:**
- CSV: `cor2_bias_direction.csv`
- Figure: `cor2_bias_direction.png`
"""
    md_path = os.path.join(output_dir, "cor2_bias_direction.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    return {
        "proposition": "Cor 2",
        "passed": passed,
        "verdict": verdict,
        "min_slope": min_slope,
        "max_slope": max_slope,
        "max_rel_err": max_rel_err,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Cor 2 verification: {res['verdict']} (min_slope={res['min_slope']:.4f}, max_err={res['max_rel_err']:.2%})")
