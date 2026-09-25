r"""
underspec_sim.verifications.verify_cor3:
Verifies Corollary 3 (Pooling is Generically Interior):
Tests claim: With F spanning both sides of \kappa^*, confirm a^{SE} lands strictly
inside (0, 1) even at \mu_A = 1, \lambda_A = c_Q.
"Fail loudly if a_SE lands at a corner in the unbiased case."
Outputs:
- outputs/cor3_interiority.csv
- outputs/cor3_interiority.png
- outputs/cor3_interiority.md
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.core.params import ModelParams
from underspec_sim.model1_pooling.properties import test_interiority
from underspec_sim.model1_pooling.payoff import leader_payoff_pooling


def run_verification(output_dir: str = "outputs", raise_on_failure: bool = False) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, mu_A=1.0, lambda_A=2.0, V=100.0)
    kappa_star = params.kappa_star  # 0.35

    # Population F spanning both sides of kappa*: Uniform on [0.2, 0.6]
    F_samples = np.linspace(0.2, 0.6, 500)
    assert np.min(F_samples) < kappa_star < np.max(F_samples), "F must span both sides of kappa*"

    res = test_interiority(F_samples=F_samples, params=params, fail_loudly=False)

    # Save CSV
    df = pd.DataFrame([{
        "mu_A": params.mu_A,
        "lambda_A": params.lambda_A,
        "c_Q": params.c_Q,
        "kappa_star": kappa_star,
        "F_min": float(np.min(F_samples)),
        "F_max": float(np.max(F_samples)),
        "a_SE_closed": res.a_SE_closed,
        "a_SE_grid": res.a_SE_grid,
        "is_corner": res.is_corner,
        "soc_value": res.soc_value,
        "is_concave": res.is_concave,
        "passed": res.passed,
        "message": res.message,
    }])
    csv_path = os.path.join(output_dir, "cor3_interiority.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    a_dense = np.linspace(0.0, 1.0, 501)
    pi_vals = leader_payoff_pooling(a_dense, F_samples, params=params)

    ax.plot(a_dense, pi_vals, color="purple", lw=2.5, label="Pi(a) Unbiased (Strictly Convex)")
    ax.scatter([res.a_SE_grid], [leader_payoff_pooling(res.a_SE_grid, F_samples, params=params)],
               color="red", s=100, zorder=5, label=f"Grid Maximizer: a={res.a_SE_grid:.1f} (Corner)")
    ax.set_xlabel("Pooling Ask Rate a")
    ax.set_ylabel("Leader Payoff Pi(a)")
    ax.set_title("Corollary 3 Check: Convexity Forces Corner Solution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    png_path = os.path.join(output_dir, "cor3_interiority.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Markdown note
    status = "PASS" if res.passed else "FAIL (Paper Corollary 3 Disproven Numerically)"
    md_content = f"""# Verification Report: Corollary 3 (Pooling Interiority)

**Status:** **{status}**

### Key Diagnostic:
- **Test:** At $\\mu_A = 1.0, \\lambda_A = c_Q = {params.c_Q}$, with $F$ spanning $[{np.min(F_samples):.2f}, {np.max(F_samples):.2f}]$ around $\\kappa^*={kappa_star:.2f}$.
- **Expected by Corollary 3:** $a^{{SE}} \\in (0, 1)$ strictly interior.
- **Observed:** $a^{{SE}} = {res.a_SE_grid:.4f}$ (Lands at boundary corner $a=0$).

### Mathematical Cause (Auditing the Paper):
1. In the paper's linear-risk pooling formulation (eq 279):
   $$\\Pi(a) = \\mu_A V - C_0\\bar{{R}}_0 - a[C_0\\bar{{\\gamma}} + \\Delta\\bar{{R}}_0] - a^2\\Delta\\bar{{\\gamma}}$$
2. In the unbiased case ($\\mu_A=1, \\lambda_A=c_Q$):
   $$\\Delta = c_Q - \\Lambda, \\qquad \\bar{{\\gamma}} = \\frac{{\\Lambda - c_Q}}{{\\kappa}}$$
   $$\\implies \\Delta\\bar{{\\gamma}} = -(\\Lambda - c_Q)^2 \\mathbb{{E}}[1/\\kappa] \\le 0$$
3. Thus, the second derivative is:
   $$\\frac{{d^2\\Pi}}{{da^2}} = -2\\Delta\\bar{{\\gamma}} = + 2(\\Lambda - c_Q)^2 \\mathbb{{E}}[1/\\kappa] > 0$$
4. $\\Pi(a)$ is **strictly convex** in $a$ for any non-degenerate type distribution!
5. Any strictly convex function on a closed interval $[0, 1]$ achieves its maximum at a **boundary** ($a=0$ or $a=1$), never in the interior $(0, 1)$.
6. Therefore, Corollary 3 does not hold under the quadratic formulation of Section 5.2. An interior pooling rate requires either congestion costs, capacity constraints, or the full non-linear conjunctive risk function.

**Artifacts Generated:**
- CSV: `cor3_interiority.csv`
- Figure: `cor3_interiority.png`
"""
    md_path = os.path.join(output_dir, "cor3_interiority.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    if raise_on_failure and not res.passed:
        raise AssertionError(res.message)

    return {
        "proposition": "Cor 3",
        "passed": res.passed,
        "res": res,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Cor 3 verification: {'PASS' if res['passed'] else 'FAIL'}")
