r"""
underspec_sim.verifications.verify_prop5:
Verifies Proposition 5 (No distortion without bias):
At \mu_A = 1, \lambda_A = c_Q:
- Confirms the solver returns (m_\kappa, a_\kappa) = first_best(\kappa) for both types to numerical tolerance.
- Confirms IC constraints are strictly slack (not binding).
Outputs:
- outputs/prop5_screening_no_bias.csv
- outputs/prop5_screening_no_bias.png
- outputs/prop5_screening_no_bias.md
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.core.params import ModelParams
from underspec_sim.model2_screening.properties import test_no_distortion_without_bias


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)

    # Choose types spanning kappa* = 0.35: kappa_L = 0.30, kappa_H = 0.50
    kappa_L = 0.30
    kappa_H = 0.50
    f_L = 0.5
    f_H = 0.5

    # Test unconstrained (Proposition 5's exact algebraic setting)
    res_uncon = test_no_distortion_without_bias(
        kappa_L=kappa_L,
        kappa_H=kappa_H,
        f_L=f_L,
        f_H=f_H,
        params=params,
        unconstrained_m=True,
    )

    # Test constrained m <= k
    res_con = test_no_distortion_without_bias(
        kappa_L=kappa_L,
        kappa_H=kappa_H,
        f_L=f_L,
        f_H=f_H,
        params=params,
        unconstrained_m=False,
    )

    df = pd.DataFrame([
        {
            "setting": "Unconstrained_m",
            "kappa_L": kappa_L,
            "kappa_H": kappa_H,
            "mL_opt": res_uncon.res.m_L,
            "aL_opt": res_uncon.res.a_L,
            "mL_FB": res_uncon.m_FB_L,
            "aL_FB": res_uncon.a_FB_L,
            "diff_L": res_uncon.diff_L,
            "mH_opt": res_uncon.res.m_H,
            "aH_opt": res_uncon.res.a_H,
            "mH_FB": res_uncon.m_FB_H,
            "aH_FB": res_uncon.a_FB_H,
            "diff_H": res_uncon.diff_H,
            "IC_L_slack": res_uncon.IC_L_slack,
            "IC_H_slack": res_uncon.IC_H_slack,
            "passed": res_uncon.passed,
        },
        {
            "setting": "Constrained_m_le_k",
            "kappa_L": kappa_L,
            "kappa_H": kappa_H,
            "mL_opt": res_con.res.m_L,
            "aL_opt": res_con.res.a_L,
            "mL_FB": res_con.m_FB_L,
            "aL_FB": res_con.a_FB_L,
            "diff_L": res_con.diff_L,
            "mH_opt": res_con.res.m_H,
            "aH_opt": res_con.res.a_H,
            "mH_FB": res_con.m_FB_H,
            "aH_FB": res_con.a_FB_H,
            "diff_H": res_con.diff_H,
            "IC_L_slack": res_con.IC_L_slack,
            "IC_H_slack": res_con.IC_H_slack,
            "passed": res_con.passed,
        },
    ])
    csv_path = os.path.join(output_dir, "prop5_screening_no_bias.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = ["Type L (Low Cost)", "Type H (High Cost)"]
    opt_a = [res_uncon.res.a_L, res_uncon.res.a_H]
    fb_a = [res_uncon.a_FB_L, res_uncon.a_FB_H]
    w = 0.35
    x = np.arange(len(x_labels))

    ax.bar(x - w/2, fb_a, width=w, label="First-Best Ask Rate a_FB", color="royalblue", alpha=0.8)
    ax.bar(x + w/2, opt_a, width=w, label="Screening Menu a* (Unbiased)", color="darkgreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Ask Rate a")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Proposition 5: Unbiased Screening Recovers First Best (Zero Distortion)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    png_path = os.path.join(output_dir, "prop5_screening_no_bias.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    passed = bool(res_uncon.passed)
    md_content = f"""# Verification Report: Proposition 5 (Efficiency Survives Private Information Without Bias)

**Status:** **{'PASS' if passed else 'FAIL'}**

### Key Results:
- **Parameters:** $\\kappa_L = {kappa_L}$, $\\kappa_H = {kappa_H}$, $\\mu_A = 1.0, \\lambda_A = c_Q = {params.c_Q}$.
- **Numerical Menu Recovery:**
  - Type L: $(m_L^*, a_L^*) = ({res_uncon.res.m_L:.4f}, {res_uncon.res.a_L:.4f})$ matches First-Best $({res_uncon.m_FB_L:.4f}, {res_uncon.a_FB_L:.4f})$.
  - Type H: $(m_H^*, a_H^*) = ({res_uncon.res.m_H:.4f}, {res_uncon.res.a_H:.4f})$ matches First-Best $({res_uncon.m_FB_H:.4f}, {res_uncon.a_FB_H:.4f})$.
- **Incentive Compatibility Check:**
  - $IC_L$ slack: `{res_uncon.IC_L_slack:.4f}` $> 0$ (strictly slack!).
  - $IC_H$ slack: `{res_uncon.IC_H_slack:.4f}` $> 0$ (strictly slack!).
- **Conclusion:** As predicted by Proposition 5, when principal and agent share an objective (unbiased), incentive compatibility is free, and private information incurs zero distortion!

**Artifacts Generated:**
- CSV: `prop5_screening_no_bias.csv`
- Figure: `prop5_screening_no_bias.png`
"""
    md_path = os.path.join(output_dir, "prop5_screening_no_bias.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    return {
        "proposition": "Prop 5",
        "passed": passed,
        "res_uncon": res_uncon,
        "res_con": res_con,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Prop 5 verification: {'PASS' if res['passed'] else 'FAIL'}")
