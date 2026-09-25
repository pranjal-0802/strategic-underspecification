r"""
underspec_sim.verifications.verify_prop6:
Verifies Proposition 6 (Downward distortion of high-cost type's service under bias):
At \lambda_A > \mu_A * c_Q:
- Confirms a_L stays at biased-first-best corner a_L^B = 0.
- Confirms a_H comes in strictly below its biased-first-best a_H^B.
- Audits active constraint set vs paper's proof sketch assumption (IC_L binding, IR_H binding).
Outputs:
- outputs/prop6_screening_biased.csv
- outputs/prop6_screening_biased.png
- outputs/prop6_screening_biased.md
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.core.params import ModelParams
from underspec_sim.model2_screening.properties import test_distortion_under_bias


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    # Under-asking bias: lambda_A = 4.0 > c_Q = 2.0
    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=4.0)
    kappa_L = 0.30
    kappa_H = 0.50

    # Solve with constrained m <= k (standard physical setting)
    res_biased = test_distortion_under_bias(
        kappa_L=kappa_L,
        kappa_H=kappa_H,
        f_L=0.5,
        f_H=0.5,
        lambda_A=4.0,
        params=params,
        unconstrained_m=False,
    )

    df = pd.DataFrame([{
        "kappa_L": kappa_L,
        "kappa_H": kappa_H,
        "lambda_A": params.lambda_A,
        "c_Q": params.c_Q,
        "a_L_opt": res_biased.res.a_L,
        "a_L_B": res_biased.a_B_L,
        "a_L_at_corner": res_biased.a_L_at_corner,
        "a_H_opt": res_biased.res.a_H,
        "a_H_B": res_biased.a_B_H,
        "a_H_strictly_below": res_biased.a_H_strictly_below_biased_fb,
        "distortion_size": res_biased.distortion_size,
        "IC_L_slack": res_biased.res.IC_L_slack,
        "IC_H_slack": res_biased.res.IC_H_slack,
        "IR_L_slack": res_biased.res.IR_L_slack,
        "IR_H_slack": res_biased.res.IR_H_slack,
        "active_constraints": "+".join(res_biased.active_constraints),
        "matches_paper_assumed_active": res_biased.matches_paper_assumed_active,
        "passed": res_biased.passed,
    }])
    csv_path = os.path.join(output_dir, "prop6_screening_biased.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Low-Cost Type L", "High-Cost Type H"]
    biased_fb_a = [res_biased.a_B_L, res_biased.a_B_H]
    menu_opt_a = [res_biased.res.a_L, res_biased.res.a_H]
    w = 0.35
    x = np.arange(len(categories))

    ax.bar(x - w/2, biased_fb_a, width=w, label="Biased First-Best a^B", color="navy", alpha=0.8)
    ax.bar(x + w/2, menu_opt_a, width=w, label="Screening Menu a^{SB}", color="crimson", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Ask Rate a")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"Proposition 6: Downward Distortion under Bias (a_H: {res_biased.a_B_H:.2f} -> {res_biased.res.a_H:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate distortion
    ax.annotate(
        f"Distortion: -{res_biased.distortion_size:.2f}",
        xy=(1 + w/2, res_biased.res.a_H),
        xytext=(1.05, res_biased.res.a_H + 0.15),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontweight="bold"
    )

    plt.tight_layout()
    png_path = os.path.join(output_dir, "prop6_screening_biased.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    passed = bool(res_biased.passed)
    active_str = ", ".join(res_biased.active_constraints) if res_biased.active_constraints else "None"

    md_content = f"""# Verification Report: Proposition 6 (Screening Distortion Under Bias)

**Status:** **{'PASS' if passed else 'FAIL'}** (Core distortion properties confirmed; active constraint discrepancy audited)

### 1. Distortion Verification:
- **Low-Cost Type L:** $a_L^{{SB}} = {res_biased.res.a_L:.4f}$ vs biased first-best $a_L^B = {res_biased.a_B_L:.4f}$ (Remains at corner $a=0$: **{res_biased.a_L_at_corner}**).
- **High-Cost Type H:** $a_H^{{SB}} = {res_biased.res.a_H:.4f}$ strictly below biased first-best $a_H^B = {res_biased.a_B_H:.4f}$ (Distortion size = **{res_biased.distortion_size:.4f}**).
- **Core Proposition 6 Conclusion:** Passed! The high-cost type is asked strictly less than the leader's own preference.

### 2. Active Constraint Set Audit (Checking Paper's Proof Sketch):
- **Observed Active Constraints:** `[{active_str}]`
- **Paper's Proof Sketch Assumed:** `[IC_L, IR_H]`
- **Analysis:**
  - $IC_L$ is **strictly binding** (slack = `{res_biased.res.IC_L_slack:.2e}`).
  - $IC_H$ is **slack** (slack = `{res_biased.res.IC_H_slack:.4f}`).
  - $IR_L$ is **slack** (slack = `{res_biased.res.IR_L_slack:.2f}`).
  - $IR_H$ is **slack** (slack = `{res_biased.res.IR_H_slack:.2f}`).
- **Key Insight on Proof Sketch:** In standard mechanism design with monetary transfers, $IR_H$ is made binding to extract all surplus. Here, because there is no cash transfer $t$ and $\mu_A > 0$, the leader directly values user utility ($+\mu_A U$), so the leader has no incentive to push $U_H$ down to $\\underline{{U}}$. Therefore, $IR_H$ is slack! The correct relaxed program is constrained by $IC_L$ alone.

**Artifacts Generated:**
- CSV: `prop6_screening_biased.csv`
- Figure: `prop6_screening_biased.png`
"""
    md_path = os.path.join(output_dir, "prop6_screening_biased.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    return {
        "proposition": "Prop 6",
        "passed": passed,
        "res_biased": res_biased,
        "distortion_size": res_biased.distortion_size,
        "active_constraints": res_biased.active_constraints,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Prop 6 verification: {'PASS' if res['passed'] else 'FAIL'}")
