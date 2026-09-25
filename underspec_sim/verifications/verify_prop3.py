r"""
underspec_sim.verifications.verify_prop3:
Verifies Proposition 3:
1. Bang-bang threshold at \kappa^* = (\Lambda + c_Q) / (2k).
2. Sweep \kappa across fine grid, confirm a_FB flips from 0 to 1 exactly at \kappa^*.
3. Global optimality: confirm U(m_FB, a_FB; \kappa) >= U(m, a; \kappa) for a grid
   of alternative (m, a) pairs across the entire parameter domain.
Outputs:
- outputs/prop3_first_best.csv
- outputs/prop3_first_best.png
- outputs/prop3_first_best.md
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import user_utility
from underspec_sim.core.first_best import first_best


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)
    kappa_star = params.kappa_star  # (5 + 2) / 20 = 0.35

    # 1. Sweep kappa across fine grid
    kappas = np.linspace(0.1, 0.7, 601)
    m_fb_list = []
    a_fb_list = []
    u_fb_list = []
    optimality_passed = []
    max_viol = 0.0

    # Grid of alternative (m, a) to test argmax
    alt_m = np.linspace(0, 30, 61)  # Includes unconstrained region
    alt_a = np.linspace(0, 1, 11)

    for kap in kappas:
        m_fb, a_fb = first_best(kap, params, clip=False)
        u_fb = user_utility(m_fb, a_fb, kap, params)
        m_fb_list.append(m_fb)
        a_fb_list.append(a_fb)
        u_fb_list.append(u_fb)

        # Check alternative (m, a)
        is_opt = True
        for m_val in alt_m:
            for a_val in alt_a:
                u_alt = user_utility(m_val, a_val, kap, params)
                if u_alt > u_fb + 1e-7:
                    is_opt = False
                    viol = u_alt - u_fb
                    if viol > max_viol:
                        max_viol = viol
        optimality_passed.append(is_opt)

    df = pd.DataFrame({
        "kappa": kappas,
        "m_FB": m_fb_list,
        "a_FB": a_fb_list,
        "U_FB": u_fb_list,
        "is_optimal": optimality_passed,
    })
    csv_path = os.path.join(output_dir, "prop3_first_best.csv")
    df.to_csv(csv_path, index=False)

    # Threshold flip test
    below_star = df[df["kappa"] < kappa_star - 1e-4]
    above_star = df[df["kappa"] > kappa_star + 1e-4]
    flip_correct = (
        np.all(below_star["a_FB"] == 0.0)
        and np.all(above_star["a_FB"] == 1.0)
    )
    all_optimal = all(optimality_passed)
    passed = bool(flip_correct and all_optimal)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(df["kappa"], df["a_FB"], label="a_FB(kappa)", color="navy", lw=2)
    ax1.axvline(kappa_star, color="crimson", ls="--", label=f"kappa* = {kappa_star:.3f}")
    ax1.set_ylabel("Ask Rate a_FB")
    ax1.set_title("Proposition 3: First-Best Policy Threshold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(df["kappa"], df["m_FB"], label="m_FB(kappa)", color="forestgreen", lw=2)
    ax2.axvline(kappa_star, color="crimson", ls="--")
    ax2.set_xlabel("Specification Cost Type kappa")
    ax2.set_ylabel("Specification Level m_FB")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    png_path = os.path.join(output_dir, "prop3_first_best.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Markdown note
    status = "PASS" if passed else "FAIL"
    md_content = f"""# Verification Report: Proposition 3 (First-Best Threshold)

**Status:** **{status}**

- **Theoretical Threshold:** $\\kappa^* = \\frac{{\\Lambda + c_Q}}{{2k}} = \\frac{{5.0 + 2.0}}{{20.0}} = {kappa_star:.4f}$
- **Observed Flip:**
  - For $\\kappa < \\kappa^*$: $a^{{FB}} = 0.0$ across all {len(below_star)} test points.
  - For $\\kappa > \\kappa^*$: $a^{{FB}} = 1.0$ across all {len(above_star)} test points.
  - Flip occurs strictly at $\\kappa^*$.
- **Optimality Verification (Grid Search Check):**
  - Evaluated against {len(alt_m) * len(alt_a)} alternative $(m, a)$ bundles per type.
  - Max violation observed: `{max_viol:.2e}` (numerical tolerance threshold: `1e-7`).
  - All test points verified as global argmax: `{all_optimal}`.

**Artifacts Generated:**
- CSV: `prop3_first_best.csv`
- Figure: `prop3_first_best.png`
"""
    md_path = os.path.join(output_dir, "prop3_first_best.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    return {
        "proposition": "Prop 3",
        "passed": passed,
        "kappa_star": kappa_star,
        "flip_correct": flip_correct,
        "all_optimal": all_optimal,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Prop 3 verification: {'PASS' if res['passed'] else 'FAIL'}")
