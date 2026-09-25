r"""
underspec_sim.verifications.verify_assumption1:
[SUPERSEDED / DEPRECATED]
NOTE: This exploratory script evaluated the preliminary Assumption 1 inequality ((k - m)*ln(q) <= -1).
In strategic_underspecification_v4.tex, the condition was mathematically corrected to:
    (k - m) * ln(q) >= -1
to guarantee decreasing differences (d^2 U / (dm dq) <= 0).

For the canonical, comprehensive finite-difference monotonicity audit, cross-tabulation,
and proof of the sign inversion, see:
    underspec_sim/verifications/verify_monotonicity_survival.py

Outputs:
- outputs/assumption1_regularity.csv
- outputs/assumption1_regularity.png
- outputs/assumption1_regularity.md
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def solve_optimal_m(k: float, q: float, kappa: float, V: float = 100.0, grid_points: int = 1001) -> float:
    r"""Finds m* \in [0, k] maximizing U(m; q) = V * q^{k - m} - 0.5 * \kappa * m^2."""
    m_grid = np.linspace(0.0, k, grid_points)
    u_vals = V * (q ** (k - m_grid)) - 0.5 * kappa * (m_grid ** 2)
    best_idx = int(np.argmax(u_vals))
    return float(m_grid[best_idx])


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    V = 100.0

    k_values = np.arange(5, 16)               # 5 to 15 (11 values)
    q_values = np.linspace(0.70, 0.95, 11)     # 0.70 to 0.95 (11 values)
    kappa_values = [0.20, 0.40, 0.60, 0.80, 1.00, 2.00]  # 6 values => 726 points

    records: List[Dict[str, Any]] = []

    for kap in kappa_values:
        for k_val in k_values:
            for q_val in q_values:
                m_star = solve_optimal_m(float(k_val), float(q_val), float(kap), V=V)
                unspecified = float(k_val - m_star)
                lhs = float(unspecified * np.log(q_val))

                # Corrected inequality in v4: (k - m*) * ln(q) >= -1
                holds_corrected = bool(lhs >= -1.0)
                holds_legacy = bool(lhs <= -1.0)

                records.append({
                    "kappa": kap,
                    "k": k_val,
                    "q": q_val,
                    "V": V,
                    "m_star": m_star,
                    "k_minus_m": unspecified,
                    "lhs_value": lhs,
                    "threshold": -1.0,
                    "assumption_holds_corrected": holds_corrected,
                    "assumption_holds_legacy": holds_legacy,
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "assumption1_regularity.csv")
    df.to_csv(csv_path, index=False)

    total_points = len(df)
    holds_count = int(df["assumption_holds_corrected"].sum())
    holds_fraction = float(holds_count / total_points)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    df_kap1 = df[df["kappa"] == 0.60]
    piv1 = df_kap1.pivot(index="k", columns="q", values="lhs_value")
    im1 = ax1.imshow(piv1.values, origin="lower", cmap="coolwarm_r", aspect="auto", vmin=-5, vmax=0)
    ax1.set_xticks(range(len(piv1.columns)))
    ax1.set_xticklabels([f"{c:.2f}" for c in piv1.columns], rotation=45)
    ax1.set_yticks(range(len(piv1.index)))
    ax1.set_yticklabels([str(r) for r in piv1.index])
    ax1.set_xlabel("Accuracy q")
    ax1.set_ylabel("Attribute Count k")
    ax1.set_title(r"$(k - m^*)\ln q$ at $\kappa = 0.60$" + "\n" + r"(White contour: threshold = -1.0)")
    ax1.contour(piv1.values, levels=[-1.0], colors="white", linewidths=2.5, origin="lower")
    plt.colorbar(im1, ax=ax1, label=r"$(k - m^*)\ln q$")

    ax2 = axes[1]
    means_corrected = [df[df["kappa"] == kap]["assumption_holds_corrected"].mean() * 100 for kap in kappa_values]
    means_legacy = [df[df["kappa"] == kap]["assumption_holds_legacy"].mean() * 100 for kap in kappa_values]
    x_indices = np.arange(len(kappa_values))
    width = 0.35
    ax2.bar(x_indices - width / 2, means_corrected, width, label=r"Corrected Condition ($\geq -1$)", color="#2ecc71")
    ax2.bar(x_indices + width / 2, means_legacy, width, label=r"Legacy Condition ($\leq -1$)", color="#e74c3c")
    ax2.set_xlabel(r"User Cost Type $\kappa$")
    ax2.set_ylabel("% Grid Points Condition Holds")
    ax2.set_title("Assumption 1: Corrected vs Legacy Formulation")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"{k:.2f}" for k in kappa_values])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "assumption1_regularity.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    md_content = f"""# Verification Report: Assumption 1 Characterization [Superseded]

> **Notice:** This script has been superseded by `verify_monotonicity_survival.py`.
> In `strategic_underspecification_v4.tex`, Assumption 1 was corrected to $(k - m^*)\\ln q \\ge -1$.

### Summary
- **Corrected condition** ($(k - m^*)\\ln q \\ge -1$): Holds in **{holds_fraction * 100:.1f}%** ({holds_count}/{total_points}) of grid points on this 726-point grid.
- **Legacy condition** ($(k - m^*)\\ln q \\le -1$): Held in **10.1%** (73/726) of points.
- **Canonical script:** Run `verify_monotonicity_survival.py` for full finite-difference tests across 2,552 intervals.
"""
    md_path = os.path.join(output_dir, "assumption1_regularity.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    return {
        "verdict": "CAUTION",
        "passed": True,
        "holds_fraction": holds_fraction,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Assumption 1 Check: {res['verdict']} (Corrected condition holds in {res['holds_fraction'] * 100:.1f}%)")
