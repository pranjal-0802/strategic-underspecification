r"""
underspec_sim.verifications.sweep_distortion:
2D sweep of heterogeneity (\kappa_H - \kappa_L) and bias (\lambda_A - c_Q).
Tests:
- Whether distortion shrinks as heterogeneity shrinks.
- Separability / monotonicity across heterogeneity and bias dimensions.
- Welfare-gap heatmaps reflecting true pooling payoffs with corner/interior regime flagged.

Outputs:
- outputs/distortion_sweep_2d.csv
- outputs/distortion_sweep_2d.png
- outputs/distortion_sweep_2d.md
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.core.params import ModelParams
from underspec_sim.model2_screening.properties import sweep_distortion_vs_heterogeneity


def run_verification(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0)

    kappa_L = 0.30
    kappa_H_values = [0.35, 0.40, 0.45, 0.50, 0.60]
    bias_values = [1.0, 2.0, 3.0, 4.0]

    df = sweep_distortion_vs_heterogeneity(
        params=params,
        kappa_L=kappa_L,
        kappa_H_values=kappa_H_values,
        bias_values=bias_values,
        unconstrained_m=False,
    )

    csv_path = os.path.join(output_dir, "distortion_sweep_2d.csv")
    df.to_csv(csv_path, index=False)

    # Plot: line panel on top, two heatmaps below (legible at page width)
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel 1: Distortion vs delta_kappa for each bias level
    for b in bias_values:
        sub = df[df["bias"] == b].sort_values("delta_kappa")
        ax1.plot(sub["delta_kappa"], sub["distortion"], "o-", label=f"Bias = {b:.1f}")
    ax1.set_xlabel(r"Heterogeneity $\Delta\kappa = \kappa_H - \kappa_L$")
    ax1.set_ylabel(r"Downward Distortion ($a_H^B - a_H^{SB}$)")
    ax1.set_title("Distortion vs Heterogeneity Across Bias")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Heatmap of distortion
    pivot_dist = df.pivot(index="bias", columns="delta_kappa", values="distortion")
    im2 = ax2.imshow(pivot_dist.values, origin="lower", cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(pivot_dist.columns)))
    ax2.set_xticklabels([f"{c:.2f}" for c in pivot_dist.columns])
    ax2.set_yticks(range(len(pivot_dist.index)))
    ax2.set_yticklabels([f"{r:.1f}" for r in pivot_dist.index])
    ax2.set_xlabel(r"Heterogeneity $\Delta\kappa$")
    ax2.set_ylabel(r"Bias ($\lambda_A - c_Q$)")
    ax2.set_title("2D Grid: Distortion Magnitude")
    plt.colorbar(im2, ax=ax2, label="Distortion")

    # Annotate distortion values
    for i in range(len(pivot_dist.index)):
        for j in range(len(pivot_dist.columns)):
            val = pivot_dist.values[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=12)

    # Panel 3: Heatmap of welfare gap (Pi_II - Pi_I) with pooling regime flagged
    pivot_gap = df.pivot(index="bias", columns="delta_kappa", values="payoff_gap")
    pivot_regime = df.pivot(index="bias", columns="delta_kappa", values="pooling_regime")
    im3 = ax3.imshow(pivot_gap.values, origin="lower", cmap="viridis", aspect="auto")
    ax3.set_xticks(range(len(pivot_gap.columns)))
    ax3.set_xticklabels([f"{c:.2f}" for c in pivot_gap.columns])
    ax3.set_yticks(range(len(pivot_gap.index)))
    ax3.set_yticklabels([f"{r:.1f}" for r in pivot_gap.index])
    ax3.set_xlabel(r"Heterogeneity $\Delta\kappa$")
    ax3.set_ylabel(r"Bias ($\lambda_A - c_Q$)")
    ax3.set_title(r"Leader-Payoff Gap ($\Pi_{II} - \Pi_I$), Pooling Regime C/I")
    plt.colorbar(im3, ax=ax3, label=r"$\Pi_{II} - \Pi_I$")

    # Annotate gap and regime ("C" for corner, "I" for interior)
    for i in range(len(pivot_gap.index)):
        for j in range(len(pivot_gap.columns)):
            gap_val = pivot_gap.values[i, j]
            reg = pivot_regime.values[i, j]
            reg_code = "C" if reg == "corner" else "I"
            ax3.text(j, i, f"{gap_val:.2f}\n({reg_code})", ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "distortion_sweep_2d.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)

    # Distortion must be downward (a_H^SB <= a_H^B) wherever asking matters (m_H < k).
    downward_when_asks = bool((df["distortion"] >= -1e-4).all())
    pivot_table = pivot_dist.round(3).to_string()
    # P7: in the saturation regime (type L at (k, 0)), positive distortion falls as heterogeneity widens.
    sat = df[(df["m_L"] >= df["m_L"].max() - 1e-6) & (df["distortion"] > 1e-6)]
    falls_with_het = {}
    for b in bias_values:
        d = sat[sat["bias"] == b].sort_values("delta_kappa")["distortion"].values
        if len(d) > 1:
            falls_with_het[b] = bool(np.all(np.diff(d) <= 1e-6))
    n_relax = int(((df["m_L"] < df["m_L"].max() - 1e-6) & (df["distortion"] > 1e-6)).sum())

    # Check dominance
    all_dominate = bool((df["payoff_gap"] >= -1e-4).all())
    regimes_count = df["pooling_regime"].value_counts().to_dict()

    passed = bool(all_dominate and downward_when_asks and all(falls_with_het.values()))
    verdict = "PASS" if passed else "FAIL"
    md_content = fr"""# Verification Report: Heterogeneity vs Bias Distortion & Welfare Sweep

**Status:** **{verdict}**

### 1. Downward distortion $a_H^B - a_H^{{SB}}$ across heterogeneity and bias ($\\kappa_L = {kappa_L}$):
```
{pivot_table}
```
- **Direction:** distortion is non-negative in every cell (downward): **{downward_when_asks}**. Cells where $m_H^{{SB}} = k$ are recorded as zero, because no attribute is left open and $a_H$ is payoff-irrelevant.
- **Zero cells:** under prohibitive friction (bias $\\ge 3$, i.e. $c_Q^{{eff}} \\ge \\Lambda = 5$) the leader never asks, and at narrow heterogeneity the high-cost type specifies fully (full-specification regime of Proposition 7).
- **Heterogeneity:** in the saturation regime (type L at $(k, 0)$), positive distortion falls as $\\Delta\\kappa$ widens, by bias level: {falls_with_het}. This matches Proposition 7 and P7 (distortion intensifies as heterogeneity narrows). {n_relax} positive cells are in the boundary-relaxation regime (type L receives clarification, item 5 of Proposition 7), where no monotonicity in $\\Delta\\kappa$ is claimed.

### 2. Leader-Payoff Gap (Screening vs Corrected Pooling) & Regime Flags:
- **Dominance Holds:** **{all_dominate}** (Minimum gap: `{df['payoff_gap'].min():.4f}`).
- **Pooling Regimes in Sweep:** `{regimes_count}`.
- Pooling ask rates in the sweep ($k=10, L=10, c_Q=2$): {sorted(df['a_SE'].round(3).unique().tolist())}.
- The leader-payoff advantage of screening ($\Pi_{{II}} - \Pi_I$) is positive in {int((df['payoff_gap'] > 1e-4).sum())} of {len(df)} cells, all with bias $\le {df.loc[df['payoff_gap'] > 1e-4, 'bias'].max():.0f}$; under prohibitive friction both regimes shut down asking and tie.

**Artifacts Generated:**
- CSV: `distortion_sweep_2d.csv`
- Figure: `distortion_sweep_2d.png`
- Summary: `distortion_sweep_2d.md`
"""
    md_path = os.path.join(output_dir, "distortion_sweep_2d.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    return {
        "sweep": "distortion_2d",
        "passed": passed,
        "downward_when_asks": downward_when_asks,
        "all_dominate": all_dominate,
        "csv_path": csv_path,
        "png_path": png_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Distortion sweep verification: {'PASS' if res['passed'] else 'FAIL'}")
