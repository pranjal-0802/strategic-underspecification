# Verification Report: Section 7 Regime Comparison (Dominance of Menus)

**Status:** **PASS** (Dominance Corollary 4 Confirmed With Corrected Pooling Solver)

### Key Results:
- **Dominance Corollary Verified:** Across all 25 2D grid points, Model II payoff weakly exceeds Model I (Min gap: `0.000000`, Mean gap: `1.2015`, Max gap: `7.0833`).
- **Audit Against Corrected Solver:**
  - Evaluated using the updated pooling solver with explicit corner-checking ($\Delta\bar\gamma \le 0$).
  - In this 2D grid ($k=10, L=10, c_Q=2$), pooling is at a corner in 100% of grid points (`pooling_regime: {'corner': 25}`).
  - Model I payoff reaches a maximum of `83.75` and minimum of `77.92`.
  - Model II payoff reaches a maximum of `86.19` and minimum of `77.92`.
  - Because revealed preference guarantees that any single pooling ask rate $a \in [0, 1]$ is a feasible menu $(m^*(a), a, m^*(a), a)$, the screening policy weakly dominates pooling under both corner and interior pooling regimes.

### Decomposition:
1. **Instrument Restriction Loss (Zero Bias):** At $\\lambda_A = c_Q$, Model II dominates Model I with a positive gap (e.g., gap = `7.08` at $\\Delta\\kappa=0.30$) purely because pooling cannot offer type-contingent ask rates.
2. **Heterogeneity Effect:** The welfare advantage $\\Pi_{II} - \\Pi_I$ grows monotonically with population cost heterogeneity $\\Delta\\kappa = \\kappa_H - \\kappa_L$.
3. **Bias Effect:** Severe bias compresses asking toward zero for both types, narrowing the operational gap between pooling and screening at extreme bias.

**Artifacts Generated:**
- CSV: `regime_comparison.csv`
- Figure: `regime_comparison.png`
- Summary: `regime_comparison.md`
