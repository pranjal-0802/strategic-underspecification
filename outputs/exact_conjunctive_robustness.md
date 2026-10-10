# Robustness Report: Exact Conjunctive Model vs. Linear-Risk Approximation

**Verdict:** **CAUTION**

### Key Findings Across the (k, g) in [3, 20] x [0.50, 0.95] Grid (30 configurations):

1. **Corollary 4 (Corner Solution Property) Survives Universally:**
   - **Corner Solution Rate:** **100.0%** (In 30/30 grid points, a_SE in [0.0, 1.0]).
   - In no region did an interior compromise optimum (a_SE in (0.02, 0.98)) emerge.
   - For small k <= 3, a_SE = 0 (never ask); for all k >= 5 across all g in [0.50, 0.95], a_SE = 1 (always ask).

2. **Calibration Reconciliation with Linear Baselines:**
   - **Taylor-Expansion Baseline (L = V = 100):** Agrees with the exact model in **25 of 30 configurations (83.3%)**. All 5 divergences occur exclusively at k=3, where compounding (q^3) makes autonomous guessing preferred over question friction. For all k >= 5, both models select a_SE = 1.0.
   - **Additive Localized-Defect Baseline (L = 10):** Agrees in **17 of 30 configurations (56.7%)**. Divergences occur at high g >= 0.90 where localized attribute stakes make guessing cost-effective earlier than systemic failure stakes.

3. **Proposition 7 (Downward Distortion Under Bias) Robustness:**
   - **Downward Distortion Rate:** a_H_SB <= a_H_B holds everywhere, with strict downward distortion a_H_SB < a_H_B whenever m does not saturate at k (holding in **66.7%** of grid points: 20% at k in [3, 5], 80% at k in [8, 20], and 100% at k in [10, 15]).
   - **Active Constraint Set:**
     - IR_H is strictly slack in 100% of tested configurations (minimum slack > 15.0), confirming that rent-minimization does not bind without monetary transfers.
     - IC_L binds in 100% of non-degenerate configurations.
     - At small k <= 5, specification effort saturates at m=k, which eliminates the asking wedge (k-m=0) and causes both IC constraints to hold with equality (pooling). For k >= 10, IC_L binds alone and generates substantial downward distortion.

4. **Validity Region of Linear-Risk Approximation:**
   - **Survives:** Qualitative direction of downward distortion (a_H_SB < a_H_B), corner nature of unbiased pooling (a_SE in [0, 1]), and the non-binding status of individual rationality (IR_H slack).
   - **Cautionary Boundary:** Quantitative distortion magnitudes diverge at low k due to boundary saturation (m=k) and compounding in the exact non-linear exponent.

**Artifacts Generated:**
- CSV: `exact_conjunctive_robustness.csv`
- Figure: `exact_conjunctive_robustness.png`
- Summary: `exact_conjunctive_robustness.md`
