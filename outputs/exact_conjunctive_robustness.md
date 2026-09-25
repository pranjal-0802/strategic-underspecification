# Robustness Report: Exact Conjunctive Model vs. Linear-Risk Approximation

**Verdict:** **CAUTION**

### Key Findings Across the $(k, g) \in [3, 20] \times [0.50, 0.95]$ Grid (30 configurations):

1. **Corollary 3 (Corner Solution Property) Survives Completely:**
   - **Corner Solution Rate:** **100.0%** (In 30/30 grid points, $a^{SE}_{\text{exact}} \in \{0.0, 1.0\}$).
   - In no region did an interior compromise optimum ($a^{SE} \in (0.02, 0.98)$) emerge.
   - For small $k \le 3$, $a^{SE} = 0$ (never ask); for $k \ge 5$, $a^{SE} = 1$ (always ask).
   - *Conclusion:* The qualitative conclusion of Corollary 3---that pooling in the unbiased regime is a corner solution picking a winner rather than a smooth interior compromise---is **robust to the exact conjunctive specification**.

2. **Proposition 6 (Downward Distortion Under Bias) Robustness:**
   - **Downward Distortion Rate:** $a_H^{SB} \le a_H^B$ holds everywhere, with strict downward distortion $a_H^{SB} < a_H^B$ whenever $m$ does not saturate at $k$ (holding in **66.7%** of grid points).
   - **Active Constraint Set:**
     - $IR_H$ is strictly slack in 100% of tested configurations (minimum slack $> 15.0$), confirming that rent-minimization does not bind without monetary transfers.
     - $IC_L$ binds in 100% of non-degenerate configurations.
     - At very small $k \le 5$, specification effort saturates at $m=k$, which eliminates the asking wedge $(k-m=0)$ and causes both IC constraints to hold with equality (pooling). For $k \ge 10$, $IC_L$ binds alone and generates substantial downward distortion (up to 0.18).

3. **Validity Region of Linear-Risk Approximation:**
   - **Survives:** Qualitative direction of downward distortion ($a_H^{SB} < a_H^B$), corner nature of unbiased pooling ($a^{SE} \in \{0, 1\}$), and the non-binding status of individual rationality ($IR_H$ slack).
   - **Cautionary Boundary:** Quantitative distortion magnitudes diverge slightly at low $k$ due to boundary saturation ($m=k$) in the exact non-linear exponent.

**Artifacts Generated:**
- CSV: `exact_conjunctive_robustness.csv`
- Figure: `exact_conjunctive_robustness.png`
- Summary: `exact_conjunctive_robustness.md`
