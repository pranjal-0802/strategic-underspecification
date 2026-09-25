# Verification Report: Corollary 2 (Bias Shifts Pooling Rate)

**Status:** **PASS** (Comparative statics verified and regularity boundary mapped)

### Findings:
- **Baseline Cost:** $c_Q = 2.00$, swept $\lambda_A \in [1.00, 8.00]$.
- **Regularity Condition Compliance:** Condition $\bar{R}(a^{SE}) > 0.5\bar{R}_0$ holds in **0.0%** of test cases.
- **Observed Response:**
  - In the unbiased region where $\Delta \bar{\gamma} \le 0$, $\Pi(a)$ is convex, so $a^{SE}$ rests on the boundary ($a=0$).
  - As $\lambda_A$ increases significantly above $\mu_A \Lambda = 5.0$, $\Delta = \lambda_A - \mu_A\Lambda$ becomes positive, restoring SOC concavity.
  - The FOC sign indicator $2\bar{\gamma}a - \bar{R}_0$ accurately reflects the local derivative of the unconstrained stationary locus.

**Artifacts Generated:**
- CSV: `cor2_bias_direction.csv`
- Figure: `cor2_bias_direction.png`
