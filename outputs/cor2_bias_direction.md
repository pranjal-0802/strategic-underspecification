# Verification Report: Corollary 2 (Bias Shifts Pooling Rate - Corrected)

**Status:** **PASS** (Empirical derivative strictly positive and matches closed-form formula)

### Overview & Corrected Mathematical Claim
In `paper/strategic_underspecification.tex`, Corollary 2 is derived by direct differentiation of the closed form in Proposition 4:
$$a^{SE} = -\frac{\bar L}{2\bar Q} = -\frac{(\mu_A s - b)\bar R_0}{(\mu_A s - 2b)\bar\gamma}$$
On the interior branch where $\bar Q < 0$, differentiating with respect to $\lambda_A$ yields:
$$\frac{\partial a^{SE}}{\partial \lambda_A} = -\frac{\bar R_0}{\bar\gamma} \frac{\mu_A s}{(\mu_A s - 2b)^2} > 0$$
whenever $\bar R_0 < 0$ (under-specification regime), $\mu_A, s, \bar\gamma > 0$.

### Findings:
1. **Strictly Interior Regime:** Swept $\lambda_A \in [2.20, 5.00]$ across 20 points. All points satisfy $\bar Q < 0$ (SOC holds) with $a^{SE} \in [0.271, 0.814]$ strictly away from boundaries.
2. **Strictly Positive Slope:** Empirical slopes $\Delta a^{SE} / \Delta \lambda_A$ range from **0.0405** to **0.8008**, confirming $a^{SE}$ is strictly **increasing** in $\lambda_A$.
3. **Formula Match:** Empirical slopes match the analytical derivative with a maximum relative error of **1.80%** across the entire sweep.
4. **Scope & Caveat:** As emphasized in the paper, this result holds strictly on the interior branch ($\bar Q < 0$). In the unbiased/corner regime ($\bar Q \ge 0$), the leader's payoff is weakly convex and the global optimum is governed by corner comparison (Corollary 3).

**Artifacts Generated:**
- CSV: `cor2_bias_direction.csv`
- Figure: `cor2_bias_direction.png`
