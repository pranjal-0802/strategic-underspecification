# Verification Report: Corollary 2 (Bias Shifts Pooling Rate - Feasible Domain)

**Status:** **PASS** (Empirical derivative strictly negative and matches closed-form formula)

### Overview & Mathematical Claim
In `paper/strategic_underspecification.tex`, Corollary 2 is derived by direct differentiation of the closed form in Proposition 4:
$$a^{SE} = -\frac{\bar L}{2\bar Q} = -\frac{(\mu_A s - b)\bar R_0}{(\mu_A s - 2b)\bar\gamma}$$
On the interior branch where $\bar Q < 0$, differentiating with respect to $\lambda_A$ yields:
$$\frac{\partial a^{SE}}{\partial \lambda_A} = -\frac{\bar R_0}{\bar\gamma} \frac{\mu_A s}{(\mu_A s - 2b)^2} \le 0$$
whenever $\bar R_0 \ge 0$ (the physically feasible domain where specification effort cannot exceed attribute count, $m \le k$). More perceived asking friction suppresses asking!

### Findings:
1. **Strictly Interior Regime:** Swept $\lambda_A \in [1.75, 1.95]$ across 21 points. All points satisfy $\bar Q < 0$ (SOC holds) with $a^{SE} \in [0.057, 0.512]$ strictly away from boundaries.
2. **Strictly Negative Slope:** Empirical slopes $\Delta a^{SE} / \Delta \lambda_A$ range from **-3.9400** to **-1.2900**, confirming $a^{SE}$ is strictly **decreasing** in $\lambda_A$.
3. **Formula Match:** Empirical slopes match the analytical derivative with a maximum relative error of **0.52%** across the entire sweep.
### Artifacts Generated:
- CSV: `cor2_bias_direction.csv`
- Figure: `cor2_bias_direction.png`
