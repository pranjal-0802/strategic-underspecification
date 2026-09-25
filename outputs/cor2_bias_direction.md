# Verification Report: Corollary 2 (Bias Shifts Pooling Rate - Corrected)

**Status:** **PASS** (Empirical derivative strictly positive and matches closed-form formula)

### Overview & Corrected Mathematical Claim
In `paper/strategic_underspecification.tex`, Corollary 2 is derived by direct differentiation of the closed form in Proposition 4:
$$a^{SE} = -\frac{\bar R_0}{2\bar\gamma} - \frac{1}{2} \left[\frac{1}{\lambda_A/(\mu_A\Lambda) - 1}\right]$$
On the interior branch where $\Delta\bar\gamma > 0$, differentiating with respect to $\lambda_A$ yields:
$$\frac{\partial a^{SE}}{\partial \lambda_A} = \frac{\mu_A\Lambda}{2(\mu_A\Lambda - \lambda_A)^2} > 0$$
whenever $\mu_A, \Lambda > 0$ and $\lambda_A \ne \mu_A\Lambda$.

### Findings:
1. **Strictly Interior Regime:** Swept $\lambda_A \in [3.50, 12.00]$ across 20 points. All points satisfy $\Delta\bar\gamma > 0$ (SOC holds) with $a^{SE} \in [0.283, 0.850]$ strictly away from boundaries.
2. **Strictly Positive Slope:** Empirical slopes $\Delta a^{SE} / \Delta \lambda_A$ range from **0.0105** to **0.3423**, confirming $a^{SE}$ is strictly **increasing** in $\lambda_A$.
3. **Formula Match:** Empirical slopes match the analytical derivative with a maximum relative error of **3.31%** across the entire sweep.
4. **Superseded Draft Claim:** The earlier draft conjectured that the sign tracked $2\bar\gamma a^{SE} - \bar R_0$ and concluded $a^{SE}$ was decreasing. That conjecture is superseded by the direct derivative above.
5. **Scope & Caveat:** As emphasized in the paper, this result holds strictly on the interior branch ($\Delta\bar\gamma > 0$). In the unbiased/corner regime ($\Delta\bar\gamma \le 0$), the leader's payoff is weakly convex and the global optimum is governed by corner comparison (Corollary 4).

**Artifacts Generated:**
- CSV: `cor2_bias_direction.csv`
- Figure: `cor2_bias_direction.png`
