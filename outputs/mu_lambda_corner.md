# Robustness Check (Task 2): Ray-Invariance in the Corner Regime

**Status:** **PASS** (Ray-Invariance Fully Proved and Numerically Verified)

---

## Executive Summary
In Model I (pooling), the leader's quadratic objective is:
$$\Pi(a) = \text{const} + \bar{L} a + \bar{Q} a^2$$
where $\bar{L} = (\mu_A s - b)\bar{R}_0$, $\bar{Q} = (\mu_A s - 2b)\bar{\gamma}/2$, with $s \equiv \Lambda - c_Q$ and $b \equiv \lambda_A - c_Q$.

Evaluating $\Pi(1) - \Pi(0)$:
$$\Pi(1) - \Pi(0) = \bar{L} + \bar{Q} = \mu_A \left[ s\left(\bar{R}_0 + \frac{\bar{\gamma}}{2}\right) - \tilde{\rho}(\bar{R}_0 + \bar{\gamma}) \right]$$
where $\tilde{\rho} \equiv \frac{\lambda_A - c_Q}{\mu_A}$ is the excess friction per unit altruism.

### Invariance of the Decision Boundary
Since $\mu_A > 0$:
$$\operatorname{sign}\big(\Pi(1) - \Pi(0)\big) = \operatorname{sign}\left( s\left(\bar{R}_0 + \frac{\bar{\gamma}}{2}\right) - \tilde{\rho} (\bar{R}_0 + \bar{\gamma}) \right)$$
The decision boundary is defined by the critical ratio:
$$\tilde{\rho}^* = \frac{s(\bar{R}_0 + \bar{\gamma}/2)}{\bar{R}_0 + \bar{\gamma}} = 0.9491$$
- When $\tilde{\rho} < \tilde{\rho}^*$: $\Pi(1) - \Pi(0) > 0 \implies a^{SE} = 1$ (always ask), unconditionally for all $\mu_A > 0$.
- When $\tilde{\rho} > \tilde{\rho}^*$: $\Pi(1) - \Pi(0) < 0 \implies a^{SE} = 0$ (never ask), unconditionally for all $\mu_A > 0$.
- **Conclusion**: The optimal corner solution is **strictly invariant** along any ray $\tilde{\rho} = (\lambda_A - c_Q) / \mu_A = \text{const}$. $\mu_A$ acts purely as a positive multiplicative scale on total profit without altering the sign of the discrete decision.

---

## Numerical Verification Results
- **Rays Tested**: 11 rays from $\tilde{\rho} = 0.20$ to $\tilde{\rho} = 1.60$ across 10 values of $\mu_A \in [0.1, 1.0]$ (110 evaluations).
- **Ray-Invariance Failures**: **0**
- **Boundary Crossings along any ray**: **0**
- **Normalized Payoff Collapse**: Across all $\mu_A$, $[\Pi(1) - \Pi(0)] / \mu_A$ collapses onto a single universal curve with zero residual ($< 10^{-14}$).

### Ray Summary Table
| Ray Ratio $\tilde{\rho}$ | Regime | Normalized Diff $[\Pi(1) - \Pi(0)]/\mu_A$ | Selected Corner $a^{SE}$ | Invariant Across All $\mu_A$? |
| :---: | :---: | :---: | :---: | :---: |
| 0.20 | Corner (Q_bar >= 0) | +6.7981 | **1** | YES |
| 0.50 | Corner (Q_bar >= 0) | +4.0755 | **1** | YES |
| 0.70 | Interior | +2.2604 | **1** | YES |
| 0.85 | Interior | +0.8991 | **1** | YES |
| 0.90 | Interior | +0.4453 | **1** | YES |
| 0.92 | Interior | +0.2638 | **1** | YES |
| 0.98 | Interior | -0.2808 | **0** | YES |
| 1.05 | Interior | -0.9160 | **0** | YES |
| 1.20 | Interior | -2.2774 | **0** | YES |
| 1.40 | Interior | -4.0925 | **0** | YES |
| 1.60 | Interior | -5.9075 | **0** | YES |

---

## Artifacts Generated
- CSV: `outputs/mu_lambda_corner.csv`
- Plot: `outputs/mu_lambda_corner.png`
- Summary Markdown: `outputs/mu_lambda_corner.md`
