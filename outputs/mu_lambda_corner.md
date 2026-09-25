# Verification Report: Ray-Invariance in the Corner Regime

## Overview
- **Reference**: `strategic_underspecification_v3.tex`, Corollary 3, Remark 3 (Scope of Corollary 3).
- **Key Question**: Does the scalar ratio invariance $\lambda_A / \mu_A$ proven for the interior branch ($\Delta\bar\gamma > 0$) extend to the corner regime ($\Delta\bar\gamma \le 0$)?
- **Theoretical Scope**: In Remark 3 of the paper, whether ray-invariance holds for corner selection was left as an open question. This check resolves that question both analytically and numerically.

---

## Executive Summary & Verdict: **PASS**

### 1. Analytical Proof of Ray-Invariance in the Corner Regime
In Model I (pooling), the leader's quadratic objective is:
$$\Pi(a) = \mu_A V - C_0 \bar R_0 - a \big[ C_0 \bar\gamma + \Delta \bar R_0 \big] - a^2 \Delta \bar\gamma$$
where $C_0 = \mu_A \Lambda$ and $\Delta = \lambda_A - \mu_A \Lambda$.

1. **Regime Boundary Invariance**:
   $$\Delta = \mu_A \left( \frac{\lambda_A}{\mu_A} - \Lambda \right)$$
   Since $\mu_A > 0$, the sign of $\Delta$, and thus whether the leader is in the interior branch ($\Delta\bar\gamma > 0$) or the corner branch ($\Delta\bar\gamma \le 0$), depends strictly on the ratio:
   $$\rho = \frac{\lambda_A}{\mu_A} \quad \text{relative to} \quad \Lambda$$

2. **Corner Payoff Difference Invariance**:
   Evaluating $\Pi(1) - \Pi(0)$:
   $$\Pi(1) - \Pi(0) = -\big[ C_0 \bar\gamma + \Delta (\bar R_0 + \bar\gamma) \big]$$
   Substituting $C_0 = \mu_A \Lambda$ and $\Delta = \lambda_A - \mu_A \Lambda$:
   $$\Pi(1) - \Pi(0) = -\Big[ \mu_A \Lambda \bar\gamma + (\lambda_A - \mu_A \Lambda)(\bar R_0 + \bar\gamma) \Big]$$
   $$\Pi(1) - \Pi(0) = \mu_A \Lambda \bar R_0 - \lambda_A (\bar R_0 + \bar\gamma)$$
   Factoring out $\mu_A$:
   $$\Pi(1) - \Pi(0) = \mu_A \cdot \left[ \Lambda \bar R_0 - \frac{\lambda_A}{\mu_A} (\bar R_0 + \bar\gamma) \right]$$

3. **Invariance of the Decision Boundary**:
   Since $\mu_A > 0$:
   $$\operatorname{sign}\big(\Pi(1) - \Pi(0)\big) = \operatorname{sign}\left( \Lambda \bar R_0 - \rho (\bar R_0 + \bar\gamma) \right)$$
   The decision boundary is defined by the critical ratio:
   $$\rho^* = \left( \frac{\lambda_A}{\mu_A} \right)^* = \frac{\Lambda \bar R_0}{\bar R_0 + \bar\gamma} = 1.7963$$
   - When $\lambda_A / \mu_A < \rho^*$: $\Pi(1) - \Pi(0) > 0 \implies a^{SE} = 1$ (always ask), unconditionally for all $\mu_A > 0$.
   - When $\lambda_A / \mu_A > \rho^*$: $\Pi(1) - \Pi(0) < 0 \implies a^{SE} = 0$ (never ask), unconditionally for all $\mu_A > 0$.
   - **Conclusion**: The optimal corner solution is **strictly invariant** along any ray $\lambda_A / \mu_A = \text{const}$. $\mu_A$ acts purely as a positive multiplicative scale on total profit without altering the sign of the discrete decision.

---

## Numerical Verification Results
- **Rays Tested**: 11 rays from $\rho = 0.20$ to $\rho = 2.00$ across 10 values of $\mu_A \in [0.1, 1.0]$ (110 evaluations).
- **Ray-Invariance Failures**: **0**
- **Boundary Crossings along any ray**: **0**
- **Normalized Payoff Collapse**: Across all $\mu_A$, $[\Pi(1) - \Pi(0)] / \mu_A$ collapses onto a single universal curve with zero residual ($< 10^{-14}$).

### Ray Summary Table
| Ray Ratio $\rho = \lambda_A / \mu_A$ | Regime | Normalized Diff $[\Pi(1) - \Pi(0)]/\mu_A$ | Selected Corner $a^{SE}$ | Invariant Across All $\mu_A$? |
| :---: | :---: | :---: | :---: | :---: |
| 0.20 | Corner (Delta <= 0) | +14.4881 | **1** | YES |
| 0.50 | Corner (Delta <= 0) | +11.7653 | **1** | YES |
| 0.80 | Corner (Delta <= 0) | +9.0426 | **1** | YES |
| 1.00 | Corner (Delta <= 0) | +7.2274 | **1** | YES |
| 1.20 | Corner (Delta <= 0) | +5.4123 | **1** | YES |
| 1.40 | Corner (Delta <= 0) | +3.5971 | **1** | YES |
| 1.60 | Corner (Delta <= 0) | +1.7819 | **1** | YES |
| 1.75 | Corner (Delta <= 0) | +0.4206 | **1** | YES |
| 1.85 | Corner (Delta <= 0) | -0.4870 | **0** | YES |
| 1.95 | Corner (Delta <= 0) | -1.3946 | **0** | YES |
| 2.00 | Corner (Delta <= 0) | -1.8484 | **0** | YES |

---

## Resolution of the Open Question in Remark 3
The paper noted in Remark 3:
> *"Whether the same non-identification holds on the corner branch, where the relevant comparison is $\Pi(1)-\Pi(0)$ rather than the interior stationary point, is a separate, currently open question..."*

**Answer**: **YES, ray-invariance holds unconditionally across the entire parameter space.**
The two bias parameters $\mu_A$ and $\lambda_A$ are strictly non-identified through the pooling ask rate $a^{SE}$ in **both** the interior regime and the corner regime:
1. In the interior regime, the continuous ask rate $a^{SE}$ is a function solely of $\lambda_A / \mu_A$.
2. In the corner regime, the discrete choice $a^{SE} \in \{0, 1\}$ is a step function solely of $\lambda_A / \mu_A$.
3. The boundary between the two regimes itself depends strictly on $\lambda_A / \mu_A = \Lambda$.

---

## Artifacts Generated
- CSV: `outputs/mu_lambda_corner.csv`
- Plot: `outputs/mu_lambda_corner.png`
- Summary Markdown: `outputs/mu_lambda_corner.md`
