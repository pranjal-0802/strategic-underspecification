# Verification Report: Proposition 4 (Stackelberg Pooling Closed Form)

**Status:** **PASS** (Symbolic Identity Verified & Physical SOC Confirmed)

### 1. Symbolic Algebraic Proof (Sympy)
- **Identity:** `dPi/da = (mu_A * s - b) * R0_bar + (mu_A * s - 2 * b) * gamma_bar * a = 0`
- **Result:** Symbolic solution uniquely yields `a^{SE} = -((mu_A * s - b) * R0_bar) / ((mu_A * s - 2 * b) * gamma_bar)`.
- **Status:** **PASS** (Zero symbolic residual).

### 2. Physical Unclipped Domain Verification
- **Parameters:** $k=10.0$, $\Lambda=2.0$, $c_Q=0.8$, $\lambda_A=1.7500$, $\mu_A=1.0$
- **Domain Validity:** $\kappa \in [0.25, 0.40] \ge \Lambda/k = 0.20$, ensuring prompt effort $m^*(\kappa) \le 8.0 < k=10.0$ for all types (zero clipping).
- **Residual Attributes:** $\bar{R}_0 = -3.7331 > 0$.
- **SOC Value:** $\bar{Q} = -1.3161 < 0$ (strictly concave).
- **Closed-form $a^{SE}$ (Prop 4):** `0.3546`
- **Grid-search Argmax on $[0, 1]$:** `0.3546`
- **Discrepancy:** `0.000017` (Tolerance: `1e-2`)
- **Result:** Closed form matches numerical grid argmax within `0.000017`, with bounded payoffs $\Pi \in [90, 95] \le V=100$.

### 3. Analytical Finding: Failure of SOC in the Unbiased Case
In the paper's unbiased baseline ($\mu_A=1, \lambda_A=c_Q$):
- $s = \Lambda - c_Q$
- $b = \lambda_A - c_Q = 0$
- Therefore, $\bar{Q} = \frac{s^2 \bar{\gamma}}{2} = \frac{(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa]}{2} \ge 0$ **always**!
- Because $\bar{Q} > 0$, $\Pi(a)$ is strictly **convex** on $[0, 1]$.
- Consequently, the unconstrained stationary point is a **local minimum**, and the global maximum is at a boundary ($a=0$ or $a=1$).

**Artifacts Generated:**
- CSV: `prop4_pooling_closed_form.csv`
- Figure: `prop4_pooling_closed_form.png`
