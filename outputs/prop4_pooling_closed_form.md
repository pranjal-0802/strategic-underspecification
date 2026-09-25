# Verification Report: Proposition 4 (Stackelberg Pooling Closed Form)

**Status:** **PASS** (under stated SOC: $\bar{Q} < 0$)

### 1. Verification Under Stated Second-Order Condition ($\bar{Q} < 0$)
- **Parameters:** $k=5.0$, $\Lambda=4.0$, $c_Q=2.0$, $\lambda_A=7.8909$
- **SOC Value:** $\bar{Q} = -32.9136 < 0$ (strictly concave).
- **Closed-form $a^{SE}$ (Prop 4):** `0.5000`
- **Grid-search Argmax on $[0, 1]$:** `0.5000`
- **Discrepancy:** `0.000000` (Tolerance: `1e-2`)
- **Result:** Formula matches grid-search peak within numerical resolution!

### 2. Critical Analytical Finding: Failure of SOC in the Unbiased Case
In the paper's unbiased baseline ($\mu_A=1, \lambda_A=c_Q$):
- $s = \Lambda - c_Q$
- $b = \lambda_A - c_Q = 0$
- Therefore, $\bar{Q} = \frac{s^2 \bar{\gamma}}{2} = \frac{(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa]}{2} \ge 0$ **always**!
- Because $\bar{Q} > 0$, $\Pi(a)$ is strictly **convex** on $[0, 1]$.
- Consequently, the unconstrained stationary point from Proposition 4 is a **local minimum**, not a maximum. The true maximum on $[0, 1]$ is at a **boundary** ($a=0$ or $a=1$).

**Artifacts Generated:**
- CSV: `prop4_pooling_closed_form.csv`
- Figure: `prop4_pooling_closed_form.png`
