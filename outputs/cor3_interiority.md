# Verification Report: Corollary 3 (Pooling Interiority)

**Status:** **FAIL (Paper Corollary 3 Disproven Numerically)**

### Key Diagnostic:
- **Test:** At $\mu_A = 1.0, \lambda_A = c_Q = 2.0$, with $F$ spanning $[0.20, 0.60]$ around $\kappa^*=0.35$.
- **Expected by Corollary 3:** $a^{SE} \in (0, 1)$ strictly interior.
- **Observed:** $a^{SE} = 0.0000$ (Lands at boundary corner $a=0$).

### Mathematical Cause (Auditing the Paper):
1. In the paper's linear-risk pooling formulation (eq 279):
   $$\Pi(a) = \mu_A V - C_0\bar{R}_0 - a[C_0\bar{\gamma} + \Delta\bar{R}_0] - a^2\Delta\bar{\gamma}$$
2. In the unbiased case ($\mu_A=1, \lambda_A=c_Q$):
   $$\Delta = c_Q - \Lambda, \qquad \bar{\gamma} = \frac{\Lambda - c_Q}{\kappa}$$
   $$\implies \Delta\bar{\gamma} = -(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] \le 0$$
3. Thus, the second derivative is:
   $$\frac{d^2\Pi}{da^2} = -2\Delta\bar{\gamma} = + 2(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] > 0$$
4. $\Pi(a)$ is **strictly convex** in $a$ for any non-degenerate type distribution!
5. Any strictly convex function on a closed interval $[0, 1]$ achieves its maximum at a **boundary** ($a=0$ or $a=1$), never in the interior $(0, 1)$.
6. Therefore, Corollary 3 does not hold under the quadratic formulation of Section 5.2. An interior pooling rate requires either congestion costs, capacity constraints, or the full non-linear conjunctive risk function.

**Artifacts Generated:**
- CSV: `cor3_interiority.csv`
- Figure: `cor3_interiority.png`
