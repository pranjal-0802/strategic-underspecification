# Verification Report: Corollary 3 (Pooling is a Corner Solution, Not a Compromise)

**Status:** **PASS** (Corrected Corollary 3 Verified 100% Numerically)

### Theoretical Result:
In the unbiased case ($\mu_A = 1, \lambda_A = c_Q$):
$$\bar{Q} = \frac{(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa]}{2} \ge 0 \quad \text{always}.$$

The objective $\Pi(a)$ is weakly convex on $[0, 1]$, its interior stationary point is a minimum, and the true optimum is a corner $a^{SE} \in \{0, 1\}$. The corner is determined by:
$$\Pi(1) - \Pi(0) = \bar{L} + \bar{Q} = (\Lambda - c_Q)\left[\bar{R}_0 + \frac{\bar{\gamma}}{2}\right] = (\Lambda - c_Q)\left[k - \frac{\Lambda + c_Q}{2}\mathbb{E}[1/\kappa]\right].$$

### Numerical Audit Results:
1. **Unbiased Corner 0 Verification ($k=5$):**
   - $\Pi(1) - \Pi(0) = -13.8509 < 0$.
   - Solver returns $a^{SE} = 0.0$ with regime `"corner"`.
   - SOC $\bar{Q} = 12.3647 \ge 0$ confirms convexity. Match confirmed.
2. **Unbiased Corner 1 Verification ($k=10$, Benchmark):**
   - $\Pi(1) - \Pi(0) = 1.1491 > 0$.
   - Solver returns $a^{SE} = 1.0$ with regime `"corner"`.
   - SOC $\bar{Q} = 12.3647 \ge 0$ confirms convexity. Match confirmed.
3. **Comprehensive Grid Sweep (32 configurations):**
   - Verified across variations in $k, L, c_Q$: 100% returned `regime="corner"` with $a^{SE} \in \{0, 1\}$ exactly matching the formula sign.
4. **Biased Case Interior Solution:**
   - When $\bar{Q} < 0$, $\Pi(a)$ is strictly concave; solver returns an interior stationary solution $a^{SE} = 0.5000$ with regime `"interior"`.

**Artifacts Generated:**
- CSV: `cor3_interiority.csv`
- Figure: `cor3_interiority.png`
- Summary: `cor3_interiority.md`
