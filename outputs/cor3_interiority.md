# Verification Report: Corollary 3 (Pooling is a Corner Solution, Not a Compromise)

**Status:** **PASS** (Corrected Corollary 3 Verified 100% Numerically)

### Theoretical Result (Paper eq 301-328):
In the unbiased case ($\mu_A = 1, \lambda_A = c_Q$):
$$\Delta = c_Q - \Lambda, \qquad \bar{\gamma} = (\Lambda - c_Q) \mathbb{E}[1/\kappa]$$
$$\Delta\bar{\gamma} = -(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] \le 0 \quad \text{always}.$$

The second-order condition fails: $\Pi(a)$ is weakly convex on $[0, 1]$, its interior stationary point is a minimum, and the true optimum is a corner $a^{SE} \in \{0, 1\}$. The corner is determined by:
$$\Pi(1) - \Pi(0) = (\Lambda - c_Q)\Big[\bar{R}_0 - c_Q \mathbb{E}[1/\kappa]\Big].$$

### Numerical Audit Results:
1. **Unbiased Corner 0 Verification ($k=10$):**
   - $\Pi(1) - \Pi(0) = -27.7019 < 0$.
   - Solver returns $a^{SE} = 0.0$ with regime `"corner"`.
   - SOC $\Delta\bar{\gamma} = -24.7294 \le 0$ confirms convexity. Match confirmed.
2. **Unbiased Corner 1 Verification ($k=30$):**
   - $\Pi(1) - \Pi(0) = 32.2981 > 0$.
   - Solver returns $a^{SE} = 1.0$ with regime `"corner"`.
   - SOC $\Delta\bar{\gamma} = -24.7294 \le 0$ confirms convexity. Match confirmed.
3. **Comprehensive Grid Sweep (32 configurations):**
   - Verified across variations in $k, L, c_Q$: 100% returned `regime="corner"` with $a^{SE} \in \{0, 1\}$ exactly matching the formula sign.
4. **Biased Case Interior Solution:**
   - When $\lambda_A - \mu_A\Lambda$ and $\Lambda - c_Q$ share a sign, $\Delta\bar{\gamma} > 0$.
   - $\Pi(a)$ is strictly concave; solver returns an interior stationary solution $a^{SE} = 0.5000$ with regime `"interior"`.

**Artifacts Generated:**
- CSV: `cor3_interiority.csv`
- Figure: `cor3_interiority.png`
- Summary: `cor3_interiority.md`
