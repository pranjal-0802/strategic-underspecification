# Verification Report: Monotonicity Survival Outside Assumption 1

## Overview
- **Reference**: `strategic_underspecification_v3.tex`, Assumption 1, Corollary 1, Proposition 2, Remark 1.
- **Key Question**: Does the monotonicity conclusion of Corollary 1 ($m^*(g)$ weakly decreasing in $g$) and Proposition 2 (naive users worse off) actually fail outside the 10.1% region where Assumption 1 holds?
- **Grid Swept**: $k \in [5, 15]$ (11 points), $q \in [0.70, 0.99]$ (30 points), $\kappa \in [0.20, 2.00]$ (8 types), totaling **2552** finite-difference intervals.

---

## Executive Summary & Verdict: **CAUTION**

### 1. Headline Findings
1. **Monotonicity Survives 100% Outside Assumption 1**:
   - Total intervals where Assumption 1 **FAILS**: **2162**
   - Intervals where monotonicity **HOLDS** when Assumption 1 fails: **2162 (100.00%)**
   - Intervals where monotonicity **BREAKS** when Assumption 1 fails: **0 (0.00%)**
   - **Conclusion**: Monotonicity **NEVER** fails outside Assumption 1's region.

2. **The Mathematical Inversion of Assumption 1**:
   - In the paper, Assumption 1 is formulated as:
     $$(k - m^*)\ln q(a, g) \le -1$$
   - Computing the cross-partial derivative of user utility $U(m; q) = V q^{k-m} - c(m) - a(k-m)c_Q$:
     $$\frac{\partial U}{\partial m} = -V\ln(q) q^{k-m} - c'(m) + ac_Q$$
     $$\frac{\partial^2 U}{\partial m \partial q} = -V q^{k-m-1} \Big[ 1 + (k - m)\ln(q) \Big]$$
   - For **decreasing differences** (submodularity, $\frac{\partial^2 U}{\partial m \partial q} \le 0$), we require:
     $$1 + (k - m)\ln(q) \ge 0 \iff (k - m)\ln(q) \ge -1$$
   - **Crucial Mathematical Insight**: The condition for decreasing differences is $(k - m)\ln(q) \ge -1$, which is the **exact opposite** of Assumption 1's inequality!
   - As a result:
     - **Inside Assumption 1** ($(k - m)\ln q \le -1$): $\frac{\partial^2 U}{\partial m \partial q} > 0$ (**increasing differences** / supermodularity). As $q$ rises, the marginal benefit of specifying increases, causing $m^*$ to **INCREASE** with $g$!
       In fact, where Assumption 1 holds, monotonicity fails in **377 / 390 (96.7%)** of tested pairs.
     - **Outside Assumption 1** ($(k - m)\ln q \ge -1$): $\frac{\partial^2 U}{\partial m \partial q} \le 0$ (**decreasing differences**). As $q$ rises, users specify less ($m^*$ is weakly decreasing in $g$) in **100.0%** of tested pairs!

3. **Proposition 2 (Naive vs. Sophisticated Users)**:
   - **Welfare**: Naive users are weakly worse off than sophisticated users ($U_{\text{naive}} \le U_{\text{soph}}$) in **100%** of grid points (2552 / 2552). This holds unconditionally by suboptimality of choosing an action against a miscalibrated belief $a^\dagger \ne a_A$.
   - **Specification Level**:
     - Where $(k - m)\ln q \ge -1$ (outside Assumption 1), naive users **under-specify** ($m^*_{\text{naive}} \le m^*_{\text{soph}}$) because $m^*$ decreases with perceived accuracy.
     - Where $(k - m)\ln q < -1$ (inside Assumption 1), naive users **over-specify** ($m^*_{\text{naive}} > m^*_{\text{soph}}$) because the positive cross-partial makes higher perceived accuracy induce more specification!

---

## Detailed Cross-Tabulation Tables

### Table 1: Corollary 1 Monotonicity ($m^*(g)$ Non-Increasing)
| Condition | Monotonicity Holds ($m_2 \le m_1$) | Monotonicity Breaks ($m_2 > m_1$) | Total Intervals |
| :--- | :---: | :---: | :---: |
| **Assumption 1 FAILS** ($(k-m)\ln q > -1$) | **2162 (100.0%)** | **0 (0.0%)** | **2162** |
| **Assumption 1 HOLDS** ($(k-m)\ln q \le -1$) | **13 (3.3%)** | **377 (96.7%)** | **390** |
| **Total** | **2175** | **377** | **2552** |

### Table 2: Proposition 2 Naive Welfare & Under-Specification
| Condition | Naive Worse Off ($U_{\text{naive}} \le U_{\text{soph}}$) | Naive Under-Specifies ($m_{\text{naive}} \le m_{\text{soph}}$) | Naive Over-Specifies ($m_{\text{naive}} > m_{\text{soph}}$) |
| :--- | :---: | :---: | :---: |
| **Assumption 1 FAILS** | **2162 / 2162 (100%)** | **2162** | **0** |
| **Assumption 1 HOLDS** | **390 / 390 (100%)** | **24** | **366** |

---

## Artifacts Generated
- CSV: `outputs/monotonicity_survival.csv`
- Plot: `outputs/monotonicity_survival.png`
- Summary Markdown: `outputs/monotonicity_survival.md`
