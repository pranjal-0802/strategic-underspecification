# Verification Report: Monotonicity Under Assumption 1 & Empirical Tightness

## Overview
- **Reference**: `paper/strategic_underspecification.tex`, Assumption 1, Corollary 1, Proposition 2, Remark 1.
- **Key Question**: Does the monotonicity conclusion of Corollary 1 ($m^*(g)$ weakly decreasing in $g$) hold whenever Assumption 1 ($(k-m^*)\ln q \ge -1$) is satisfied, and does monotonicity break when Assumption 1 fails (confirming tightness)?
- **Grid Swept**: $k \in [5, 15]$ (11 points), $q \in [0.70, 0.99]$ (30 points), $\kappa \in [0.20, 2.00]$ (8 types), totaling **2552** finite-difference intervals.

---

## Executive Summary & Verdict: **PASS**

### 1. Headline Findings
1. **Monotonicity Holds Unconditionally Whenever Assumption 1 Holds**:
   - Total intervals where Assumption 1 **HOLDS** ($(k-m^*)\ln q \ge -1$): **2162** (84.72% of grid)
   - Intervals where monotonicity **HOLDS**: **2162 (100.00%)**
   - Intervals where monotonicity **BREAKS**: **0 (0.00%)**
   - **Conclusion**: Topkis's theorem and Corollary 1 are confirmed with 100% empirical fidelity on the canonical grid. Whenever $(k-m^*)\ln q \ge -1$, $m^*(g)$ is weakly decreasing in $g$ with zero violations.

2. **Empirical Tightness of Assumption 1**:
   - Total intervals where Assumption 1 **FAILS** ($(k-m^*)\ln q < -1$): **390** (15.28% of grid)
   - Intervals where monotonicity **BREAKS**: **377 (96.67%)**
   - Intervals where monotonicity **HOLDS**: **13 (3.33%)** (solely due to boundary saturation where $m^* = k$)
   - **Conclusion**: When Assumption 1 fails, the cross-partial $\partial^2 U/\partial m\partial q = -V q^{k-m-1}[1 + (k-m)\ln q]$ becomes strictly positive (supermodularity / increasing differences). Higher guessing accuracy makes users specify *more* attributes, causing monotonicity to invert in 96.67% of cases. Assumption 1 is therefore practically necessary as well as sufficient.

3. **Proposition 2 (Naive vs. Sophisticated Users)**:
   - **Welfare**: Naive users are weakly worse off than sophisticated users ($U_{\text{naive}} \le U_{\text{soph}}$) in **100%** of grid points (2552 / 2552). This holds unconditionally by suboptimality of optimizing against a miscalibrated belief $a^\dagger \ne a_A$.
   - **Specification Level**:
     - Where Assumption 1 holds ($(k - m)\ln q \ge -1$), naive users **under-specify** ($m^*_{\text{naive}} \le m^*_{\text{soph}}$) in **100.0%** of intervals (2162 / 2162) because $m^*$ decreases with perceived accuracy.
     - Where Assumption 1 fails ($(k - m)\ln q < -1$), naive users **over-specify** ($m^*_{\text{naive}} > m^*_{\text{soph}}$) in **93.85%** of intervals (366 / 390) because the positive cross-partial makes higher perceived accuracy induce higher specification effort.

---

## Detailed Cross-Tabulation Tables

### Table 1: Corollary 1 Monotonicity ($m^*(g)$ Non-Increasing)
| Condition | Monotonicity Holds ($m_2 \le m_1$) | Monotonicity Breaks ($m_2 > m_1$) | Total Intervals |
| :--- | :---: | :---: | :---: |
| **Assumption 1 HOLDS** ($(k-m)\ln q \ge -1$) | **2162 (100.0%)** | **0 (0.0%)** | **2162** |
| **Assumption 1 FAILS** ($(k-m)\ln q < -1$) | **13 (3.3%)** | **377 (96.7%)** | **390** |
| **Total** | **2175** | **377** | **2552** |

### Table 2: Proposition 2 Naive Welfare & Under-Specification
| Condition | Naive Worse Off ($U_{\text{naive}} \le U_{\text{soph}}$) | Naive Under-Specifies ($m_{\text{naive}} \le m_{\text{soph}}$) | Naive Over-Specifies ($m_{\text{naive}} > m_{\text{soph}}$) |
| :--- | :---: | :---: | :---: |
| **Assumption 1 HOLDS** | **2162 / 2162 (100%)** | **2162** | **0** |
| **Assumption 1 FAILS** | **390 / 390 (100%)** | **24** | **366** |

---

## Artifacts Generated
- CSV: `outputs/monotonicity_survival.csv`
- Plot: `outputs/monotonicity_survival.png`
- Summary Markdown: `outputs/monotonicity_survival.md`
