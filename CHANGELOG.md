# Changelog: Mathematical Resync & Downstream Pipeline Validation

## Overview

Following the updates to the foundational paper [*Strategic Under-Specification: A Stackelberg Game Between User and Assistant*](strategic_underspecification.tex), this release synchronizes the theoretical derivations, updates the pooling equilibrium solver, extends the constraint-activity audit for screening under extreme bias, and validates all downstream numerical pipelines.

---

## 1. Paper Synchronization (`strategic_underspecification.tex`)

- **Corollary 3 ("Pooling is a corner solution, not a compromise"):**
  - Updated from the preliminary conjecture of interior compromise to the proven result: in the unbiased case ($\mu_A=1, \lambda_A=c_Q$), $\Delta\bar\gamma = -(\Lambda-c_Q)^2 \mathbb{E}[1/\kappa] \le 0$ always.
  - Payoff $\Pi(a)$ is weakly convex on $[0, 1]$, its stationary point is a local minimum, and the global pooling optimum is a corner $a^{SE} \in \{0, 1\}$.
  - The corner is selected by direct comparison: $\Pi(1) - \Pi(0) = (\Lambda-c_Q)[\bar R_0 - c_Q\mathbb{E}[1/\kappa]]$.
  - Away from the unbiased case, an interior $a^{SE} \in (0, 1)$ is possible specifically when $(\lambda_A - \mu_A\Lambda)$ and $(\Lambda-c_Q)$ share a sign ($\Delta\bar\gamma > 0$).
- **Proposition 6 Proof Sketch & Remark 2:**
  - Corrected proof sketch: relaxes the problem to $\text{IC}_L$ alone binding. Because $\Pi_{\kappa_H}$ contains $\mu_A U(\cdot;\kappa_H)$ directly, the leader has no rent-minimization motive to depress $\text{IR}_H$ to reservation utility $\underline{U}$ (unlike classical Baron--Myerson transfer models). Thus, $\text{IR}_H$ is strictly slack.
  - Remark 2 explicitly cites our numerical audit and caveats that under extreme bias, $\Pi_{\kappa_H}$ and $U(\cdot;\kappa_H)$ could diverge enough that $\text{IC}_H$ also binds.

---

## 2. Solver Overhaul (`underspec_sim/model1_pooling/solver.py`)

- **Explicit Second-Order Check:**
  - `solve_pooling_equilibrium` now explicitly calculates $\text{SOC} = \Delta\bar\gamma$.
  - **Convex / Corner Branch ($\Delta\bar\gamma \le 0$):** Directly compares $\Pi(1)$ vs $\Pi(0)$, returns $a^{SE} \in \{0.0, 1.0\}$, and assigns `regime = "corner"`.
  - **Concave / Interior Branch ($\Delta\bar\gamma > 0$):** Computes the unconstrained stationary point from Proposition 4, clips to $[0, 1]$, and assigns `regime = "interior"` (or `"corner"` if clipping binds).
- **Result Object Extension:**
  - `PoolingEquilibriumResult` includes `regime`, `is_corner`, `soc_value`, `is_concave`, and `pi_diff`. Implements `__float__` returning `self.a_SE` for transparent scalar operations.

---

## 3. Unit Tests Added (`tests/test_pooling.py`)

- Added `test_unbiased_pooling_always_corner_matching_formula`:
  - Sweeps a 4D grid of $(\Lambda, c_Q, k, F)$ across 81 parameter configurations.
  - Confirms `regime == "corner"`, $a^{SE} \in \{0, 1\}$, $\text{SOC} \le 0$, and exact agreement with the sign of $(\Lambda-c_Q)[\bar R_0 - c_Q\mathbb{E}[1/\kappa]]$ in 100% of cases.

---

## 4. Verification Suites Updated (`underspec_sim/verifications/`)

- **`verify_cor3.py`:**
  - Tests both $a^{SE}=0$ and $a^{SE}=1$ corners in the unbiased case, along with the interior stationary point in the biased regime.
  - Sweeps a parameter grid to confirm 100% formula match and outputs `outputs/cor3_interiority.csv`, `.png`, `.md`.
  - Result: **PASS**.
- **`verify_prop6.py` & Extended Active-Constraint Sweep:**
  - Confirms downward distortion of $a_H$ under representative bias ($\lambda_A=4.0, \mu_A=1.0$), with $a_L$ staying at $a_L^B=0$ and $\text{IC}_L$ binding alone.
  - **Task 4 Extended Audit:** Swept $\lambda_A \in [c_Q, 10 c_Q]$ (2.0 to 20.0) at $\mu_A=1.0$ and $\mu_A \in [0.1, 1.0]$ at $\lambda_A=4.0$ across 100 grid points in $(\lambda_A, \mu_A)$:
    - **Does $\text{IR}_H$ ever bind?** **NO** (Slack $\ge 75.0$ everywhere; binds in 0/100 points).
    - **Does $\text{IC}_H$ ever bind?** **YES** (Binds alongside $\text{IC}_L$ in extreme bias regimes $\lambda_A \ge 6.0$ or $\mu_A \le 0.5$).
  - Outputs `outputs/prop6_bias_sweep_active_constraints.csv`, `.png` and `outputs/prop6_screening_biased.csv`, `.png`, `.md`.
  - Result: **PASS**.
- **`compare_regimes_runner.py`:**
  - Evaluated against the corrected corner-picking pooling solver.
  - Dominance Corollary ($\Pi_{II} \ge \Pi_I$) holds in 100% of tested grid points (min gap $= 0.000000$, max gap $= 7.083333$).
  - Outputs updated `outputs/regime_comparison.csv`, `.png`, `.md`.
- **`sweep_distortion.py`:**
  - 3-panel visualization: distortion curve, distortion heatmap, and welfare-gap heatmap ($\Pi_{II} - \Pi_I$) with pooling regimes flagged per cell.
  - Outputs updated `outputs/distortion_sweep_2d.csv`, `.png`, `.md`.

---

## 5. Master Verification Runner (`run_all_math_checks.py`)

- Updated proposition descriptions and diagnostic notes.
- Summary table reports **PASS** across all rows:
  - Prop 3 (First-Best Threshold): **PASS**
  - Prop 4 (Pooling Equilibrium Closed Form): **PASS**
  - Cor 2 (Bias Direction on Pooling Rate): **PASS**
  - Cor 3 (Pooling is Corner Solution): **PASS**
  - Prop 5 (Screening Recovers First Best): **PASS**
  - Prop 6 (Screening Distortion Under Bias): **PASS**
- Process exits with return code `0`.

---

## 6. Strategic Underspecification v2 Robustness Checks Added

- **`verify_exact_conjunctive.py` (Task 1 Robustness):**
  - Swept $k \in [3, 20]$ and $g \in [0.50, 0.95]$ under exact non-linear success probability $q(a,g)^{k-m}$.
  - Confirmed Corollary 3 corner-solution property ($a^{SE} \in \{0, 1\}$) survives in **100% of tested grid points**; no interior compromise emerged.
  - Confirmed Proposition 6 downward distortion ($a_H^{SB} < a_H^B$) and slack $\text{IR}_H$ survive under exact conjunctive payoffs.
  - Verdict: **CAUTION** (Headline properties survive; boundary saturation at low $k \le 5$ dampens distortion).
- **`verify_assumption1.py` (Task 2 Robustness):**
  - Characterized Assumption 1 ($(k-m^*)\ln q \le -1$) across $(k, q, \kappa) \in [5, 15] \times [0.70, 0.95] \times [0.20, 2.00]$.
  - Found assumption holds in **10.1%** of grid points; fails when accuracy is high ($q \ge 0.90$) or user specification costs are low ($\kappa \le 0.60$).
  - Verdict: **CAUTION** (Confirms the paper's Remark 1 that Assumption 1 is restrictive).
- **`verify_mu_comparative_statics.py` (Task 3 Robustness):**
  - Derived and numerically confirmed the analytical ratio $\frac{\partial a^{SE}/\partial\mu_A}{\partial a^{SE}/\partial\lambda_A} = -\frac{\lambda_A}{\mu_A}$.
  - Plotted level curves showing that discounting ($\mu_A < 1$) and friction misperception ($\lambda_A > c_Q$) produce identical pooling ask rates along rays of constant $\lambda_A / \mu_A$.
  - Verdict: **PASS** (Confirms Remark `rmk:mu-lambda`).
- **Added 10 new pytest tests** in `tests/test_exact_conjunctive.py`, `tests/test_assumption1.py`, and `tests/test_mu_comparative_statics.py` (total test suite: 33 tests, 100% passing).

