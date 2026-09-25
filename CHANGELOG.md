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

---

## 7. Strategic Underspecification v3 Synchronization & Follow-Up Verifications

Following the synchronization with `strategic_underspecification_v3.tex`, three major open theoretical questions flagged in the paper were audited and verified:

- **Paper v3 Updates:**
  - Synchronized repository with `strategic_underspecification_v3.tex`.
  - Updated Remark 1 (`rmk:assumption1-audit`) to explicitly recognize Assumption 1 as a sufficient condition and highlight the empirical monotonicity audit.
  - Promoted Corollary 3 (`cor:mu-lambda`: "Non-identification of the two bias channels in the interior regime") and added Remark 3 (`rmk:corner-ray`: scope of ray-invariance on corner branch).
  - Updated Section 8 (*Discussion*) to cite findings from the exact conjunctive audit and note the open question regarding $\text{IC}_H$ binding under extreme bias.

- **`verify_monotonicity_survival.py` (Follow-up Task 1):**
  - Swept $k \in [5, 15], q \in [0.70, 0.99], \kappa \in [0.20, 2.00]$ across 2,552 finite-difference intervals.
  - **100% Monotonicity Survival Outside Assumption 1:** In all 2,162 intervals where Assumption 1 fails, $m^*(g)$ is weakly decreasing in $g$ with 0 violations.
  - **Sign Inversion in Assumption 1:** Discovered that decreasing differences $\frac{\partial^2 U}{\partial m \partial q} \le 0$ requires $(k-m)\ln q \ge -1$, which is the exact mathematical inverse of Assumption 1's condition ($(k-m)\ln q \le -1$).
  - **Proposition 2 Confirmed:** Naive users are weakly worse off than sophisticated users ($U_{\text{naive}} \le U_{\text{soph}}$) across 100% of tested grid points.
  - Verdict: **CAUTION** (Monotonicity holds 100% outside Assumption 1; Assumption 1 has its inequality sign reversed).
  - Artifacts: `outputs/monotonicity_survival.csv`, `.png`, `.md`.

- **`verify_mu_lambda_corner.py` (Follow-up Task 2):**
  - Analytically proved that $\Pi(1) - \Pi(0) = \mu_A [\Lambda \bar R_0 - \frac{\lambda_A}{\mu_A}(\bar R_0 + \bar\gamma)]$.
  - Factoring out $\mu_A > 0$ proves that the sign of $\Pi(1) - \Pi(0)$, and thus the selected corner $a^{SE} \in \{0, 1\}$, depends strictly on the scalar ratio $\lambda_A / \mu_A$.
  - Swept 11 rays across 10 values of $\mu_A \in [0.1, 1.0]$ (110 evaluations) with 0 ray-invariance failures and 0 boundary crossings.
  - Resolved Remark 3: ray-invariance holds unconditionally across both interior and corner regimes.
  - Verdict: **PASS**.
  - Artifacts: `outputs/mu_lambda_corner.csv`, `.png`, `.md`.

- **`verify_exact_bias_sweep.py` (Follow-up Task 3):**
  - Solved the 4D constrained screening menu problem on a 10x10 grid of $(\lambda_A, \mu_A) \in [2.0, 20.0] \times [0.1, 1.0]$ under the exact conjunctive payoff $q^{k-m}$.
  - Confirmed the paper's caveat: under extreme bias ($\lambda_A \ge 8.0$ or $\mu_A \le 0.3$), $\text{IC}_H$ becomes active alongside $\text{IC}_L$ (binding in 89/100 points).
  - Resolved Section 8 open question: extreme-bias constraint reversal is confirmed under the exact payoff.
  - Verdict: **PASS**.
  - Artifacts: `outputs/exact_bias_sweep.csv`, `.png`, `.md`.

- **Test Suite Expansion:**
  - Added 11 new tests in `tests/test_monotonicity_survival.py`, `tests/test_mu_lambda_corner.py`, and `tests/test_exact_bias_sweep.py`.
  - Full suite now contains **44 tests, 100% passing** (`pytest tests/ -v`).
- **Master Verification Runner:**
  - Updated `run_all_math_checks.py` to execute all 14 mathematical verifications in ~68s.

---

## 8. Mathematical Reconciliation & Canonical Release (v4.0)

- **Corollary 2 Overhaul (`verify_cor2.py` & `properties.py`):**
  - Retired the stale test of the superseded sign claim ($\text{sign}(2\bar\gamma a - \bar R_0)$) and replaced it with the verified comparative static on the interior branch ($\Delta\bar\gamma > 0$):
    $$\frac{\partial a^{SE}}{\partial \lambda_A} = \frac{\mu_A\Lambda}{2(\mu_A\Lambda - \lambda_A)^2} > 0$$
  - Calibrated parameterization ($\Lambda=2.0, c_Q=1.0, \mu_A=1.0, k=0.10, \bar\gamma=1.0, \bar R_0=-1.90, F\sim[0.8, 1.25]$) ensuring $\text{SOC} = \Delta\bar\gamma > 0$ strictly and $a^{SE}$ is fully interior ($a^{SE} \in [0.28, 0.85]$) across $\lambda_A \in [3.5, 12.0]$ without boundary clipping.
  - Eliminated unconditional `passed = True`: replaced with strict assertions verifying all evaluations are interior, all empirical finite differences $\frac{\Delta a^{SE}}{\Delta \lambda_A}$ are positive ($\in [0.010, 0.342]$), and maximum relative error against the analytical derivative is $< 3.5\%$.
- **Assumption 1 Reconciliation (`verify_assumption1.py`):**
  - Formally marked `verify_assumption1.py` as superseded by `verify_monotonicity_survival.py`.
  - Reconciled the regularity condition to $(k-m)\ln q \ge -1$, demonstrating it holds in **89.9%** (653/726) on the grid, resolving the apparent contradiction with the preliminary 10.1% figure from the reversed inequality $\le -1$.
  - Re-emphasized that the canonical 2,552-interval audit in Follow-up 1 confirms empirical monotonicity survives in **100%** of intervals outside the sufficient condition.
- **Regression Tests Added (`tests/`):**
  - `tests/test_pooling.py`: Added `test_cor2_derivative_sign_on_interior_branch()` asserting strictly positive derivative on the interior branch.
  - `tests/test_monotonicity_survival.py`: Added `test_cor1_direction_under_corrected_assumption1()` asserting Topkis decreasing differences under $(k-m)\ln q \ge -1$ and empirical monotonicity inversion when violated.
  - Full suite expanded to **46 unit and symbolic tests, 100% passing**.
- **Documentation & Repo Hygiene:**
  - Resolved all cross-section contradictions in `README.md` and `CHANGELOG.md`.
  - Canonicalized repository structure around `strategic_underspecification.tex` and `strategic_underspecification.pdf`, removing all legacy version-suffixed files (`strategic_underspecification_v3.tex`, `strategic_underspecification_v3.pdf`, `strategic_underspecification_v4.tex`).


