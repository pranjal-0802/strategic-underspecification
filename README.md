# underspec_sim

Numerical verification, simulation, and LLM-driven experimentation framework for the formal Stackelberg-game model of **strategic under-specification** in AI coding assistants, based on [*Strategic Under-Specification: A Stackelberg Game Between User and Assistant*](paper/strategic_underspecification.tex) (Pranjal Agarwal, BITS Pilani).

---

## 1. Overview & Architecture

When users query AI coding assistants, they face a fundamental trade-off: spend cognitive effort specifying detailed constraints, edge cases, and architectural invariants, or leave them implicit and rely on the assistant to guess or ask clarifying questions. The assistant, in turn, commits to an asking policy—either a single population-wide ask rate (**Model I: Pooling**) or an incentive-compatible menu of asking rates conditioned on user specification effort (**Model II: Screening**).

This repository provides complete, end-to-end computational verification of the paper's theoretical framework:
- **First-Best Benchmark (Section 4.2):** Fully-informed, welfare-aligned social optimum $(m^{FB}, a^{FB})$ with a bang-bang threshold $\kappa^*$.
- **Model I: Pooling Policy (Section 5):** The assistant commits to a single population-wide ask rate $a \in [0, 1]$. In the unbiased baseline, leader payoff is weakly convex, making pooling a corner solution ($a^{SE} \in \{0, 1\}$); in the biased regime, strict concavity produces an interior optimum.
- **Model II: Screening Policy (Section 6):** The assistant offers a menu $\{(m_L, a_L), (m_H, a_H)\}$ separating high- and low-cost specification types. Without bias, screening achieves First-Best; under bias, asking rates are distorted downward.
- **Regime Comparison (Section 7):** Weak dominance of screening over pooling and decomposition of welfare losses.
- **LLM Experimentation Layer:** Simulated users with heterogeneous specification cost types $\kappa$ interact with assistants applying the derived policies, audited in SQLite (`runs.db`) with a deterministic `--dry-run` stub.

```
underspec_sim/
├── core/                   # Primitives (k, g, L, c_Q, V, mu_A, lambda_A), payoffs, best response, first-best
├── model1_pooling/         # Quadratic pooling payoff Pi(a), closed-form solver, interiority & bias checks
├── model2_screening/       # General SLSQP constrained menu solver over (mL, aL, mH, aH), constraint audits
├── comparison/             # Weak dominance verification and 2D welfare gap decomposition
├── verifications/          # Standalone verification runners producing CSVs, PNGs, and Markdown reports
├── llm/                    # Anthropic client (safe env ANTHROPIC_API_KEY, retry, SQLite runs.db, dry-run stub)
└── experiments_llm/        # Simulated user LLM experiments with Claude Sonnet 4.6 default
```

---

## 2. Proposition & Corollary Verification Mapping

The table below maps every proposition, corollary, and robustness check in the paper to its verification script, unit test, primary artifacts, and computational result.

| Paper Section | Proposition / Corollary | Script / Module | Test File | Primary Artifacts | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sec 4.2** | **Prop 3** (Bang-Bang First-Best Threshold) | `underspec_sim/verifications/verify_prop3.py` | `tests/test_first_best.py` | `outputs/prop3_first_best.csv`, `.png`, `.md` | **PASS** *(verified on grid)* |
| **Sec 5.2** | **Prop 4** (Stackelberg Pooling Rate $a^{SE}$) | `underspec_sim/verifications/verify_prop4.py` | `tests/test_pooling.py` | `outputs/prop4_pooling_closed_form.csv`, `.png`, `.md` | **PASS** *(matches grid under SOC)* |
| **Sec 5.2** | **Cor 2** (Bias Shifts Pooling Rate on Interior Branch) | `underspec_sim/verifications/verify_cor2.py` | `tests/test_pooling.py` | `outputs/cor2_bias_direction.csv`, `.png`, `.md` | **PASS** *(strictly $\partial a^{SE}/\partial\lambda_A > 0$ on interior)* |
| **Sec 5.2** | **Cor 3** (Pooling is a Corner Solution) | `underspec_sim/verifications/verify_cor3.py` | `tests/test_pooling.py` | `outputs/cor3_interiority.csv`, `.png`, `.md` | **PASS** *(corner selected by sign of $\Pi(1)-\Pi(0)$)* |
| **Sec 6.2** | **Prop 5** (Zero Distortion Without Bias) | `underspec_sim/verifications/verify_prop5.py` | `tests/test_screening.py` | `outputs/prop5_screening_no_bias.csv`, `.png`, `.md` | **PASS** *(recovers FB; IC slack)* |
| **Sec 6.3** | **Prop 6** (Downward Distortion Under Bias) | `underspec_sim/verifications/verify_prop6.py` | `tests/test_screening.py` | `outputs/prop6_screening_biased.csv`, `.png`, `.md`, `outputs/prop6_bias_sweep_active_constraints.csv`, `.png` | **PASS** *(active set characterized)* |
| **Sec 6.3** | **Remark** (Heterogeneity vs Bias Separability) | `underspec_sim/verifications/sweep_distortion.py` | `tests/test_screening.py` | `outputs/distortion_sweep_2d.csv`, `.png`, `.md` | **PASS** *(2D sweep verified)* |
| **Sec 7** | **Cor 4** (Dominance of Menus over Pooling) | `underspec_sim/verifications/compare_regimes_runner.py` | `tests/test_comparison.py` | `outputs/regime_comparison.csv`, `.png`, `.md` | **PASS** *(dominance verified across grid)* |
| **Robustness 1** | **Exact vs Linear-Risk Conjunctive Model** | `underspec_sim/verifications/verify_exact_conjunctive.py` | `tests/test_exact_conjunctive.py` | `outputs/exact_conjunctive_robustness.csv`, `.png`, `.md` | **CAUTION** *(corner survives in 100% of tested points)* |
| **Robustness 2** | **Assumption 1 Regularity Grid Audit** | `underspec_sim/verifications/verify_assumption1.py` | `tests/test_assumption1.py` | `outputs/assumption1_regularity.csv`, `.png`, `.md` | **CAUTION** *(corrected $\ge -1$ holds in 89.9%; see Follow-up 1)* |
| **Robustness 3** | **$\mu_A$ Comparative Statics & Confound** | `underspec_sim/verifications/verify_mu_comparative_statics.py` | `tests/test_mu_comparative_statics.py` | `outputs/mu_comparative_statics.csv`, `.png`, `.md` | **PASS** *(Remark rmk:mu-lambda verified: ratio $= -\lambda_A/\mu_A$)* |
| **Follow-up 1 (Task 1)** | **Monotonicity Survival Outside Assump 1** | `underspec_sim/verifications/verify_monotonicity_survival.py` | `tests/test_monotonicity_survival.py` | `outputs/monotonicity_survival.csv`, `.png`, `.md` | **CAUTION** *(mono survives in 100% of tested intervals outside Ass1)* |
| **Follow-up 2 (Task 2)** | **Ray-Invariance in Corner Regime** | `underspec_sim/verifications/verify_mu_lambda_corner.py` | `tests/test_mu_lambda_corner.py` | `outputs/mu_lambda_corner.csv`, `.png`, `.md` | **PASS** *(sign of $\Pi(1)-\Pi(0)$ ray-invariant in tested space)* |
| **Follow-up 3 (Task 3)** | **Exact Bias Sweep & $\text{IC}_H$ Binding** | `underspec_sim/verifications/verify_exact_bias_sweep.py` | `tests/test_exact_bias_sweep.py` | `outputs/exact_bias_sweep.csv`, `.png`, `.md` | **PASS** *(extreme-bias reversal observed under exact payoff)* |

---

## 3. Mathematical Audits and Analytical Findings

The computational verification suite evaluates the analytical claims across representative parameter grids:

### 1. Corollary 3 (Pooling is a Corner Solution, Not a Compromise)
- **Proposition 4** provides the closed-form pooling stationary point:
  $$a^{SE} = -\frac{C_0\bar\gamma + \Delta\bar R_0}{2\Delta\bar\gamma}, \quad \text{provided } \Delta\bar\gamma > 0 \text{ (SOC)}$$
- In the unbiased baseline ($\mu_A = 1, \lambda_A = c_Q$):
  $$\Delta = c_Q - \Lambda, \quad \bar\gamma = (\Lambda - c_Q)\mathbb{E}[1/\kappa] \implies \Delta\bar\gamma = -(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] \le 0$$
- Because the quadratic term in $\Pi(a)$ is $-a^2 \Delta \bar\gamma \ge 0$, **the leader payoff $\Pi(a)$ is weakly convex in $a$** on $[0, 1]$.
- Any weakly convex function on a compact interval achieves its maximum at a **boundary corner** ($a = 0$ or $a = 1$).
- Direct comparison yields the selection criterion:
  $$\Pi(1) - \Pi(0) = (\Lambda - c_Q)\Big[\bar R_0 - c_Q \mathbb{E}[1/\kappa]\Big]$$
  If positive, $a^{SE} = 1$; if negative, $a^{SE} = 0$.
- **Biased Regime:** Away from the unbiased baseline, when $(\lambda_A - \mu_A \Lambda)$ and $(\Lambda - c_Q)$ share a sign, $\Delta\bar\gamma > 0$ and $\Pi(a)$ becomes strictly concave, producing a true interior stationary maximum.
- The solver and automated verification suite agree with this closed-form selection rule across all tested configurations.

### 2. Corollary 2 (Bias Shifts Pooling Rate on Interior Branch)
- On the interior branch ($\Delta\bar\gamma > 0$ with $a^{SE} \in (0, 1)$), direct differentiation of Proposition 4's closed form yields:
  $$\frac{\partial a^{SE}}{\partial \lambda_A} = \frac{\mu_A\Lambda}{2(\mu_A\Lambda - \lambda_A)^2} > 0$$
- Over a calibrated parameter sweep ($\lambda_A \in [3.5, 12.0]$ with $\mu_A=1.0, \Lambda=2.0, c_Q=1.0$), $a^{SE}$ ranges strictly from $0.28$ to $0.85$ without hitting boundary corners.
- Verification (`verify_cor2.py`) confirms that empirical finite differences $\frac{\Delta a^{SE}}{\Delta \lambda_A} > 0$ everywhere (slopes $\in [0.010, 0.342]$) and match the analytical derivative within 3.31% relative error.

### 3. Proposition 6 & Active Constraint Audit (Screening Under Bias)
- **Down-and-Out Distortion:** At representative under-asking bias ($\lambda_A > \mu_A c_Q$), the low-cost type remains at $a_L^{SB} = a_L^B = 0$, while the high-cost type's asking rate is strictly distorted downward ($a_H^{SB} < a_H^B$).
- **Active Constraint Set Audit:**
  - In representative bias regions ($\lambda_A = 4.0, \mu_A = 1.0$), **$\text{IC}_L$ binds alone**; $\text{IR}_H$, $\text{IC}_H$, and $\text{IR}_L$ are strictly slack.
  - *Economic Intuition:* Unlike classical Baron--Myerson transfer models where the principal pays cash rents, here $\Pi_{\kappa_H}$ directly contains user utility $\mu_A U(\cdot;\kappa_H)$. The leader has no rent-minimization incentive to push $U_H$ down to $\underline{U}$.
- **Extended Sweep (100 configurations in $(\lambda_A, \mu_A)$ plane):**
  - **Does $\text{IR}_H$ ever bind?** **NO** (Slack $\ge 75.0$ across all 100 tested configurations).
  - **Does $\text{IC}_H$ ever bind?** **YES** (Under extreme friction $\lambda_A \ge 6.0$ or low altruism $\mu_A \le 0.5$, $\text{IC}_H$ binds alongside $\text{IC}_L$, confirming Remark 2).

### 4. Exact Conjunctive Robustness (Task 1)
- Evaluated whether headline results survive replacing the linear-risk approximation $L(1-q)(k-m)$ with the exact conjunctive probability $q(a,g)^{k-m}$:
  - **Unbiased Pooling Corner Property:** Survives in **100% of tested $(k, g) \in [3, 20] \times [0.50, 0.95]$ configurations** ($a^{SE}_{\text{exact}} \in \{0, 1\}$).
  - **Screening Downward Distortion:** Survives with $a_H^{SB} < a_H^B$ whenever $m$ does not saturate at $k$ (strict in 66.7% of grid, weak in 100%). $\text{IR}_H$ remains strictly slack in all tested cases.

### 5. Regularity Assumption 1: Resolution & Monotonicity Survival (Task 2 & Follow-up 1)
- **Mathematical Form & Direction:** The cross-partial of exact user utility is:
  $$\frac{\partial^2 U}{\partial m\,\partial g} = -V(1-a)\,q^{\,k-m-1}\Big[(k-m)\ln q + 1\Big]$$
  Topkis decreasing differences ($\le 0$) mathematically requires $(k - m^*)\ln q \ge -1$ (equivalently $(k - m^*)\ln q + 1 \ge 0$).
- **Grid Audit on Canonical 726-Point Grid** ($k \in [5, 15], q \in [0.70, 0.95], \kappa \in [0.20, 2.00]$):
  - Under the package benchmark valuation ($V = 100.0$), the corrected condition holds in **89.9%** (653 / 726) of configurations.
  - The legacy draft condition ($(k - m^*)\ln q \le -1$) held in **10.1%** (73 / 726) of configurations.
  - Under lower task valuations ($V = 10.0$, as in the paper's illustrative numerical example), users specify fewer attributes ($m^*$ lower, $k - m^*$ higher), and the condition holds in **34.3%** (249 / 726) of configurations (and 31.7% at $V = 8.0$).
- **Empirical Monotonicity Survival (`verify_monotonicity_survival.py`):**
  - Evaluated empirical monotonicity across 2,552 finite-difference parameter intervals in $(k, q, \kappa) \in [5, 15] \times [0.70, 0.99] \times [0.20, 2.00]$.
  - In all 2,162 intervals where the sufficient condition fails, $m^*(g)$ remains weakly decreasing in $g$ with **zero violations (100% survival on tested intervals)**.
- **Proposition 2 (Naive Welfare):** Naive users (who believe $a^\dagger > a_A$) suffer a welfare loss $U_{\text{naive}} \le U_{\text{soph}}$ in **100%** of tested grid points.

### 6. $\mu_A$ Comparative Statics & Confounding (Task 3)
- Derived the exact relationship between the two bias channels in pooling:
  $$\frac{\partial a^{SE} / \partial \mu_A}{\partial a^{SE} / \partial \lambda_A} = -\frac{\lambda_A}{\mu_A}$$
- Level curves of $a^{SE}$ form constant rays along $\lambda_A / \mu_A = \text{constant}$.
- Confirms Corollary 3 (`cor:mu-lambda`): an assistant with discounting ($\mu_A < 1$) behaves identically along the ask-rate margin to an assistant with friction misperception ($\lambda_A > c_Q$). The two parameters are observationally confounded from $a^{SE}$ alone.

### 7. Ray-Invariance in the Corner Regime (Follow-up Task 2)
- Resolved Remark 3: does ray-invariance extend to the corner regime ($\Delta\bar\gamma \le 0$) where the choice is $a^{SE} \in \{0, 1\}$?
  - **Analytical Derivation:** In the corner regime, $\Pi(1) - \Pi(0) = \mu_A \cdot \left[ \Lambda \bar R_0 - \frac{\lambda_A}{\mu_A} (\bar R_0 + \bar\gamma) \right]$.
  - Because $\mu_A > 0$ factors out cleanly, the sign of $\Pi(1) - \Pi(0)$ depends strictly and solely on the ratio $\lambda_A / \mu_A$.
  - **Numerical Sweep:** Tested 11 rays across 10 values of $\mu_A \in [0.1, 1.0]$ (110 evaluations). Zero boundary flips occurred along any ray.

### 8. Constraint Binding ($\text{IC}_H$) Under Exact Conjunctive Payoff (Follow-up Task 3)
- Solved the 4D constrained screening menu problem on a 10x10 grid of $(\lambda_A, \mu_A) \in [2.0, 20.0] \times [0.1, 1.0]$ under the exact conjunctive payoff $q^{k-m}$.
- Under severe friction ($\lambda_A \ge 8.0$ at $\mu_A = 1.0$) or low altruism ($\mu_A \le 0.3$), $\text{IC}_H$ becomes active alongside $\text{IC}_L$ in 89% of grid points (matching line 705 of the paper).

---

## 4. Setup and Quickstart

### Virtual Environment & Dependencies
Dependencies: `anthropic`, `numpy`, `scipy`, `pandas`, `matplotlib`, `sympy`, `pydantic`, `pytest`.

```bash
# Using uv (recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .

# Or standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Non-LLM Mathematical Verifications
Runs all 14 verification and follow-up robustness suites, generates all CSVs and PNGs in `outputs/`, and prints a formatted summary table:

```bash
# Strict mode: exits nonzero (1) if any proposition fails (offline, fast ~67s)
python3 run_all_math_checks.py

# Non-strict reporting mode: prints table and exits 0
python3 run_all_math_checks.py --ignore-failures
```

### Running Unit Tests (pytest)
Runs 46 comprehensive algebraic, numerical, regression, and symbolic tests:
```bash
pytest tests/ -v
```

---

## 5. LLM-Driven Experiments

The package includes an empirical LLM layer (`underspec_sim/experiments_llm/`):
- Configurable model string (defaults to `claude-sonnet-4-6`).
- Automatic fallback to deterministic `--dry-run` stub so the entire suite runs free of API cost.
- SQLite audit logging of every prompt, response, latency, and metadata in `runs.db` (git-ignored).
- Live execution requires setting `ANTHROPIC_API_KEY` in your environment (API calls may incur cost).

```bash
# Run deterministic dry-run experiment (default, free, no API key needed):
python3 underspec_sim/experiments_llm/run_all_llm_experiments.py

# Run live with Anthropic API:
export ANTHROPIC_API_KEY="your-api-key"
python3 underspec_sim/experiments_llm/run_all_llm_experiments.py --live --model claude-sonnet-4-6
```

---

## 6. Reproducibility Caveats & Scientific Scope

To ensure scientific traceability, this repository and manuscript maintain a clear distinction between levels of evidence:
1. **Analytical / Theoretical Results:** Propositions and corollaries formally proved via calculus, envelope theorems, and Topkis's theorem in `paper/strategic_underspecification.tex`.
2. **Numerical Verifications:** High-resolution grid sweeps and constrained optimization (`SLSQP`) validating closed forms, second-order conditions, and constraint activity sets.
3. **Simulation Evidence:** Parametric experiments comparing policy regimes and examining exact conjunctive non-linearities.
4. **LLM Experiments:** Prompt-based simulated agent experiments with structured logging.
5. **Human-Subject Study:** **The computational evaluation consists of analytical verification, numerical experiments, and LLM-based simulation experiments; no human-subject study is reported.** The human-subject study outlined in Section 9 of the paper represents a proposed experimental design for future empirical research.

---

## 7. Repository Layout

```
strategic-underspecification/
├── README.md                           # Comprehensive documentation & verification mapping
├── LICENSE                             # Dual license: MIT (Software) & CC-BY-4.0 (Manuscript)
├── CITATION.cff                        # Machine-readable citation metadata for GitHub (v4.0.0)
├── CHANGELOG.md                        # Version history and release notes
├── pyproject.toml                      # Package configuration & dependencies (v4.0.0)
├── .gitignore                          # Ignored artifacts, virtual environments, and secrets
├── paper/                              # Sole canonical manuscript directory
│   ├── strategic_underspecification.tex# Full LaTeX source of the paper (v4)
│   ├── strategic_underspecification.pdf# Compiled preprint of the paper
│   ├── references.bib                  # BibTeX bibliography for cited literature
│   └── figures/                        # High-resolution figures from verification suite
├── underspec_sim/                      # Core simulation & verification package
│   ├── core/                           # Primitives, payoffs, best-response, first-best
│   ├── model1_pooling/                 # Quadratic pooling payoff & closed-form solver
│   ├── model2_screening/               # Constrained menu solver & active-set audit
│   ├── comparison/                     # Regime dominance & welfare decomposition
│   ├── verifications/                  # 14 standalone proposition & robustness runners
│   ├── llm/                            # Anthropic client with retry, SQLite logging, dry-run
│   └── experiments_llm/                # Simulated user experiments
├── tests/                              # Pytest test suite (46 unit & symbolic tests)
├── outputs/                            # Generated artifacts (CSVs, high-res PNGs, Markdown)
└── run_all_math_checks.py              # Master runner executing all 14 mathematical checks
```

---

## 8. Citation

If you use this framework, reproduction code, or cite the formal Stackelberg model, please cite:

```bibtex
@article{agarwal2026underspec,
  title={Strategic Under-Specification: A Stackelberg Game Between User and Assistant},
  author={Agarwal, Pranjal},
  journal={Working Paper, BITS Pilani},
  year={2026},
  url={https://github.com/pranjal-0802/strategic-underspecification}
}
```

GitHub also supports direct citation via [`CITATION.cff`](CITATION.cff).

---

## 9. License

- **Software Source Code:** Licensed under the [MIT License](LICENSE).
- **Manuscript, Preprints & Analytical Documentation:** Licensed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](LICENSE).

