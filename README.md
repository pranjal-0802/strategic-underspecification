# Strategic Under-Specification

Numerical verification, simulation, and LLM-based experimentation framework for *"Strategic Under-Specification: A Stackelberg Game Between User and Assistant"* (Pranjal Agarwal, BITS Pilani).

---

## 1. Overview & Architecture

When users query AI coding assistants, they face a fundamental trade-off: spend cognitive effort specifying detailed constraints, edge cases, and architectural invariants, or leave them implicit and rely on the assistant to guess or ask clarifying questions. The assistant, in turn, commits to an asking policy: either a single population-wide ask rate (**Model I: Pooling**) or an incentive-compatible menu of asking rates conditioned on user specification effort (**Model II: Screening**).

This repository provides complete, end-to-end computational verification of the paper's theoretical framework:
- **First-Best Benchmark (Section 3.2):** Fully-informed, welfare-aligned social optimum $(m^{FB}, a^{FB})$ where asking weakly dominates guessing across the physically feasible domain $m \in [0, k]$.
- **Model I: Pooling Policy (Section 4):** The assistant commits to a single population-wide ask rate $a \in [0, 1]$. In the unbiased baseline, leader payoff is weakly convex, making pooling a corner solution ($a^{SE} \in \{0, 1\}$, Corollary 5); in the biased regime, strict concavity produces an interior optimum, where perceived friction suppresses clarification ($\partial a^{SE}/\partial\lambda_A \le 0$, Corollary 2).
- **Model II: Screening Policy (Section 5):** The assistant offers a menu $\{(m_L, a_L), (m_H, a_H)\}$ separating high- and low-cost specification types. Without bias, screening achieves First-Best (Proposition 5); under bias, asking rates are distorted downward (Proposition 6).
- **Regime Comparison (Section 6):** Weak dominance of screening over pooling and decomposition of welfare losses (Corollary 6).
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
| **Sec 3.2** | **Prop 3** (First-Best Policy on Feasible Domain) | `underspec_sim/verifications/verify_prop3.py` | `tests/test_first_best.py` | `outputs/prop3_first_best.csv`, `.png`, `.md` | **PASS** *(verified on grid)* |
| **Sec 4.2** | **Prop 4** (Stackelberg Pooling Rate $a^{SE}$) | `underspec_sim/verifications/verify_prop4.py` | `tests/test_pooling.py` | `outputs/prop4_pooling_closed_form.csv`, `.png`, `.md` | **PASS** *(matches grid under SOC)* |
| **Sec 4.2** | **Cor 2** (Friction Bias Suppresses Pooling Rate) | `underspec_sim/verifications/verify_cor2.py` | `tests/test_pooling.py` | `outputs/cor2_bias_direction.csv`, `.png`, `.md` | **PASS** *(strictly $\partial a^{SE}/\partial\lambda_A \le 0$ on valid domain)* |
| **Sec 4.2** | **Cor 3** (Non-Identification of Bias Channels) | `underspec_sim/verifications/verify_mu_comparative_statics.py` | `tests/test_mu_comparative_statics.py` | `outputs/mu_comparative_statics.csv`, `.png`, `.md` | **PASS** *(derivative ratio $= -\tilde\rho$)* |
| **Sec 4.2** | **Cor 4** (Ray-Invariance in Corner Regime) | `underspec_sim/verifications/verify_mu_lambda_corner.py` | `tests/test_mu_lambda_corner.py` | `outputs/mu_lambda_corner.csv`, `.png`, `.md` | **PASS** *(sign of $\Delta\Pi$ ray-invariant)* |
| **Sec 4.2** | **Cor 5** (Pooling is a Corner Solution in Unbiased Case) | `underspec_sim/verifications/verify_cor3.py` | `tests/test_pooling.py` | `outputs/cor3_interiority.csv`, `.png`, `.md` | **PASS** *(unbiased is corner; biased is interior)* |
| **Sec 5.2** | **Prop 5** (Zero Distortion Without Bias) | `underspec_sim/verifications/verify_prop5.py` | `tests/test_screening.py` | `outputs/prop5_screening_no_bias.csv`, `.png`, `.md` | **PASS** *(recovers FB; IC slack)* |
| **Sec 5.3** | **Prop 6** (Downward Distortion Under Bias) | `underspec_sim/verifications/verify_prop6.py` | `tests/test_screening.py` | `outputs/prop6_screening_biased.csv`, `.png`, `.md`, `outputs/prop6_bias_sweep_active_constraints.csv`, `.png` | **PASS** *(active set characterized)* |
| **Sec 5.3** | **Remark** (Heterogeneity vs Bias Separability) | `underspec_sim/verifications/sweep_distortion.py` | `tests/test_screening.py` | `outputs/distortion_sweep_2d.csv`, `.png`, `.md` | **PASS** *(2D sweep verified)* |
| **Sec 6** | **Cor 6** (Dominance of Menus over Pooling) | `underspec_sim/verifications/compare_regimes_runner.py` | `tests/test_comparison.py` | `outputs/regime_comparison.csv`, `.png`, `.md` | **PASS** *(dominance verified across grid)* |
| **Sec 8** | **Robustness 1** (Exact vs Linear-Risk Conjunctive Model) | `underspec_sim/verifications/verify_exact_conjunctive.py` | `tests/test_exact_conjunctive.py` | `outputs/exact_conjunctive_robustness.csv`, `.png`, `.md` | **CAUTION** *(corner survives in 100% of tested points)* |
| **Sec 8** | **Robustness 2** (Assumption 1 Regularity Grid Audit) | `underspec_sim/verifications/verify_assumption1.py` | `tests/test_assumption1.py` | `outputs/assumption1_regularity.csv`, `.png`, `.md` | **CAUTION** *(corrected $\ge -1$ holds in 89.9%)* |
| **Sec 8** | **Robustness 3** (Monotonicity Survival & Tightness) | `underspec_sim/verifications/verify_monotonicity_survival.py` | `tests/test_monotonicity_survival.py` | `outputs/monotonicity_survival.csv`, `.png`, `.md` | **PASS** *(mono holds 100% under Ass1; breaks in 96.7% outside)* |
| **Sec 8** | **Robustness 4** (Exact Bias Sweep & $\text{IC}_H$ Binding) | `underspec_sim/verifications/verify_exact_bias_sweep.py` | `tests/test_exact_bias_sweep.py` | `outputs/exact_bias_sweep.csv`, `.png`, `.md` | **PASS** *(extreme-bias reversal observed under exact payoff)* |

---

## 3. Mathematical Audits and Analytical Findings

The computational verification suite evaluates the analytical claims across representative parameter grids:

### 1. Corollary 5 (Pooling is a Corner Solution in the Unbiased Case)
- **Proposition 4** provides the closed-form pooling stationary point derived directly from primitives:
  $$\Pi(a) = \text{const} + \bar L\,a + \bar Q\,a^2, \qquad a^{SE} = -\frac{\bar L}{2\bar Q} = -\frac{(\mu_A s - b)\bar R_0}{(\mu_A s - 2b)\bar\gamma}$$
  provided $\bar Q < 0$ (strict concavity SOC, equivalent to $2(\lambda_A - c_Q) > \mu_A(\Lambda - c_Q)$), where $s \equiv \Lambda - c_Q$ and $b \equiv \lambda_A - c_Q$.
- In the unbiased baseline ($\mu_A = 1, \lambda_A = c_Q \implies b = 0$):
  $$\bar Q = \frac{s^2\,\bar\gamma}{2} = \frac{(\Lambda - c_Q)^2\,\mathbb{E}[1/\kappa]}{2} \ge 0 \quad \text{always}$$
- Because $\bar Q \ge 0$, **the leader payoff $\Pi(a)$ is weakly convex in $a$** on $[0, 1]$.
- Any weakly convex function on a compact interval achieves its maximum at a **boundary corner** ($a = 0$ or $a = 1$).
- Direct comparison in unclipped algebra yields the selection criterion:
  $$\Pi(1) - \Pi(0) = \bar L + \bar Q = (\Lambda - c_Q)\left(\bar R_0 + \frac{\bar\gamma}{2}\right) = (\Lambda - c_Q)\left[k - \frac{\Lambda + c_Q}{2}\mathbb{E}\left[\frac{1}{\kappa}\right]\right]$$
  When followers are constrained to the feasible domain $m \le k$, Proposition 3 implies that asking ($a^{SE} = 1$) weakly dominates guessing unconditionally under linear risk.
- **Biased Regime:** Away from the unbiased baseline, when friction is sufficiently high ($2b > \mu_A s$), $\bar Q < 0$ and $\Pi(a)$ becomes strictly concave, producing a true interior stationary maximum.
- The solver and automated verification suite agree with this closed-form selection rule across all tested configurations.

### 2. Corollary 2 (Friction Bias Suppresses Pooling Rate on Interior Branch)
- On the interior branch ($\bar Q < 0$ with $a^{SE} \in (0, 1)$), direct differentiation of Proposition 4's closed form yields:
  $$\frac{\partial a^{SE}}{\partial \lambda_A} = -\frac{\bar R_0}{\bar\gamma}\frac{\mu_A s}{(\mu_A s - 2b)^2} \le 0$$
  which is strictly non-positive on the valid domain $\bar R_0 \ge 0$.
- Over a calibrated parameter sweep ($k=3, \Lambda=2.0, c_Q=1.0, \lambda_A \in [1.75, 1.95]$), $a^{SE}$ falls monotonically from $0.51$ to $0.06$, matching fine-grid optima within $0.5\%$ relative error.

### 3. Proposition 6 & Active Constraint Audit (Screening Under Bias)
- **Down-and-Out Distortion:** At representative under-asking bias ($0 < \lambda_A - c_Q < \mu_A(\Lambda - c_Q)$, so $c_Q < c_Q^{eff} < \Lambda$), the low-cost type remains at $a_L^{SB} = a_L^B = 0$, while the high-cost type's asking rate is strictly distorted downward ($a_H^{SB} < a_H^B$).
- **Active Constraint Set Audit:**
  - In representative bias regions ($\lambda_A = 4.0, \mu_A = 1.0$), **$\text{IC}_L$ binds alone**; $\text{IR}_H$, $\text{IC}_H$, and $\text{IR}_L$ are strictly slack.
  - *Economic Intuition:* Unlike classical Baron--Myerson transfer models where the principal pays cash rents, here $\Pi_{\kappa_H}$ directly contains user utility $\mu_A U(\cdot;\kappa_H)$. The leader has no rent-minimization incentive to push $U_H$ down to $\underline{U}$.
- **Extended Sweep:**
  - **Does $\text{IR}_H$ ever bind?** **NO** (Slack $\ge 75.0$ across all 100 tested configurations).
  - **Does $\text{IC}_H$ ever bind?** **YES** (Under extreme friction $\lambda_A \ge 6.0$ or low altruism $\mu_A \le 0.5$, $\text{IC}_H$ binds alongside $\text{IC}_L$, confirming Remark 2).

### 4. Exact Conjunctive Robustness
- Evaluated whether headline results survive replacing the linear-risk approximation $L(1-q)(k-m)$ with the exact conjunctive probability $q(a,g)^{k-m}$:
  - **Unbiased Pooling Corner Property:** Survives in tested $(k, g) \in [3, 20] \times [0.50, 0.95]$ configurations ($a^{SE}_{\text{exact}} \in \{0, 1\}$).
  - **Screening Downward Distortion:** Survives with $a_H^{SB} < a_H^B$ whenever $m$ does not saturate at $k$. $\text{IR}_H$ remains strictly slack in all tested cases.

### 5. Regularity Assumption 1: Resolution & Monotonicity Survival
- **Mathematical Form & Direction:** The cross-partial of exact user utility is:
  $$\frac{\partial^2 U}{\partial m\,\partial g} = -V(1-a)\,q^{\,k-m-1}\Big[(k-m)\ln q + 1\Big]$$
  Topkis decreasing differences ($\le 0$) mathematically requires $(k - m^*)\ln q \ge -1$ (equivalently $(k - m^*)\ln q + 1 \ge 0$).
- **Empirical Monotonicity & Tightness:**
  - Across tested finite-difference parameter intervals in $(k, q, \kappa) \in [5, 15] \times [0.70, 0.99] \times [0.20, 2.00]$, whenever Assumption 1 holds ($(k-m^*)\ln q \ge -1$), $m^*(g)$ is weakly decreasing in $g$ without exception.
  - Where Assumption 1 fails ($(k-m^*)\ln q < -1$), the cross-partial turns positive and monotonicity breaks in the vast majority of intervals, confirming the necessity of the regularity condition.

### 6. Corollary 3: Bias Channel Identification & Ray-Invariance
- Derived the relationship between the two bias channels in pooling:
  $$\frac{\partial a^{SE} / \partial \mu_A}{\partial a^{SE} / \partial \lambda_A} = -\frac{\lambda_A - c_Q}{\mu_A} = -\tilde\rho$$
- Level curves of $a^{SE}$ form constant rays along $\tilde\rho = \text{constant}$, establishing observational equivalence along rays of normalized friction.

### 7. Corollary 4: Ray-Invariance in the Corner Regime
- In the corner regime, the selection difference satisfies $\Pi(1) - \Pi(0) = \mu_A \cdot \left[ s(\bar R_0 + \bar\gamma/2) - \tilde\rho(\bar R_0 + \bar\gamma) \right]$.
- Because $\mu_A > 0$ factors out cleanly, the sign of $\Pi(1) - \Pi(0)$ depends strictly and solely on the ratio $\tilde\rho = (\lambda_A - c_Q)/\mu_A$.

### 8. Constraint Co-activity ($\text{IC}_H$) Under Extreme Bias
- Solving the 4D constrained screening menu problem under the exact conjunctive payoff $q^{k-m}$ confirms that under severe friction ($\lambda_A \ge 8.0$ at $\mu_A = 1.0$) or low altruism ($\mu_A \le 0.3$), $\text{IC}_H$ becomes co-active alongside $\text{IC}_L$.

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
Runs 49 comprehensive algebraic, numerical, regression, and symbolic tests:
```bash
pytest tests/ -q
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
├── CITATION.cff                        # Machine-readable citation metadata for GitHub
├── CHANGELOG.md                        # Version history and release notes
├── pyproject.toml                      # Package configuration & dependencies
├── .gitignore                          # Ignored artifacts, virtual environments, and secrets
├── paper/                              # Sole canonical manuscript directory
│   ├── strategic_underspecification.tex# Full LaTeX source of the paper
│   ├── strategic_underspecification.pdf# Compiled preprint of the paper
│   └── figures/                        # High-resolution figures from verification suite
├── underspec_sim/                      # Core simulation & verification package
│   ├── core/                           # Primitives, payoffs, best-response, first-best
│   ├── model1_pooling/                 # Quadratic pooling payoff & closed-form solver
│   ├── model2_screening/               # Constrained menu solver & active-set audit
│   ├── comparison/                     # Regime dominance & welfare decomposition
│   ├── verifications/                  # 14 standalone proposition & robustness runners
│   ├── llm/                            # Anthropic client with retry, SQLite logging, dry-run
│   └── experiments_llm/                # Simulated user experiments
├── tests/                              # Pytest test suite (48 unit, regression & symbolic tests)
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

