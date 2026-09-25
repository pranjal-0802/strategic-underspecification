# underspec_sim

Numerical verification, simulation, and LLM-driven experimentation framework for the formal Stackelberg-game model of **strategic under-specification** in AI coding assistants, based on [*Strategic Under-Specification: A Stackelberg Game Between User and Assistant*](strategic_underspecification.tex) (Pranjal Agarwal, BITS Pilani).

---

## 1. Overview & Architecture

The framework instantiates both policy regimes analyzed in the paper:
- **First-Best Benchmark (Section 4.2):** Fully-informed, welfare-aligned social optimum $(m^{FB}, a^{FB})$.
- **Model I: Pooling Policy (Section 5):** The assistant (leader) commits to a single population-wide ask rate $a \in [0, 1]$; heterogeneous users best-respond.
- **Model II: Screening Policy (Section 6):** The assistant offers an incentive-compatible menu of bundles $\{(m_L, a_L), (m_H, a_H)\}$ conditioned on specification effort.
- **Regime Comparison (Section 7):** Dominance of menus over pooling and decomposition of welfare losses.
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

| Paper Section | Proposition / Corollary | Script / Module | Test File | Primary Artifacts | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sec 4.2** | **Prop 3** (Bang-Bang First-Best Threshold) | `underspec_sim/verifications/verify_prop3.py` | `tests/test_first_best.py` | `prop3_first_best.csv`, `.png`, `.md` | **PASS** |
| **Sec 5.2** | **Prop 4** (Stackelberg Pooling Rate $a^{SE}$) | `underspec_sim/verifications/verify_prop4.py` | `tests/test_pooling.py` | `prop4_pooling_closed_form.csv`, `.png`, `.md` | **PASS** (under SOC) |
| **Sec 5.2** | **Cor 2** (Bias Shifts Pooling Rate) | `underspec_sim/verifications/verify_cor2.py` | `tests/test_pooling.py` | `cor2_bias_direction.csv`, `.png`, `.md` | **PASS** |
| **Sec 5.2** | **Cor 3** (Pooling is a Corner Solution) | `underspec_sim/verifications/verify_cor3.py` | `tests/test_pooling.py` | `cor3_interiority.csv`, `.png`, `.md` | **PASS** *(formula verified)* |
| **Sec 6.2** | **Prop 5** (Zero Distortion Without Bias) | `underspec_sim/verifications/verify_prop5.py` | `tests/test_screening.py` | `prop5_screening_no_bias.csv`, `.png`, `.md` | **PASS** |
| **Sec 6.3** | **Prop 6** (Downward Distortion Under Bias) | `underspec_sim/verifications/verify_prop6.py` | `tests/test_screening.py` | `prop6_screening_biased.csv`, `.png`, `.md`, `prop6_bias_sweep_active_constraints.csv`, `.png` | **PASS** *(active set characterized)* |
| **Sec 6.3** | **Remark** (Heterogeneity vs Bias Separability) | `underspec_sim/verifications/sweep_distortion.py` | `tests/test_screening.py` | `distortion_sweep_2d.csv`, `.png`, `.md` | **PASS** |
| **Sec 7** | **Cor 4** (Dominance of Menus over Pooling) | `underspec_sim/verifications/compare_regimes_runner.py` | `tests/test_comparison.py` | `regime_comparison.csv`, `.png`, `.md` | **PASS** |
| **Robustness 1** | **Exact vs Linear-Risk Conjunctive Model** | `underspec_sim/verifications/verify_exact_conjunctive.py` | `tests/test_exact_conjunctive.py` | `exact_conjunctive_robustness.csv`, `.png`, `.md` | **CAUTION** *(corner survives 100%; boundary saturation at low k)* |
| **Robustness 2** | **Assumption 1 Regularity Characterization** | `underspec_sim/verifications/verify_assumption1.py` | `tests/test_assumption1.py` | `assumption1_regularity.csv`, `.png`, `.md` | **CAUTION** *(holds in 10.1%; restrictive for q>0.9 or low kappa)* |
| **Robustness 3** | **$\mu_A$ Comparative Statics & Confound** | `underspec_sim/verifications/verify_mu_comparative_statics.py` | `tests/test_mu_comparative_statics.py` | `mu_comparative_statics.csv`, `.png`, `.md` | **PASS** *(Remark rmk:mu-lambda verified: ratio = -lambda/mu)* |

---

## 3. Mathematical Audits and Analytical Findings

The numerical simulations verified the core mechanics and informed the paper's theoretical corrections:

### 1. Corollary 3 (Pooling is a Corner Solution, Not a Compromise)
- **Proposition 4** provides the closed-form pooling stationary point:
  $$a^{SE} = -\frac{C_0\bar\gamma + \Delta\bar R_0}{2\Delta\bar\gamma}, \quad \text{provided } \Delta\bar\gamma > 0 \text{ (SOC)}$$
- In the unbiased baseline ($\mu_A = 1, \lambda_A = c_Q$):
  $$\Delta = c_Q - \Lambda, \quad \bar\gamma = (\Lambda - c_Q)\mathbb{E}[1/\kappa] \implies \Delta\bar\gamma = -(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] \le 0$$
- Because the quadratic term in $\Pi(a)$ is $-a^2 \Delta \bar\gamma \ge 0$, **the leader payoff $\Pi(a)$ is weakly convex in $a$**!
- Any weakly convex function on a closed interval $[0, 1]$ achieves its maximum at a **boundary corner** ($a = 0$ or $a = 1$), never in the interior $(0, 1)$.
- Direct comparison yields the exact selection criterion:
  $$\Pi(1) - \Pi(0) = (\Lambda - c_Q)\Big[\bar R_0 - c_Q \mathbb{E}[1/\kappa]\Big]$$
  If positive, $a^{SE} = 1$; if negative, $a^{SE} = 0$.
- **Biased Regime:** Away from the unbiased baseline, when $(\lambda_A - \mu_A \Lambda)$ and $(\Lambda - c_Q)$ share a sign, $\Delta\bar\gamma > 0$ and $\Pi(a)$ becomes strictly concave, producing a true interior stationary maximum.
- The solver and test suite confirm 100% agreement with this closed-form rule.

### 2. Proposition 6 & Active Constraint Audit (Screening Under Bias)
- **Down-and-Out Distortion:** At representative under-asking bias ($\lambda_A > \mu_A c_Q$), the low-cost type remains at $a_L^{SB} = a_L^B = 0$, while the high-cost type's asking rate is strictly distorted downward ($a_H^{SB} < a_H^B$).
- **Active Constraint Set Audit:**
  - In representative bias regions ($\lambda_A = 4.0, \mu_A = 1.0$), **$\text{IC}_L$ binds alone**; $\text{IR}_H$, $\text{IC}_H$, and $\text{IR}_L$ are strictly slack.
  - *Economic Intuition:* Unlike classical Baron--Myerson transfer models where the principal pays cash rents, here $\Pi_{\kappa_H}$ directly contains user utility $\mu_A U(\cdot;\kappa_H)$. The leader has no rent-minimization incentive to push $U_H$ down to $\underline{U}$.
- **Extended Sweep (100 configurations in $(\lambda_A, \mu_A)$ plane):**
  - **Does $\text{IR}_H$ ever bind?** **NO** (Slack $\ge 75.0$ everywhere; binds in 0/100 points).
  - **Does $\text{IC}_H$ ever bind?** **YES** (Under extreme friction $\lambda_A \ge 6.0$ or low altruism $\mu_A \le 0.5$, $\text{IC}_H$ binds alongside $\text{IC}_L$, confirming the paper's caveat in Remark 2).

### 3. Exact Conjunctive Robustness (Task 1)
- Evaluated whether headline results survive replacing the linear-risk approximation $L(1-q)(k-m)$ with the exact conjunctive probability $q(a,g)^{k-m}$:
  - **Unbiased Pooling Corner Property:** Survives in **100% of tested $(k, g) \in [3, 20] \times [0.50, 0.95]$ configurations** ($a^{SE}_{\text{exact}} \in \{0, 1\}$). No interior compromise emerged anywhere.
  - **Screening Downward Distortion:** Survives with $a_H^{SB} < a_H^B$ whenever $m$ does not saturate at $k$ (strict in 66.7% of grid, weak in 100%). $\text{IR}_H$ remains strictly slack everywhere.

### 4. Regularity Assumption 1 Characterization (Task 2)
- Evaluated $(k - m^*)\ln q \le -1$ across $(k, q, \kappa) \in [5, 15] \times [0.70, 0.95] \times [0.2, 2.0]$:
  - **Satisfaction Rate:** Holds in only **10.1%** of the grid.
  - Fails when users have low specification costs ($\kappa \le 0.60$) because $m^* \to k \implies (k-m^*)\ln q \to 0 > -1$.
  - Fails when accuracy is high ($q \ge 0.90$) because satisfying the bound requires $k-m^* \ge 1/|\ln q| \ge 20$, exceeding the attribute budget $k \le 15$.
  - Holds when guessing is noticeably noisy ($q \le 0.85$) and users have high specification costs ($\kappa \ge 1.0$).

### 5. $\mu_A$ Comparative Statics & Confounding (Task 3)
- Derived the exact relationship between the two bias channels in pooling:
  $$\frac{\partial a^{SE} / \partial \mu_A}{\partial a^{SE} / \partial \lambda_A} = -\frac{\lambda_A}{\mu_A}$$
- Level curves of $a^{SE}$ form constant rays along $\lambda_A / \mu_A = \text{constant}$.
- Proves Remark `rmk:mu-lambda`: an assistant with discounting ($\mu_A < 1$) behaves identically along the ask-rate margin to an assistant with friction misperception ($\lambda_A > c_Q$). The two parameters are observationally confounded from $a^{SE}$ alone.

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
Runs all 8 verification suites in under 3 seconds, generates all CSVs and PNGs in `outputs/`, and prints a formatted summary table:

```bash
# Strict mode: exits nonzero (1) if any proposition fails (as requested before spending LLM budget)
python3 run_all_math_checks.py

# Non-strict reporting mode: prints table and exits 0
python3 run_all_math_checks.py --ignore-failures
```

### Running Unit Tests (pytest)
Runs 22 comprehensive algebraic and symbolic tests:
```bash
pytest tests/ -v
```

---

## 5. LLM-Driven Experiments

The package includes an empirical LLM layer (`underspec_sim/experiments_llm/`):
- Configurable model string (defaults to `claude-sonnet-4-6`).
- Automatic fallback to deterministic `--dry-run` stub so the entire suite runs free of API cost.
- SQLite audit logging of every prompt, response, latency, and metadata in `runs.db`.

```bash
# Run deterministic dry-run experiment (default, free):
python3 underspec_sim/experiments_llm/run_all_llm_experiments.py

# Run live with Anthropic API (requires ANTHROPIC_API_KEY in environment):
export ANTHROPIC_API_KEY="your-api-key"
python3 underspec_sim/experiments_llm/run_all_llm_experiments.py --live --model claude-sonnet-4-6
```

---

## 6. Known Gaps & Unverified Extensions

Per Section 10 (*Discussion and Limitations*) of the paper:
1. **Conjunctive vs. Linear-Risk Robustness Check:**
   The paper relies on the linear-risk approximation expected loss $\approx \Lambda(1-a)(k-m)$ instead of the exact conjunctive probability $q(a, g)^{k-m} = (a + (1-a)g)^{k-m}$. While single-crossing and bang-bang thresholds generalize qualitatively, the exact conjunctive form introduces higher-order terms in $(k-m)$ that can create interior pooling equilibria.
2. **Continuum-Type Screening Extension:**
   Model II is restricted to two types $\{\kappa_L, \kappa_H\}$ with discrete shares $\{f_L, f_H\}$. Extending screening to a continuous distribution $\kappa \sim F[\kappa_L, \kappa_H]$ requires the full Mirrleesian optimal control formulation with an ironing condition if monotonicity fails.
3. **Truthful Specification Revelation:**
   The model assumes users truthfully reveal $m$ attributes conditional on choosing to specify them. Adding a cheap-talk layer where users can provide vague or noisy attributes is an open extension.
4. **Competitive Assistant Market:**
   The current model assumes a monopoly Stackelberg leader assistant. Market competition among assistants could discipline under-asking bias $\lambda_A > c_Q$.
