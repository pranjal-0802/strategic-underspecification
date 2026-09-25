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
| **Sec 5.2** | **Cor 3** (Pooling Generically Interior) | `underspec_sim/verifications/verify_cor3.py` | `tests/test_pooling.py` | `cor3_interiority.csv`, `.png`, `.md` | **FAIL** *(audit finding)* |
| **Sec 6.2** | **Prop 5** (Zero Distortion Without Bias) | `underspec_sim/verifications/verify_prop5.py` | `tests/test_screening.py` | `prop5_screening_no_bias.csv`, `.png`, `.md` | **PASS** |
| **Sec 6.3** | **Prop 6** (Downward Distortion Under Bias) | `underspec_sim/verifications/verify_prop6.py` | `tests/test_screening.py` | `prop6_screening_biased.csv`, `.png`, `.md` | **PASS** *(constraint audit)* |
| **Sec 6.3** | **Remark** (Heterogeneity vs Bias Separability) | `underspec_sim/verifications/sweep_distortion.py` | `tests/test_screening.py` | `distortion_sweep_2d.csv`, `.png`, `.md` | **PASS** |
| **Sec 7** | **Cor 4** (Dominance of Menus over Pooling) | `underspec_sim/verifications/compare_regimes_runner.py` | `tests/test_comparison.py` | `regime_comparison.csv`, `.png`, `.md` | **PASS** |

---

## 3. Mathematical Audits and Analytical Findings

The numerical simulations verified the core mechanics while uncovering two critical theoretical nuances in the paper's LaTeX proofs:

### 1. Corollary 3 & Proposition 4 Convexity (Why Pooling Lands at a Corner)
- **Proposition 4** provides the closed-form pooling stationary point:
  $$a^{SE} = -\frac{C_0\bar\gamma + \Delta\bar R_0}{2\Delta\bar\gamma}, \quad \text{provided } \Delta\bar\gamma > 0 \text{ (SOC)}$$
- In the unbiased baseline ($\mu_A = 1, \lambda_A = c_Q$):
  $$\Delta = c_Q - \Lambda, \quad \bar\gamma = (\Lambda - c_Q)\mathbb{E}[1/\kappa] \implies \Delta\bar\gamma = -(\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] \le 0$$
- Because the quadratic term in $\Pi(a)$ is $-a^2 \Delta \bar\gamma = +a^2 (\Lambda - c_Q)^2 \mathbb{E}[1/\kappa] > 0$, **the leader payoff $\Pi(a)$ is strictly convex in $a$**!
- Any strictly convex function on a closed interval $[0, 1]$ achieves its maximum at a **boundary** ($a = 0$ or $a = 1$), never in the interior $(0, 1)$.
- Therefore, **Corollary 3 fails under the paper's linear-risk quadratic formulation**. An interior pooling ask rate requires non-linear risk, congestion costs, or capacity constraints.

### 2. Proposition 6 Proof Sketch Audit (Active Constraint Set in Screening)
- Proposition 6's proof sketch assumed a standard relaxed problem where $IC_L$ and $IR_H$ bind, while $IC_H$ and $IR_L$ are slack.
- The numerical solver solved the **unrelaxed 4-variable problem** with all 4 constraints explicit:
  - **Observed Active Constraints:** `[IC_L]` strictly binding; `[IR_H]`, `[IC_H]`, and `[IR_L]` are **slack**.
  - **Economic Explanation:** In classical mechanism design (e.g., Baron--Myerson), the principal pays a monetary transfer $t$ and therefore pushes $IR_H$ to bind to minimize informational rents. Here, **there is no monetary transfer $t$**, and the leader's payoff $\Pi_\kappa = \mu_A U - (\lambda_A - c_Q)a(k-m)$ values user utility directly ($+\mu_A U$). The leader has no incentive to depress user utility down to reservation utility $\underline{U}$. The correct relaxed program is constrained by $IC_L$ alone.

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
