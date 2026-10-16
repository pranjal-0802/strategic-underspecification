# Changelog

## Unreleased: Review corrections and strengthenings (2026-10-16)

### Corrected errors
- **Monotonicity claim (Remark 1, Discussion):** the manuscript said $m^*(g)$ stays monotone in 100% of cases where the local regularity condition fails. The repository's own audit shows that monotonicity *breaks* in 96.7% (377/390) of those intervals. The text now reports this, so the local condition is close to necessary, and Corollary 1 separates the global (Topkis) and local (implicit function theorem) statements.
- **Proposition 7, IR step:** the proof asserted $\max_m U(m,0;\kappa) = U(k,0;\kappa)$ for both types, which is false when $\kappa_H > \Lambda/k$. It is replaced by Lemma 2 (implementability with free prompt choice), which proves that both types weakly prefer their bundles to any off-menu prompt answered with $a = 0$.
- **Proposition 7, regime 4:** "collapse to $(k,0)$" fails when $\kappa_L > \bar\kappa_L$ (beaten by the exact optimum in 19/40 draws). The saturation condition now governs regimes 3 and 4. The effort-distortion regime gets its own, correctly derived saturation condition, $f_H(c_Q^{eff} - \kappa_H \hat m) \le f_L(c_Q^{eff} - \kappa_L k)$.
- **Welfare-ordering remark and P6:** wrong thresholds ($k_{\text{slack}} = 10.33$, $k_{\text{crit}} = 8.33$, $a_H = 0.833$ at $\lambda_A = 3$) are replaced by the exact optimum $(8.67, 1), (5.56, 0.97)$. The pooling breakpoints are exactly $\lambda_A = 3.0$ ($a$: $1 \to 2/3$) and $3.5$ ($\to 0$), not "$\approx 3.33$".
- **Prohibitive-friction shutdown:** types are left at $\min\{k, \Lambda/\kappa\}$, not always at $m = k$.
- **Discussion contradiction:** it said that participation does not bind at the bottom of the type distribution, while Section 6 said that it does. Both now state the reversed rent pattern consistently.
- **Power analysis:** $n = 130$ per arm (not 134); a Bonferroni-corrected requirement and a cell-level caveat were added.
- **Cost externality:** the effort gap is $(L_{\text{soc}} - L)(1-g)(1-a)/\kappa$; it previously omitted $(1-a)$ and named the wrong primitive.
- **Unbacked statistic:** the "7% / 35%" guessing-trap incidence was not produced by any script. It is replaced by a reproducible sweep (12.2% / 51.6%, `verify_guessing_trap_incidence.py`).

### New results
- Unbiased pooling is a corner solution for **any** type distribution **with clipping** (convexity of $\max_m U(m,a)$ in $a$).
- Corollary 2 now holds for any type distribution, clipped or not (Topkis, decreasing differences in $(a, \lambda_A)$); the numerical claim it replaces is no longer needed.
- Asking distortion bound $0 < 1 - a_H^{SB} < \tilde\rho/s$.
- Reversed rent pattern: without transfers, the efficient specifier is held to its outside option and the distorted type keeps a rent.
- A $4\times2$ experimental design with a policy-disclosure factor, which identifies the belief channel (Prop. 2).

### Verification
- `verify_prop7_regimes.py`: an exact grid solver, independent of SLSQP. Closed forms match in 201/201 saturated draws; the effort-regime condition is sufficient in 82/82; Lemma 2 holds in 270/270.
- `verify_guessing_trap_incidence.py`, plus `tests/test_review_strengthenings.py` (7 new tests; 63 total).

## v1.0.0: Submitted Paper Version

Canonical release accompanying the manuscript *"Strategic Under-Specification: A Stackelberg Game Between User and Assistant"* (Agarwal, 2026).

### Summary of Framework & Contributions

- **Canonical Manuscript & Preprint:** Complete LaTeX source in `paper/strategic_underspecification.tex`, compiled PDF in `paper/strategic_underspecification.pdf`, and figures in `paper/figures/`.
- **First-Best Social Optimum (Section 4.1):**
  - Enforces physical feasibility $m \in [0, k]$ throughout.
  - Characterizes asking dominance ($a^{FB} = 1$) across the feasible domain under linear risk ($\Lambda > c_Q$), clarifying that unclipped threshold $\kappa^*$ requires $m > k$.
  - Formalizes genuine interior asking-versus-guessing trade-offs under the exact conjunctive model and heterogeneous attribute stakes.
- **Strategic Under-Specification (Section 3):**
  - Delineates rational effort substitution ($m^*(a) \le m^*(0)$) from belief-driven under-specification under opaque policies.
  - Proposition 2 proves equilibrium under-specification under naive beliefs ($m^*_{\text{naive}} < m^*_{\text{soph}}$) and characterizes the exact quadratic welfare loss.
  - Clarifies that under transparent Stackelberg policies, under-asking bias induces equilibrium over-specification ($m^*(a) > m^{FB}$).
- **Model I: Pooling Equilibrium (Section 5):**
  - Exact quadratic leader payoff $\Pi(a) = \text{const} + \alpha_1 a + \alpha_2 a^2$ derived directly from primitives.
  - Lemma 1 establishes effective question cost $c_Q^{eff} = c_Q + \tilde\rho$ and unconditional ray-invariance in $\tilde\rho = (\lambda_A - c_Q)/\mu_A$.
  - Proposition 4 closed-form pooling ask rate $a^{SE} = -\frac{\alpha_1}{2\alpha_2}$ under strict concavity ($\alpha_2 < 0$).
  - Corollary 2 proves $\partial a^{SE}/\partial\lambda_A \le 0$ on the interior specification domain ($\kappa \ge \Lambda/k$, $\bar R_0 > 0$): perceived clarification friction suppresses the equilibrium asking rate.
  - Corollary 3 characterizes the continuous transition of $a^{SE}$ across friction regimes on $[0, 1]$.
  - Corollary 4 unbiased corner solution: $\alpha_2 \ge 0$ weakly convex, selecting $a^{SE} = 1$ under physical feasibility, achieving First-Best with zero welfare loss at zero bias under linear risk.
  - Proposition 5 proves the equilibrium capability reversal (The Guessing Trap): higher model guessing capability $g$ perversely lengthens prompts ($d\mathbb{E}[m^*]/dg > 0$) and lowers user welfare ($dW/dg < 0$). Characterizes exact necessary and sufficient derivative condition $\mu_A \bar R_0 > (\mu_A s - b)\mathbb{E}[1/\kappa](1 - 2a^{SE})$, proves prompt lengthening and welfare reduction are mathematically simultaneous, and calibrates on the admissible binary domain $g \in [0.50, 0.60]$.
- **Model II: Screening Mechanism (Section 6):**
  - Proposition 6: Full efficiency survives private cost information under an unbiased leader ($\Pi_\kappa \equiv U$), recovering First-Best with slack IC constraints; menu is degenerate along asking margin ($a=1$).
  - Proposition 7: Downward distortion of high-cost type under asking friction bias ($a_H^{SB} < 1$). Characterizes the explicit saturation threshold $\bar\kappa_L \equiv \frac{c_Q^{eff} + t c_Q}{k(1 + t)}$ with $t \equiv \frac{f_H}{f_L}\frac{s - \tilde\rho}{s}$ for arbitrary population shares $f_L, f_H$, and proves that boundary relaxation ($\bar\kappa_L < \kappa_L \le c_Q^{eff}/k$) tempers downward distortion by granting clarification to low-cost types ($a_L = 1, m_L < k$).
  - Participation constraints characterized under endogenous silence outside option $\underline{U}(\kappa) \equiv \max_m U(m, 0; \kappa) = V - \frac{\kappa}{2}k^2$, under which $\text{IR}_L$ binds at the bottom ($U_L = \underline{U}(\kappa_L)$).
  - Clarifies that at $m_L = k$, $a_L$ is multiplied by $k - m_L = 0$ and is payoff-irrelevant.
  - Active constraint characterization: Five operational regimes delineated (unconstrained slack at $k \le k_{\text{slack}}$, pure effort distortion at $k_{\text{slack}} < k \le k_{\text{crit}}$, adverse-selection asking distortion at $k > k_{\text{crit}}$, boundary saturation pooling at $k < m_H^*$, and uniform clarification shutdown at $c_Q^{eff} \ge \Lambda$).
- **Regime Comparison (Section 7):**
  - Corollary 5 proves menus weakly dominate pooling in leader payoff ($\Pi_{II} \ge \Pi_I$).
  - Characterizes non-monotonic user welfare ranking: pooling yields higher user welfare under mild bias ($W = 85.33$ vs $85.00$ at $\lambda_A = 2.5$; $W = 85.33$ vs $84.52$ at $\lambda_A = 3.0$), while screening yields higher welfare under moderate bias ($W = 82.45$ vs $80.00$ at $\lambda_A = 3.5$; $W = 81.53$ vs $80.00$ at $\lambda_A = 4.0$), with both regimes collapsing to complete clarification shutdown at severe friction ($W = 80.00$ at $\lambda_A \ge 5.0$).
  - Table 1 generated directly via `underspec_sim/verifications/generate_table1.py` (recorded in `outputs/table1_unbiased_sensitivity.csv`), confirming asking strictly dominates guessing across all 11 configurations.
- **Robustness & Computational Verification Suite:**
  - Master test runner `run_all_math_checks.py` executing 17 mathematical checks with zero hardcoded passes (100% verified in ~75s).
  - Robustness to exact conjunctive success probabilities $q(a,g)^{k-m}$ across parameter grids.
  - Monotonicity survival verified across 2,552 parameter intervals.
- **Test Suite:**
  - 54 automated pytest unit, regression, and symbolic tests (100% passing).
- **Literature & Scholarly Framing:**
  - Integrated foundational citations: Holmström (1984), Aghion and Tirole (1997), Kamenica and Gentzkow (2011), Hadfield-Menell & Hadfield (2019), Dessein (2002), Laban et al. (2025), Mirrlees (1971), Myerson (1979), and Ben-Porath et al. (2014).
  - Clean academic exposition with keywords, code availability, and AI-use disclosures.
  - Zero em dashes throughout the manuscript and codebase.
- **Licensing & Metadata:**
  - Dual open-source license: MIT for simulation code; CC-BY-4.0 for manuscript; `CITATION.cff` for citation metadata.
