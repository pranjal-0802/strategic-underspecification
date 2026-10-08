# Changelog

## v1.0.0: Submitted Paper Version

Canonical release accompanying the manuscript *"Strategic Under-Specification: A Stackelberg Game Between User and Assistant"* (Agarwal, 2026).

### Summary of Framework & Contributions

- **Canonical Manuscript & Preprint:** Complete LaTeX source in `paper/strategic_underspecification.tex`, compiled PDF in `paper/strategic_underspecification.pdf`, and figures in `paper/figures/`.
- **First-Best Social Optimum (Section 3.2):**
  - Enforces physical feasibility $m \in [0, k]$ throughout.
  - Characterizes asking dominance ($a^{FB} = 1$) across the feasible domain under linear risk ($\Lambda > c_Q$), and clarifies that the unclipped threshold $\kappa^*$ reflects solutions where $m > k$.
  - Formalizes genuine interior asking-versus-guessing trade-offs under the exact conjunctive model and heterogeneous attribute stakes.
- **Strategic Under-Specification (Section 2.3):**
  - Formal definition of strategic under-specification.
  - Proposition 2 proves equilibrium under-specification under naive beliefs ($m^*_{\text{naive}} < m^*_{\text{soph}}$) and characterises the exact quadratic welfare loss.
  - Documents exact-model counterexample where question-answering costs induce upstream over-specification.
- **Model I: Pooling Equilibrium (Section 4):**
  - Exact quadratic leader payoff $\Pi(a) = \text{const} + \bar L a + \bar Q a^2$ derived directly from primitives.
  - Proposition 4 closed-form pooling ask rate $a^{SE} = -\frac{\bar L}{2\bar Q}$ under strict concavity ($\bar Q < 0$).
  - Corollary 2 proves $\partial a^{SE}/\partial\lambda_A \le 0$ on the physically valid domain $\bar R_0 \ge 0$: perceived clarification friction suppresses the equilibrium asking rate.
  - Corollary 3 non-identification and ray-invariance in the normalized friction ratio $\tilde\rho = (\lambda_A - c_Q)/\mu_A$.
  - Corollary 4 ray-invariance extension to the corner regime.
  - Corollary 5 unbiased corner solution: $\bar Q \ge 0$ weakly convex, selecting $a^{SE} = 1$ under physical feasibility.
- **Model II: Screening Mechanism (Section 5):**
  - Proposition 5: Full efficiency survives private cost information under an unbiased leader ($\Pi_\kappa \equiv U$), recovering First-Best with slack IC constraints.
  - Proposition 6: Downward distortion of high-cost type under under-asking bias ($a_H^{SB} < 1$), proven globally via IC boundary geometry under explicit complexity threshold $k > k_{\text{crit}}$ with off-menu deterrence schedule.
  - Active constraint characterization: $\text{IC}_L$ binds alone under moderate bias; $\text{IC}_H$ becomes co-active under extreme bias.
- **Robustness & Computational Verification Suite:**
  - Master test runner `run_all_math_checks.py` executing 14 mathematical checks with zero hardcoded passes (100% verified).
  - Robustness to exact conjunctive success probabilities $q(a,g)^{k-m}$ across parameter grids.
  - Monotonicity survival verified across 2,552 parameter intervals.
- **Test Suite:**
  - 49 automated pytest unit, regression, and symbolic tests (100% passing).
- **Literature & Scholarly Framing:**
  - Integrated foundational citations: Holmström (1984), Aghion and Tirole (1997), Kamenica and Gentzkow (2011).
  - Clean academic exposition with keywords, code availability, and AI-use disclosures.
  - Zero em dashes throughout the manuscript and codebase.
- **Licensing & Metadata:**
  - Dual open-source license: MIT for simulation code; CC-BY-4.0 for manuscript; `CITATION.cff` for citation metadata.
