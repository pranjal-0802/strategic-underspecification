# Changelog

## v1.0.0 — Submitted Paper Version

Canonical release accompanying the manuscript *"Strategic Under-Specification: A Stackelberg Game Between User and Assistant"* (Agarwal, 2026).

### Summary of Framework & Contributions

- **Canonical Manuscript & Preprint:** Complete LaTeX source in `paper/strategic_underspecification.tex`, BibTeX database in `paper/references.bib`, and high-resolution figures in `paper/figures/`.
- **First-Best Social Optimum (Section 4):** Closed-form bang-bang threshold $\kappa^* = (\Lambda + c_Q)/(2k)$ separating high-cost users (always ask) from low-cost users (never ask).
- **Model I: Pooling Equilibrium (Section 5):**
  - Exact quadratic leader payoff $\Pi(a) = \text{const} + \bar L a + \bar Q a^2$ derived directly from primitives including specification cost $\frac{\kappa}{2}(m^*)^2$.
  - Proposition 4 closed-form pooling ask rate $a^{SE} = -\frac{\bar L}{2\bar Q}$ under strict concavity ($\bar Q < 0$).
  - Corollary 2 comparative statics showing $\partial a^{SE}/\partial\lambda_A > 0$ strictly on the interior branch.
  - Non-identification and ray-invariance in the excess friction ratio $\tilde\rho = (\lambda_A - c_Q)/\mu_A$ across both interior and corner regimes.
  - Corollary 3 unbiased corner solution: $\bar Q \ge 0$ weakly convex, with corner choice determined by $\Pi(1) - \Pi(0) = (\Lambda - c_Q)[k - k^*]$.
- **Model II: Screening Mechanism (Section 6):**
  - Proposition 5: Private information is free under an unbiased leader ($\Pi_\kappa \equiv U$), fully recovering First-Best with slack IC constraints.
  - Proposition 6: Downward distortion of high-cost type under under-asking bias ($\lambda_A > \mu_A c_Q$), with active constraint set characterized across bias magnitudes.
- **Robustness & Computational Verification Suite:**
  - Complete master test runner `run_all_math_checks.py` executing 14 mathematical checks with zero hardcoded passes (100% verified).
  - Robustness to exact conjunctive success probabilities $q(a,g)^{k-m}$ across parameter grids.
  - Monotonicity survival verified across 2,552 parameter intervals.
- **Test Suite:**
  - 48 automated pytest unit, regression, and symbolic tests (100% passing).
- **Licensing & Metadata:**
  - Open-source MIT license for simulation software; CC-BY-4.0 for manuscript; `CITATION.cff` for scholarly citation integration.
