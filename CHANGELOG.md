# Changelog

## Unreleased: Revision after preprint rejection (2026-10-19)

New results take the next free numbers (Theorem 1, Lemma 5, Propositions 9–10, Corollaries 7–8, Remark 13), so no existing result is renumbered. Sections 8 (continuum) and 9 (competition) are new; Predictions, Empirical Strategy and Discussion are now Sections 10–12.

### New results
- **Theorem 1 (minimum-specification standard).** With a continuum of users, writing the value of clarification as $x = a s(k-m)$ makes the leader's problem quasilinear screening with a capped in-kind transfer that the leader values. Under moderate bias the optimum is a standard: ask about every open attribute once $m \ge \bar m^* = c_Q^{eff}/\mathbb{E}[\kappa \mid \kappa \ge \kappa_b]$. Diligent users keep their first best, terse users pool at $\bar m^*$, and nobody is left to guesses. The standard is optimal if and only if a tail condition holds (for uniform types, iff $\bar m^* \le \Lambda/\bar\kappa$). Proof in the new Appendix B: Lagrangian bound with a constructed measure, ironing, and a perturbation argument for necessity.
- **Lemma 5** (reformulation), **Proposition 9** (no effort dictation without pooling: the two-type effort dictation of Corollary 6 does not survive with a continuum), **Corollary 7** (the welfare cost of bias is third order, $\frac43 f(\bar\kappa)\tilde\rho^3/c_Q$, against second order with two types), **Corollary 8** (the bunching point identifies $\tilde\rho$), **Remark 13** (large bias: a partial-clarification tier with an inverse-hazard formula).
- **Proposition 10 (competition and disclosure).** Under disclosure, Bertrand competition yields the ask rate that maximizes user surplus net of the real cost of questions; under opacity clarification is a shrouded attribute and competition drives it to zero.
- **Predictions P9–P10** and a bunching-based test in the experimental design.

### Presentation
- Abstract rewritten as one paragraph around the main results; the introduction now states four contributions.
- Script paths removed from the prose and collected in a replication appendix (Appendix D).
- The unclipped-versus-clipped sensitivity analysis (Table 1) moved to Appendix C.
- Retitled "Strategic Under-Specification: Optimal Clarification Standards for AI Assistants".
- The AI-use disclosure now states that AI assistance was used for simulation code and for deriving and checking some parts of the analysis.

### Verification
- New package `underspec_sim/model3_continuum/` (standard, tail condition, optimal N-type mechanism) and check 23, `verify_continuum_standard.py`; new `tests/test_continuum.py` (80 tests in total). All 23 checks pass, with the same 2 CAUTIONs.
  - Where the tail condition holds, the optimal 40- and 80-type mechanisms give every type full clarification and exceed the continuum standard by an O(1/N) term (extrapolated gap ≤ 4e-4).
  - Where it fails (uniform types on [0.22, 0.60], r = 0.6), the optimum beats the standard by 0.044 and the tersest users get a = 0.43; their prompts follow the inverse-hazard formula of Remark 13 to within 0.03.
- `run_all_math_checks.py` and `tests/conftest.py` pin BLAS to one thread: the solvers use small dense matrices, and multithreading made the new check take 16 minutes instead of 45 seconds.

## Unreleased: Third review round (2026-10-18)

New results get the next free number (Lemma 4, Proposition 8), so no existing result is renumbered.

### New results
- **Lemma 4 (continuous implementation):** the screening menu does not need the discontinuous $a = 0$ punishment of Lemma 3. The schedule $\alpha(m) = \min\{1, \bar a_L(m), \bar a_H(m)\}$, built from each type's indifference curve through its bundle, is continuous, implements the menu, and is the most generous schedule that does. It cannot be monotone: a minimum-effort threshold (ask at rate $a_H$ for every $m \ge m_H$) fails in the saturation regimes where $\text{IC}_L$ binds (regimes 2 and 3 of Proposition 7) and in Corollary 6, because the diligent type would shorten its prompt and keep the clarification. The reviewer's suggestion that a threshold suffices in Corollary 6 was checked and is incorrect there.
- **Proposition 8 (heterogeneous attribute stakes):** with ordered stakes $\Lambda(x)$ and an attribute-blind ask rate, the first best switches from guessing to asking at a unique $\hat\kappa$: diligent users guess and terse users ask, the opposite of the exact conjunctive model. The zero-bias pooling gap is positive but small (0.08 on $V = 100$ in the benchmark); asking targeted by stake restores degeneracy.
- **Guessing-trap intuition (after Proposition 5):** a fixed wedge becomes a growing share of a shrinking net gain $s(g)$, against which users leave more attributes open. The trap needs the first force to win, which Proposition 5(i) shows is not guaranteed.
- **Experimental design:** performance bonuses per hidden test (linear-risk payoff) or for passing all tests (conjunctive payoff), a per-question deduction controlling $c_Q$, and a validated real-time classifier for $m$ in Treatment C, which uses the continuous schedule.
- **P4** adds the non-monotone ask-rate signature of screening.

### Corrected errors
- **`biased_first_best(clip=True)`** still applied the unclipped threshold $(\Lambda + c_Q^{eff})/(2k)$, reporting $a_H^B = 0$ for types whose preferred bundle is $(c_Q^{eff}/\kappa, 1)$. This produced negative "downward distortions" (down to $-1$) in Figure 4. Fixed. The menu solver now also reports the payoff-irrelevant ask rate as 0 when $m = k$.
- **Figure 4:** relaid out as one line plot above two heatmaps with larger text. The third panel was mislabelled "Welfare Gap"; it is the leader-payoff gap. The report's claims that distortion rises with heterogeneity contradicted P7 and are replaced by computed statements.
- **Figure titles:** $\bar Q$ replaced by $\alpha_2$ (Figures 1–2); Proposition numbers corrected in Figure 6 and two non-paper figures.
- Three typos the reviewer reported ($y > 1/2$, $\kappa$ for $k$ in the proof of Proposition 3, a duplicated term in Step 2 of Appendix A) do not occur in the source or its history; no change.

### Verification
- New checks 21 (`verify_continuous_implementation.py`) and 22 (`verify_heterogeneous_stakes.py`), plus 4 tests (71 total). All 22 checks pass, with the same 2 CAUTIONs.

## Unreleased: Second review round (2026-10-17)

Lemma numbers shift by one from the first round below: the new Lemma 1 is self-confirming under-specification, the effective-cost lemma is now Lemma 2, and implementability is now Lemma 3.

### Corrected errors
- **Exact conjunctive model, direction inverted (Remark 5, Sections 5–7 and 10, P6):** the paper said guessing is first best for low-cost types who specify most attributes, and that the exact model creates a zero-bias pooling loss. The per-question value of asking, $V(1-g)q^{k-m-1}$, is *highest* when few attributes remain. A new scan (`verify_exact_firstbest_gap.py`, 1,500 draws) finds a zero-bias gap in 61 draws, none with $k \le 12$, and in every one the *terse* type guesses while the diligent type asks; the terse type's question bill $c_Q(k-m)$ is then at least $0.498\,V$.
- **"Divergences at k = 3" were ties:** at $k = 3$ every type specifies fully under both policies, so the exact pooling payoff is flat in $a$ and the solver's `argmax` picked $a = 0$. `verify_exact_conjunctive.py` now reports ties explicitly. Against the Taylor baseline there are 0 strict divergences. Against the additive baseline there are 10, all where $\Lambda = L(1-g) \le c_Q$ (outside the maintained assumption), so the earlier sentence placing all divergences at $g \ge 0.90$ was wrong as well.
- **Welfare ordering at $\lambda_A = 3.0$:** the pooling leader is exactly indifferent between $a = 1$ and $a = 2/3$, so the reported gap of $-0.81$ was a tie-break artifact. The text now says so.
- **P4** relied on $a_L$, which is unobservable when $m_L = k$; it is restated as a per-unresolved-attribute ask rate, which is weakly lower for terse prompts under screening.
- **P1** is now stated for linear risk, where $\partial^2U/\partial m\,\partial g = -L(1-a) \le 0$ holds globally, without Assumption 1.
- **Citations:** Ambrus and Egorov (2017) is *Delegation and nonmonetary incentives*, JET 171; CLAM (Kuhn et al.) is arXiv:2212.07769 with an ICML 2023 workshop version, not NeurIPS.
- **README:** stale counts (17 checks, 54 tests), "three regimes", Model I section numbers, a wrong claim that $\text{IR}_L$ is slack, and the shutdown effort level.

### New results and framing
- **Lemma 1 (self-confirming under-specification):** formalizes the silent-failure channel. The naive learning path is monotone and converges to a fixed point below the correct-belief effort, ordered in the detection rate. P5 now follows from it.
- **Corollary 6 (effort dictation):** when pooling asks universally, it gives users their first best. If, in addition, the leader's preferred interior menu is incentive compatible, $W_{\text{pool}} - W_{\text{screen}} = \tilde\rho^2\mathbb{E}[1/\kappa]/2$ exactly, which is second order in the bias.
- **Remark 3 (user vs social welfare):** splits $\lambda_A - c_Q$ into real provider cost and pure bias, and states that the welfare claims are about users.
- **Remark 7 (scope of non-identification):** the $(\mu_A, \lambda_A)$ result depends on $\mu_A$ scaling all of $U$; if $\mu_A$ discounts only the defect loss, the two wedges are separable.
- **Framing:**
  - Under-specification is defined as the residual $(1-a)(k-m)$, which can be positive while prompts lengthen.
  - The guessing trap is described as a small (0.35%), second-best effect with the wedge held fixed.
  - Two new Discussion paragraphs cover the commitment and renegotiation requirement of the screening menu and the zero-bias degeneracy of the analytic model.
- **Exposition:**
  - The linear-risk specification ($\Lambda$, $s$, $\kappa$) is now defined in Section 2.3, before first use.
  - All figures are referenced in the text.
  - The proof of Proposition 7 has moved to an appendix.
  - Three directly related papers were added: Vijayvargiya et al. (2025), Yang et al. (2025), and Edwards and Schuster (2026).

### Verification
- New check 20, `verify_exact_firstbest_gap.py`, plus 4 tests (67 total). All 20 checks pass, with the same 2 CAUTIONs as before.

## Review corrections and strengthenings (2026-10-16)

### Corrected errors
- **Monotonicity claim (Remark 1, Discussion):** the manuscript said $m^*(g)$ stays monotone in 100% of cases where the local regularity condition fails. The repository's own audit shows that monotonicity *breaks* in 96.7% (377/390) of those intervals. The text now reports this, so the local condition is close to necessary, and Corollary 1 separates the global (Topkis) and local (implicit function theorem) statements.
- **Proposition 7, IR step:** the proof asserted $\max_m U(m,0;\kappa) = U(k,0;\kappa)$ for both types, which is false when $\kappa_H > \Lambda/k$. It is replaced by Lemma 2 (now Lemma 3; implementability with free prompt choice), which proves that both types weakly prefer their bundles to any off-menu prompt answered with $a = 0$.
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
