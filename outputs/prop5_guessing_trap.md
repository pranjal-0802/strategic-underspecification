# Verification Report: Proposition 5 (The Guessing Trap)

**Status:** **PASS**

### Key Results:
1. **Admissible Domain ($g \in [0.50, 0.60]$):**
   - At $g = 0.50$: $a^{SE} = 0.661$, $\mathbb{E}[m] = 7.728$, $W = 82.134$.
   - At $g = 0.60$: $a^{SE} = 0.424$, $\mathbb{E}[m] = 7.953$, $W = 81.843$.
   - **Equilibrium Capability Reversal:** Better guessing causes $a^{SE}$ to fall ($0.661 \to 0.424$), lengthening prompts ($7.728 \to 7.953$) and lowering user welfare ($82.134 \to 81.843$).

2. **Exact Necessary and Sufficient Derivative Condition:**
   - Differentiating the unclipped closed form yields:
     $$\frac{da^{SE}}{dg} < 0 \iff \mu_A \bar R_0 > (\mu_A s - b)\mathbb{E}[1/\kappa](1 - 2a^{SE})$$
   - **Guaranteed Domain:** Holds strictly for all $a^{SE} \ge 1/2$ (since $1 - 2a^{SE} \le 0$).
   - **Counterexample ($a^{SE} < 1/2$):** For $k=12, L=17, c_Q=4.5, \lambda_A=11.5, \kappa \sim U[1.4, 1.5]$, $a^{SE}$ increases from $0.211$ ($g=0.05$) to $0.230$ ($g=0.10$).

3. **Equivalence of Prompt Lengthening and Welfare Decline:**
   - $\frac{d\mathbb{E}[m^*]}{dg} > 0 \iff \frac{dW}{dg} < 0 \iff -\frac{da^{SE}}{dg} > \frac{L(1 - a^{SE})}{\Lambda(g) - c_Q}$.
