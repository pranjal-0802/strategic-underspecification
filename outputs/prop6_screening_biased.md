# Verification Report: Proposition 6 (Screening Distortion Under Bias)

**Status:** **PASS** (Downward distortion verified; active constraint structure characterized)

### 1. Representative Distortion Verification ($\lambda_A = 4.0, \mu_A = 1.0$):
- **Low-Cost Type L:** $a_L^{SB} = 0.0000$ vs biased first-best $a_L^B = 0.0000$ (Remains at corner $a=0$: **True**).
- **High-Cost Type H:** $a_H^{SB} = 0.7500$ strictly below biased first-best $a_H^B = 1.0000$ (Distortion size = **0.2500**).
- **Active Constraints:** `[IC_L]` (matches corrected proof sketch where $IC_L$ binds alone).

### 2. Extended Active-Constraint Audit Across Bias Magnitudes (Task 4):
Across a 2D sweep of $\lambda_A \in [c_Q, 10 c_Q] = [2.0, 20.0]$ and $\mu_A \in [0.1, 1.0]$ (100 configurations):
- **Does $IR_H$ ever bind?** **NO** ($IR_H$ binds in 0/100 configurations; minimum slack = `75.00`).
  *Rationale:* Because the assistant values user welfare ($\mu_A > 0$) and there are no monetary transfers to extract rent, the leader has no incentive to depress user utility to $\underline{U}$.
- **Does $IC_H$ ever bind?** **YES** ($IC_H$ becomes active in 86/100 configurations under extreme bias $\lambda_A \ge 6.0$ or $\mu_A \le 0.5$).
  *Rationale:* Under extreme friction or very low altruism, $\Pi_{\kappa_H}$ and $U(\cdot;\kappa_H)$ diverge substantially, causing both incentive compatibility constraints to bind simultaneously (pooling or boundary saturation at $m=k$).

### 3. Conclusion on Corrected Paper:
The paper's updated proof sketch and Remark 2 accurately reflect these findings:
1. In representative under-asking regions, $IC_L$ binds alone while $IR_H$ remains slack.
2. Under extreme bias, $IC_H$ can also bind, confirming the paper's caveat that the active set is parameter-region specific.
3. In all regions with under-asking bias, $a_H^{SB} < a_H^B$ holds strictly.

**Artifacts Generated:**
- CSV: `prop6_screening_biased.csv`
- Figure: `prop6_screening_biased.png`
- Bias Sweep CSV: `prop6_bias_sweep_active_constraints.csv`
- Bias Sweep Figure: `prop6_bias_sweep_active_constraints.png`
- Summary: `prop6_screening_biased.md`
