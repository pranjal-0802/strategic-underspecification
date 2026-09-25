# Verification Report: Proposition 6 (Screening Distortion Under Bias)

**Status:** **PASS** (Core distortion properties confirmed; active constraint discrepancy audited)

### 1. Distortion Verification:
- **Low-Cost Type L:** $a_L^{SB} = 0.0000$ vs biased first-best $a_L^B = 0.0000$ (Remains at corner $a=0$: **True**).
- **High-Cost Type H:** $a_H^{SB} = 0.7500$ strictly below biased first-best $a_H^B = 1.0000$ (Distortion size = **0.2500**).
- **Core Proposition 6 Conclusion:** Passed! The high-cost type is asked strictly less than the leader's own preference.

### 2. Active Constraint Set Audit (Checking Paper's Proof Sketch):
- **Observed Active Constraints:** `[IC_L]`
- **Paper's Proof Sketch Assumed:** `[IC_L, IR_H]`
- **Analysis:**
  - $IC_L$ is **strictly binding** (slack = `-9.95e-14`).
  - $IC_H$ is **slack** (slack = `3.0556`).
  - $IR_L$ is **slack** (slack = `85.00`).
  - $IR_H$ is **slack** (slack = `78.06`).
- **Key Insight on Proof Sketch:** In standard mechanism design with monetary transfers, $IR_H$ is made binding to extract all surplus. Here, because there is no cash transfer $t$ and $\mu_A > 0$, the leader directly values user utility ($+\mu_A U$), so the leader has no incentive to push $U_H$ down to $\underline{U}$. Therefore, $IR_H$ is slack! The correct relaxed program is constrained by $IC_L$ alone.

**Artifacts Generated:**
- CSV: `prop6_screening_biased.csv`
- Figure: `prop6_screening_biased.png`
