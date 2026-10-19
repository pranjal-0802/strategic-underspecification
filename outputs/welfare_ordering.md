# Verification Report: User Welfare Ordering Across Regimes

**Status:** **PASS**

### Key Results Across Friction Bands:
1. **Mild Bias ($\lambda_A \in [2.5, 3.0]$):**
   - At $\lambda_A = 2.5$: $W_{pool} = 85.33$, $W_{screen} = 85.00$ (Gap: $-0.33$). Pooling delivers higher user welfare because it preserves universal clarification ($a=1.0$) with autonomous follower effort choices ($m_L=6.67, m_H=4.0$), whereas screening exploits contractible specification to dictate higher effort ($m_L=8.33, m_H=5.00$), reducing user welfare.
   - At $\lambda_A = 3.0$: $W_{pool} = 85.33$, $W_{screen} = 84.52$ (Gap: $-0.81$), where screening additionally distorts asking downward ($a_H = 0.97$). This point is a knife-edge: the pooling leader is exactly indifferent between $a = 1$ and $a = 2/3$, so $W_{pool}$ is $85.33$ or $82.00$ depending on the tie-break and the sign of the gap is not determined.
   - For $\lambda_A \le 2.5$ the gap equals $\tilde\rho^2 \mathbb{E}[1/\kappa]/2$ exactly (Corollary 6, effort dictation): second order in the bias.
2. **Moderate Bias ($\lambda_A \in [3.5, 4.5]$):**
   - At $\lambda_A = 3.5$: $W_{pool} = 80.00$, $W_{screen} = 82.45$ (Gap: $+2.45$). Screening delivers higher user welfare because pooling collapses to zero asking ($a=0$), while screening preserves partial clarification for high-cost types ($a_H > 0$).
   - At $\lambda_A = 4.0$: $W_{pool} = 80.00$, $W_{screen} = 81.53$ (Gap: $+1.53$).
3. **Severe / Prohibitive Friction ($\lambda_A \ge 5.0$, where $\tilde\rho \ge s = 3$):**
   - At $\lambda_A = 5.0$: $W_{pool} = 80.00$, $W_{screen} = 80.00$ (Gap: $0.00$). Both regimes shut down asking completely ($a = 0$), so user welfare ties at $80.00$.
