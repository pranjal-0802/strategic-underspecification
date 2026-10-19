# Verification Report: Heterogeneity vs Bias Distortion & Welfare Sweep

**Status:** **PASS**

### 1. Downward distortion $a_H^B - a_H^{SB}$ across heterogeneity and bias ($\\kappa_L = 0.3$):
```
delta_kappa  0.05   0.10  0.15   0.20   0.30
bias                                        
1.0           0.0  0.037  0.06  0.031  0.000
2.0           0.0  0.000  0.31  0.250  0.167
3.0           0.0  0.000  0.00  0.000  0.000
4.0           0.0  0.000  0.00  0.000  0.000
```
- **Direction:** distortion is non-negative in every cell (downward): **True**. Cells where $m_H^{SB} = k$ are recorded as zero, because no attribute is left open and $a_H$ is payoff-irrelevant.
- **Zero cells:** under prohibitive friction (bias $\\ge 3$, i.e. $c_Q^{eff} \\ge \\Lambda = 5$) the leader never asks, and at narrow heterogeneity the high-cost type specifies fully (full-specification regime of Proposition 7).
- **Heterogeneity:** in the saturation regime (type L at $(k, 0)$), positive distortion falls as $\\Delta\\kappa$ widens, by bias level: {2.0: True}. This matches Proposition 7 and P7 (distortion intensifies as heterogeneity narrows). 3 positive cells are in the boundary-relaxation regime (type L receives clarification, item 5 of Proposition 7), where no monotonicity in $\\Delta\\kappa$ is claimed.

### 2. Leader-Payoff Gap (Screening vs Corrected Pooling) & Regime Flags:
- **Dominance Holds:** **True** (Minimum gap: `0.0000`).
- **Pooling Regimes in Sweep:** `{'corner': 18, 'interior': 2}`.
- Pooling ask rates in the sweep ($k=10, L=10, c_Q=2$): [0.0, 0.333, 0.667, 1.0].
- The leader-payoff advantage of screening ($\Pi_{II} - \Pi_I$) is positive in 8 of 20 cells, all with bias $\le 2$; under prohibitive friction both regimes shut down asking and tie.

**Artifacts Generated:**
- CSV: `distortion_sweep_2d.csv`
- Figure: `distortion_sweep_2d.png`
- Summary: `distortion_sweep_2d.md`
