# Verification Report: Assumption 1 Regularity Characterization

### Conceptual Distinction:
1. **Global Topkis Submodularity ($k\ln q \ge -1$):**
   Requires decreasing differences $\partial^2 U / \partial m \partial q \le 0$ globally on the entire strategy space $m \in [0, k]$.
   Because $(k - m)\ln q$ is most negative at $m = 0$, global submodularity reduces to $k\ln q \ge -1$.
   On the primitive grid ($k \in [5, 15], q \in [0.70, 0.95]$), this stringent condition holds in **24.8%** (180/726) of points.
2. **Local Decreasing Differences at Endogenous Optimum ($(k - m^*)\ln q \ge -1$):**
   Under optimizing behavior, users articulate the bulk of attributes upstream, keeping residual attributes $k - m^*$ small.
   At the endogenous optimum $m^*$, the local condition $(k - m^*)\ln q \ge -1$ holds in **89.9%** (653/726) of grid points.
3. **Monotonicity Survival:**
   The follow-up finite-difference audit (`verify_monotonicity_survival.py`) confirms that empirical monotonicity of $m^*(g)$ holds in **100%** of tested intervals (2,162/2,162) where the local condition holds.
   Even where global submodularity fails, boundary constraints and local strict concavity prevent non-monotonic reversals across the vast majority of practical settings.

**Artifacts Generated:**
- CSV: `assumption1_regularity.csv`
- Figure: `assumption1_regularity.png`
