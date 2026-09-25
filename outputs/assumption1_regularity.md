# Verification Report: Assumption 1 Regularity Characterization

> **Notice:** In the canonical manuscript (`paper/strategic_underspecification.tex`), Assumption 1
> is formulated as $(k - m^*)\ln q \ge -1$ to guarantee decreasing differences ($\partial^2 U / \partial m \partial q \le 0$).
> For the comprehensive finite-difference monotonicity audit, see `verify_monotonicity_survival.py`.

### Grid Audit Results ($k \in [5, 15], q \in [0.70, 0.95], \kappa \in [0.20, 2.00]$, $V = 100$, 726 points):
- **Corrected condition** ($(k - m^*)\ln q \ge -1$): Holds in **89.9%** (653/726) of grid points.
- **Legacy draft condition** ($(k - m^*)\ln q \le -1$): Held in **10.1%** (73/726) of grid points.
- **Sensitivity:** Under lower task valuations ($V = 10$, as in the paper's numerical example), users specify fewer attributes and the corrected condition holds in roughly 34.3% (31.7% at $V = 8$).
- **Monotonicity Survival:** The follow-up audit (`verify_monotonicity_survival.py`) confirms that empirical monotonicity of $m^*(g)$ survives in **100%** of tested intervals (2,162/2,162) outside the sufficient condition.
