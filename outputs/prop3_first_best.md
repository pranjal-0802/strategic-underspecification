# Verification Report: Proposition 3 (First-Best Threshold)

**Status:** **PASS**

- **Theoretical Threshold:** $\kappa^* = \frac{\Lambda + c_Q}{2k} = \frac{5.0 + 2.0}{20.0} = 0.3500$
- **Observed Flip:**
  - For $\kappa < \kappa^*$: $a^{FB} = 0.0$ across all 250 test points.
  - For $\kappa > \kappa^*$: $a^{FB} = 1.0$ across all 350 test points.
  - Flip occurs strictly at $\kappa^*$.
- **Optimality Verification (Grid Search Check):**
  - Evaluated against 671 alternative $(m, a)$ bundles per type.
  - Max violation observed: `0.00e+00` (numerical tolerance threshold: `1e-7`).
  - All test points verified as global argmax: `True`.

**Artifacts Generated:**
- CSV: `prop3_first_best.csv`
- Figure: `prop3_first_best.png`
