# Verification Report: Proposition 3 (First-Best on Feasible Domain)

**Status:** **PASS**

- **Theoretical Domain Feasibility:** For all feasible $m \in [0, k]$ and $\Lambda > c_Q$, $\frac{\partial U}{\partial a} = (\Lambda - c_Q)(k - m) \ge 0$.
- **Result:** Asking ($a^{FB} = 1$) weakly dominates guessing ($a=0$) across all 601 types on $[0, k]$.
- **Independent Grid Search Verification:** Evaluated against 1111 alternative $(m, a)$ pairs on $[0, k] \times [0, 1]$.
  - Max violation observed: `0.00e+00` (numerical tolerance threshold: `1e-7`).
  - All test points verified as global argmax: `True`.

**Artifacts Generated:**
- CSV: `prop3_first_best.csv`
- Figure: `prop3_first_best.png`
