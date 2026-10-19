# Continuous Implementation of Screening Menus (Lemma 4)

**Verdict:** **PASS**

- Menus audited: 212 (200 solver menus, seed 7; 12 Corollary 6 menus).
- (ii) Largest gain from any prompt under alpha(m): 1.4e-14 (no type gains).
- (i) Smallest alpha(m_theta) - a_theta: -3.7e-11 (solver tolerance); largest step of alpha on a grid of spacing 5.0e-04: 4.8e-03, and refining the grid tenfold shrinks every menu's largest step by a factor of at least 9.9 (continuous; a jump would not shrink).
- (iv) Menus with binding IC_L and type L at (k, 0): 28. Largest |alpha - [Lambda - kappa_L(k+m)/2]/s| above m_H: 8.2e-12.
- (v) A minimum-effort threshold a(m) = a_H 1{m >= m_H} lets type L gain in 28 of 28 binding menus and in 12 of 12 Corollary 6 menus (largest gain 2.503).

Artifacts: `continuous_implementation.csv`, `continuous_implementation.md`.
