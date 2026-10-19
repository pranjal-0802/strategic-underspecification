# Heterogeneous Attribute Stakes (Proposition 8)

**Verdict:** **PASS**

- Random stake profiles with Lambda(k) < c_Q < mean(Lambda): 40 (seed 13).
- Sign pattern of G(kappa) (zero up to Lambda(k)/k, negative, then positive with one sign change at kappa_hat > c_Q/x*): **40 of 40**.
- Envelope identity G'(kappa) = (m_0^2 - m_1^2)/2: largest finite-difference error 0.009.
- Benchmark (k = 10, c_Q = 2, Lambda(x) = 9 - 0.8x, mean 5): kappa_hat = 0.313. With kappa_L = 0.3, kappa_H = 0.5 and equal shares, the diligent type guesses and the terse type asks (True); zero-bias pooling gap = 0.076; targeted asking (ask iff Lambda(x) > c_Q) adds 0.549 over the uniform first best.

Artifacts: `heterogeneous_stakes.csv`, `heterogeneous_stakes.md`.
