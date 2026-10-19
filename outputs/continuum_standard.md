# Continuum Screening: The Minimum-Specification Standard (Section 8)

**Verdict:** **PASS**

```
         distribution   r   mbar  kappa_b  tail_condition  gap_N40  gap_N80  gap_extrapolated  gap_same_types_N80  min_a
   uniform[0.30,0.60] 0.2 5.3333   0.3750          0.0000   0.0100   0.0050            0.0000              0.0050 1.0000
   uniform[0.30,0.60] 0.4 7.1111   0.3000          0.0000  -0.0000  -0.0000           -0.0000             -0.0000 1.0000
increasing[0.22,0.80] 0.2 3.8953   0.5134          0.0000   0.0238   0.0121            0.0004              0.0121 0.9996
increasing[0.22,0.80] 0.4 5.1778   0.3863          0.0000   0.0322   0.0162            0.0001              0.0161 0.9995
   uniform[0.22,0.60] 0.6 9.2683   0.2200         -0.0804   0.0481   0.0438            0.0395              0.0438 0.4283
```
- Tail condition holds (4 cases): every type receives full clarification in the N-type optimum (True); its value exceeds the continuum standard by an O(1/N) term, and the extrapolation 2 gap(N=80) - gap(N=40) is below 2e-3 (True).
- Tail condition fails (1 case): on the same discrete types the optimum beats the best standard by more than 0.01 and the tersest users get a < 0.9 (True). On the partial-clarification tier, prompt lengths are within 0.029 of Lambda / (kappa + (1-r)/r (1-F)/f).
- Theorem 1(i): largest |mbar - c_Q^eff / E[kappa | pooled]| = 4.3e-05. Corollary 8: largest wedge-recovery error 2.7e-05.
- Proposition 9: with finitely many types the leader can still dictate some effort near the pool boundary; in the tail-condition cases the largest distance from the standard's allocation shrinks from N = 40 to N = 80 (True; values [0.148, 0.0, 0.426, 0.461] -> [0.079, 0.0, 0.248, 0.261]).

Artifacts: `continuum_standard.csv`, `continuum_standard.md`, `continuum_standard.png`.
