# Zero-Bias Instrument-Restriction Gap Under the Exact Conjunctive Payoff

**Verdict:** **PASS**

- Draws: 1500 two-type populations (seed 11); sampling box in the module docstring.
- Positive gap (> 1e-6): **61** draws (4.1%). Largest gap 5.22 on V = 100.
- Short tasks (k <= 12): 0 of 517 draws show a gap; the smallest k with a gap is 13.
- Direction: in 61 of 61 gap draws the terse type (kappa_H) guesses and the diligent type asks; in 0 the reverse.
- Mechanism: among gap draws, the terse type's question bill c_Q (k - m) under full asking is at least 0.498 V (median 1.00 V). By the envelope theorem the gain from asking, G(kappa) = max_m U(m,1) - max_m U(m,0), satisfies G'(kappa) = (m_0^2 - m_1^2)/2, so it falls with kappa wherever a type specifies more under asking than under guessing (m_1 > m_0); this holds for the terse type in 61 of 61 gap draws.

Artifacts: `exact_firstbest_gap.csv`, `exact_firstbest_gap.md`.
