r"""
underspec_sim.verifications.verify_exact_firstbest_gap:
Zero-bias instrument-restriction gap under the exact conjunctive payoff.

With an unbiased leader (mu_A = 1, lambda_A = c_Q) screening attains every type's
first best (each type's global maximizer is incentive compatible, Prop. 6), while
pooling attains max_a E_kappa[max_m U(m, a; kappa)]. The zero-bias
instrument-restriction gap is therefore

    Gap = E_kappa[max_{m,a} U] - max_a E_kappa[max_m U(m, a; kappa)] >= 0,

with U(m, a; kappa) = V q(a,g)^{k-m} - kappa m^2 / 2 - a c_Q (k - m).

The script draws two-type populations and records, for every draw with a
positive gap, which type asks and which guesses at its first best. The paper
uses it to check two claims:
  (i)  a gap requires long tasks (none for k <= 12 in the sampled box);
  (ii) when it arises, the terse (high-kappa) type guesses and the diligent type
       asks, and the terse type's question bill c_Q (k - m) is a large share of V. The marginal value of asking, (k-m)[V(1-g)q^{k-m-1} - c_Q], is
       largest when few attributes remain, so guessing is first best only for a
       type facing a large question bill c_Q (k - m).

Sampling box: V = 100, k in {3,...,30}, g ~ U[0.5, 0.98], c_Q ~ U[0.5, 6],
kappa_L ~ U[0.05, 1], kappa_H = kappa_L * U[1.2, 6], f_L = f_H = 1/2.

Outputs:
- outputs/exact_firstbest_gap.csv
- outputs/exact_firstbest_gap.md
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

A_GRID = np.linspace(0.0, 1.0, 101)


def value_curve(kappa: float, g: float, k: float, c_Q: float, V: float, nm: int = 1201) -> np.ndarray:
    """max_m U(m, a; kappa) on the a-grid (vectorized over a and m)."""
    m = np.linspace(0.0, k, nm)[None, :]
    a = A_GRID[:, None]
    q = a + (1.0 - a) * g
    u = V * q ** (k - m) - 0.5 * kappa * m ** 2 - a * c_Q * (k - m)
    return u.max(axis=1)


def unassisted_effort(kappa: float, g: float, k: float, V: float, nm: int = 1201) -> float:
    m = np.linspace(0.0, k, nm)
    u = V * g ** (k - m) - 0.5 * kappa * m ** 2
    return float(m[int(np.argmax(u))])


def run_verification(output_dir: str = "outputs", n_draws: int = 1500, seed: int = 11) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if output_dir == "outputs":
        output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    V = 100.0
    rows = []
    for _ in range(n_draws):
        k = float(rng.integers(3, 31))
        g = float(rng.uniform(0.5, 0.98))
        c_Q = float(rng.uniform(0.5, 6.0))
        kL = float(rng.uniform(0.05, 1.0))
        kH = kL * float(rng.uniform(1.2, 6.0))
        uL, uH = value_curve(kL, g, k, c_Q, V), value_curve(kH, g, k, c_Q, V)
        gap = 0.5 * uL.max() + 0.5 * uH.max() - (0.5 * uL + 0.5 * uH).max()
        m1_H = min(k, c_Q / kH)
        rows.append({
            "k": k, "g": g, "c_Q": c_Q, "kappa_L": kL, "kappa_H": kH,
            "gap": float(gap),
            "a_FB_L": float(A_GRID[int(np.argmax(uL))]),
            "a_FB_H": float(A_GRID[int(np.argmax(uH))]),
            "bill_H_over_V": c_Q * (k - m1_H) / V,
            "m0_H": unassisted_effort(kH, g, k, V),
            "m1_H": m1_H,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "exact_firstbest_gap.csv")
    df.to_csv(csv_path, index=False)

    pos = df[df["gap"] > 1e-6]
    n_pos = int(len(pos))
    n_short = int((df["k"] <= 12).sum())
    n_pos_short = int((pos["k"] <= 12).sum())
    terse_guesses = int(((pos["a_FB_H"] < 0.5) & (pos["a_FB_L"] > 0.5)).sum())
    diligent_guesses = int(((pos["a_FB_L"] < 0.5) & (pos["a_FB_H"] > 0.5)).sum())
    min_k = float(pos["k"].min()) if n_pos else float("nan")
    min_bill = float(pos["bill_H_over_V"].min()) if n_pos else float("nan")
    med_bill = float(pos["bill_H_over_V"].median()) if n_pos else float("nan")
    max_gap = float(pos["gap"].max()) if n_pos else 0.0
    preempt = int((pos["m1_H"] > pos["m0_H"]).sum())

    passed = bool(n_pos > 0 and diligent_guesses == 0 and terse_guesses == n_pos)
    verdict = "PASS" if passed else "FAIL"

    md = f"""# Zero-Bias Instrument-Restriction Gap Under the Exact Conjunctive Payoff

**Verdict:** **{verdict}**

- Draws: {n_draws} two-type populations (seed {seed}); sampling box in the module docstring.
- Positive gap (> 1e-6): **{n_pos}** draws ({100 * n_pos / n_draws:.1f}%). Largest gap {max_gap:.2f} on V = 100.
- Short tasks (k <= 12): {n_pos_short} of {n_short} draws show a gap; the smallest k with a gap is {min_k:.0f}.
- Direction: in {terse_guesses} of {n_pos} gap draws the terse type (kappa_H) guesses and the diligent type asks; in {diligent_guesses} the reverse.
- Mechanism: among gap draws, the terse type's question bill c_Q (k - m) under full asking is at least {min_bill:.3f} V (median {med_bill:.2f} V). By the envelope theorem the gain from asking, G(kappa) = max_m U(m,1) - max_m U(m,0), satisfies G'(kappa) = (m_0^2 - m_1^2)/2, so it falls with kappa wherever a type specifies more under asking than under guessing (m_1 > m_0); this holds for the terse type in {preempt} of {n_pos} gap draws.

Artifacts: `exact_firstbest_gap.csv`, `exact_firstbest_gap.md`.
"""
    md_path = os.path.join(output_dir, "exact_firstbest_gap.md")
    with open(md_path, "w") as f:
        f.write(md)

    return {
        "verdict": verdict, "passed": passed, "n_draws": n_draws, "n_pos": n_pos,
        "n_pos_short": n_pos_short, "n_short": n_short, "min_k": min_k,
        "terse_guesses": terse_guesses, "diligent_guesses": diligent_guesses,
        "min_bill": min_bill, "med_bill": med_bill, "max_gap": max_gap, "preempt": preempt,
        "csv_path": csv_path, "md_path": md_path,
    }


if __name__ == "__main__":
    res = run_verification()
    print(f"Exact first-best gap: {res['verdict']} ({res['n_pos']}/{res['n_draws']} draws with a gap)")
