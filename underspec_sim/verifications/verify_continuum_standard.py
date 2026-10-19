r"""
underspec_sim.verifications.verify_continuum_standard:
Screening a continuum of users (Section 8): the minimum-specification standard.

The optimal mechanism for N equally spaced types (exact discrete incentive constraints) is
compared with Theorem 1:
  (a) Tail condition holds: the N-type optimum gives every type full clarification, and its value
      exceeds the continuum standard's only by an O(1/N) term (the discrete-type dictation rent of
      Corollary 6). The Richardson extrapolation 2 gap(2N) - gap(N) is ~0.
  (b) Tail condition fails clearly: on the same discrete types the optimum beats the best standard by
      a margin that does not vanish with N, and the tersest users receive partial clarification.
      On that tier, prompt lengths follow Lambda / (kappa + (1-r)/r * (1-F)/f) (Remark 13).
  (c) mbar = c_Q^eff / E[kappa | kappa >= kappa_b] (Theorem 1(i)) and the wedge is recovered from the
      bunching point (Corollary 8).
  (d) Proposition 9: discrete types allow O(1/N) effort dictation near the pool boundary; the distance
      between the N-type allocation and the standard's allocation shrinks as N grows.
Parameters: k = 10, L = 10, g = 0.5 (Lambda = 5, s = 3), c_Q = 2, V = 100.

Outputs:
- outputs/continuum_standard.csv
- outputs/continuum_standard.md
- outputs/continuum_standard.png
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from underspec_sim.model3_continuum import (
    TypeDistribution, standard_threshold, standard_on_types, tail_condition, solve_discrete_menu, recover_wedge,
)

V, K, LAM, C_Q = 100.0, 10.0, 5.0, 2.0
S = LAM - C_Q

UNIF = TypeDistribution(0.30, 0.60, lambda t: np.ones_like(t))
WIDE = TypeDistribution(0.22, 0.60, lambda t: np.ones_like(t))
INCR = TypeDistribution(0.22, 0.80, lambda t: 0.2 + (t - 0.22))

HOLD_CASES = [("uniform[0.30,0.60]", UNIF, 0.2), ("uniform[0.30,0.60]", UNIF, 0.4),
              ("increasing[0.22,0.80]", INCR, 0.2), ("increasing[0.22,0.80]", INCR, 0.4)]
FAIL_CASES = [("uniform[0.22,0.60]", WIDE, 0.6)]
FIGURE_R = [0.2, 0.4, 0.6]


def _solve(dist, n, r, st_cont=None):
    kaps, p = dist.discrete(n)
    st = standard_on_types(kaps, p, V, LAM, C_Q, K, r * S)
    start = np.concatenate([st["m"], S * (K - st["m"])])
    sol = solve_discrete_menu(kaps, p, V, LAM, C_Q, K, r, start=start)
    return kaps, p, st, sol


def run_verification(output_dir: str = "outputs", n_values=(40, 80)) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if output_dir == "outputs":
        output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    n_mid, n_hi = n_values

    rows = []
    for name, dist, r in HOLD_CASES + FAIL_CASES:
        st_c = standard_threshold(dist, V, LAM, C_Q, K, r * S)
        tc = tail_condition(dist, st_c["mbar"], LAM, C_Q, r)
        gaps_cont, gaps_same, devs, last = [], [], [], None
        for n in (n_mid, n_hi):
            kaps, p, st_d, sol = _solve(dist, n, r)
            gaps_cont.append(sol["value"] - st_c["value"])
            gaps_same.append(sol["value"] - st_d["value"])
            # distance from the continuum standard's allocation (which satisfies Proposition 9)
            devs.append(float(np.abs(sol["m"] - np.maximum(st_c["mbar"], np.minimum(K, C_Q / kaps))).max()))
            last = (kaps, p, sol)
        kaps, p, sol = last
        row = {"distribution": name, "r": r, "mbar": st_c["mbar"], "mbar_foc": st_c["foc_mbar"],
               "kappa_b": st_c["kappa_b"], "tail_condition": tc, "tail_holds": tc >= -1e-6,
               f"gap_N{n_mid}": gaps_cont[0], f"gap_N{n_hi}": gaps_cont[1],
               "gap_extrapolated": 2 * gaps_cont[1] - gaps_cont[0],
               f"gap_same_types_N{n_mid}": gaps_same[0], f"gap_same_types_N{n_hi}": gaps_same[1],
               "min_a": float(sol["a"].min()), f"max_dev_m_N{n_mid}": devs[0], f"max_dev_m_N{n_hi}": devs[1],
               "rho_recovered": recover_wedge(st_c["mbar"], st_c["cond_mean"], C_Q), "rho": r * S}
        if not row["tail_holds"]:
            tier = sol["a"] < 1 - 1e-3
            F = np.cumsum(p) - p / 2
            dens = p / (kaps[1] - kaps[0])
            formula = LAM / (kaps + (1 - r) / r * (1 - F) / dens)
            row["tail_tier_share"] = float(p[tier].sum())
            row["tail_formula_maxdev"] = float(np.abs(sol["m"][tier] - formula[tier]).max()) if tier.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "continuum_standard.csv")
    df.to_csv(csv_path, index=False)

    hold = df[df["tail_holds"]]
    fail = df[~df["tail_holds"]]
    conv = bool((hold["gap_extrapolated"].abs() < 2e-3).all() and (hold[f"gap_N{n_hi}"] <= hold[f"gap_N{n_mid}"] + 1e-6).all())
    full_ask = bool((hold["min_a"] > 1 - 1e-3).all())
    beats = bool((fail[f"gap_same_types_N{n_hi}"] > 1e-2).all() and (fail["min_a"] < 0.9).all())
    formula_ok = bool((fail["tail_formula_maxdev"] < 0.25).all())
    foc = float((df["mbar"] - df["mbar_foc"]).abs().max())
    rec = float((df["rho"] - df["rho_recovered"]).abs().max())
    # Proposition 9: discrete types allow O(1/N) effort dictation; it must vanish as N grows
    nodict = bool((hold[f"max_dev_m_N{n_hi}"] <= 0.75 * hold[f"max_dev_m_N{n_mid}"] + 1e-3).all())
    passed = bool(conv and full_ask and beats and formula_ok and foc < 1e-3 and rec < 1e-3 and nodict)
    verdict = "PASS" if passed else "FAIL"

    # Figure: uniform[0.30, 0.60] population, N = n_hi
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in FIGURE_R:
        kaps, p, st_d, sol = _solve(UNIF, n_hi, r)
        line, = axes[0].plot(kaps, sol["m"], lw=2, label=f"$r = {r}$")
        axes[0].plot(kaps, st_d["m"], ls="--", color=line.get_color(), lw=1)
        axes[1].plot(kaps, sol["a"], lw=2, color=line.get_color(), label=f"$r = {r}$")
    kaps = UNIF.discrete(n_hi)[0]
    axes[0].plot(kaps, C_Q / kaps, color="gray", ls=":", label=r"first best $c_Q/\kappa$")
    axes[0].set_xlabel(r"cost type $\kappa$"); axes[0].set_ylabel(r"specification $m(\kappa)$")
    axes[0].set_title("Prompt length: optimum (solid), standard (dashed)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r"cost type $\kappa$"); axes[1].set_ylabel(r"ask rate $a(\kappa)$")
    axes[1].set_ylim(0.85, 1.01); axes[1].set_title("Clarification")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "continuum_standard.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    cols = ["distribution", "r", "mbar", "kappa_b", "tail_condition", f"gap_N{n_mid}", f"gap_N{n_hi}",
            "gap_extrapolated", f"gap_same_types_N{n_hi}", "min_a"]
    table = df[cols].round(4).to_string(index=False)
    md = f"""# Continuum Screening: The Minimum-Specification Standard (Section 8)

**Verdict:** **{verdict}**

```
{table}
```
- Tail condition holds ({len(hold)} cases): every type receives full clarification in the N-type optimum ({full_ask}); its value exceeds the continuum standard by an O(1/N) term, and the extrapolation 2 gap(N={n_hi}) - gap(N={n_mid}) is below 2e-3 ({conv}).
- Tail condition fails ({len(fail)} case): on the same discrete types the optimum beats the best standard by more than 0.01 and the tersest users get a < 0.9 ({beats}). On the partial-clarification tier, prompt lengths are within {float(fail['tail_formula_maxdev'].max()):.3f} of Lambda / (kappa + (1-r)/r (1-F)/f).
- Theorem 1(i): largest |mbar - c_Q^eff / E[kappa | pooled]| = {foc:.1e}. Corollary 8: largest wedge-recovery error {rec:.1e}.
- Proposition 9: with finitely many types the leader can still dictate some effort near the pool boundary; in the tail-condition cases the largest distance from the standard's allocation shrinks from N = {n_mid} to N = {n_hi} ({nodict}; values {[round(v, 3) for v in hold[f'max_dev_m_N{n_mid}']]} -> {[round(v, 3) for v in hold[f'max_dev_m_N{n_hi}']]}).

Artifacts: `continuum_standard.csv`, `continuum_standard.md`, `continuum_standard.png`.
"""
    md_path = os.path.join(output_dir, "continuum_standard.md")
    with open(md_path, "w") as f:
        f.write(md)

    return {"verdict": verdict, "passed": passed, "df": df, "n_hold": int(len(hold)), "n_fail": int(len(fail)),
            "converges": conv, "full_ask": full_ask, "beats": beats, "formula_ok": formula_ok,
            "foc_err": foc, "rho_err": rec, "no_dictation": nodict,
            "csv_path": csv_path, "md_path": md_path, "png_path": png_path}


if __name__ == "__main__":
    res = run_verification()
    print(f"Continuum standard: {res['verdict']} ({res['n_hold']} cases with the tail condition, {res['n_fail']} without)")
