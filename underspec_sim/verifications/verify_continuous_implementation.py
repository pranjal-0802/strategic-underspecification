r"""
underspec_sim.verifications.verify_continuous_implementation:
Continuous implementation of screening menus (Lemma 4).

For a menu {(m_L, a_L), (m_H, a_H)} with bundle utilities U_theta, the ask rate that leaves
type theta indifferent between prompt m < k and its bundle is

    abar_theta(m) = [U_theta - V + Lambda (k - m) + kappa_theta m^2 / 2] / [s (k - m)],

and the candidate schedule is alpha(m) = min{1, abar_L(m), abar_H(m)}. The script checks, on
random menus from the numerical solver and on the effort-dictation menus of Corollary 6:
  (i)   alpha is continuous and alpha(m_theta) = a_theta;
  (ii)  under a(m) = alpha(m) no type gains from any prompt;
  (iv)  when IC_L binds with type L at (k, 0), alpha = abar_L = [Lambda - kappa_L (k + m)/2]/s above m_H;
  (v)   a minimum-effort threshold a(m) = a_H 1{m >= m_H} lets type L gain in those regimes and in Cor 6.
Parameters: k = 10, L = 10, g = 0.5 (Lambda = 5), c_Q = 2, V = 100.

Outputs:
- outputs/continuous_implementation.csv
- outputs/continuous_implementation.md
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from underspec_sim.core.params import ModelParams
from underspec_sim.model2_screening.solver import solve_menu

V, K, L_LOSS, G_ACC, C_Q = 100.0, 10.0, 10.0, 0.5, 2.0
LAM = L_LOSS * (1.0 - G_ACC)
S = LAM - C_Q
M_GRID = np.linspace(0.0, K, 20001)[:-1]


def utility(m, a, kappa):
    return V - LAM * (1.0 - a) * (K - m) - 0.5 * kappa * m ** 2 - a * C_Q * (K - m)


def abar(m, u_theta, kappa):
    return (u_theta - V + LAM * (K - m) + 0.5 * kappa * m ** 2) / (S * (K - m))


def schedule(m, menu, kL, kH):
    (mL, aL), (mH, aH) = menu
    uL, uH = utility(mL, aL, kL), utility(mH, aH, kH)
    return np.clip(np.minimum(abar(m, uL, kL), abar(m, uH, kH)), 0.0, 1.0)


def audit(menu, kL, kH):
    (mL, aL), (mH, aH) = menu
    uL, uH = utility(mL, aL, kL), utility(mH, aH, kH)
    alpha = schedule(M_GRID, menu, kL, kH)
    gain_L = float((utility(M_GRID, alpha, kL) - uL).max())
    gain_H = float((utility(M_GRID, alpha, kH) - uH).max())
    attain = min(
        float(schedule(np.array([mH]), menu, kL, kH)[0] - aH) if mH < K - 1e-9 else 0.0,
        float(schedule(np.array([mL]), menu, kL, kH)[0] - aL) if mL < K - 1e-9 else 0.0,
    )
    # Continuity: the largest step must shrink in proportion to the grid spacing (a jump would not).
    max_jump = float(np.abs(np.diff(alpha)).max())
    fine = np.linspace(0.0, K, 200001)[:-1]
    jump_ratio = float(np.abs(np.diff(schedule(fine, menu, kL, kH))).max()) / max(max_jump, 1e-15)
    above = M_GRID >= mH
    abar_L_formula = (LAM - kL * (K + M_GRID[above]) / 2.0) / S
    iv_err = float(np.abs(alpha[above] - abar_L_formula).max()) if above.any() else 0.0
    thr = np.where(M_GRID >= mH, aH, 0.0)
    if mL < K - 1e-9:
        thr = np.where(M_GRID >= mL, max(aL, aH), thr)
    thr_gain_L = float((utility(M_GRID, thr, kL) - uL).max())
    return dict(gain_L=gain_L, gain_H=gain_H, attain=attain, max_jump=max_jump, jump_ratio=jump_ratio,
                iv_err=iv_err, thr_gain_L=thr_gain_L)


def run_verification(output_dir: str = "outputs", n_draws: int = 200, seed: int = 7) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if output_dir == "outputs":
        output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    params = ModelParams(k=K, g=G_ACC, L=L_LOSS, c_Q=C_Q, V=V)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        kL = float(rng.uniform(0.10, 0.45))
        kH = kL * float(rng.uniform(1.1, 4.0))
        lam = float(rng.uniform(2.05, 4.9))
        fL = float(rng.uniform(0.2, 0.8))
        r = solve_menu(kL, kH, fL, 1.0 - fL, mu_A=1.0, lambda_A=lam, c_Q=C_Q, params=params)
        menu = ((r.m_L, r.a_L), (r.m_H, r.a_H))
        binding = bool(abs(r.IC_L_slack) < 1e-5 and r.m_L >= K - 1e-6 and r.m_H < K - 1e-6)
        rows.append({"source": "solver", "kappa_L": kL, "kappa_H": kH, "lambda_A": lam, "f_L": fL,
                     "m_L": r.m_L, "a_L": r.a_L, "m_H": r.m_H, "a_H": r.a_H,
                     "binding_L_at_k": binding, **audit(menu, kL, kH)})
    # Corollary 6 (effort dictation): bundles (c_eff / kappa, 1) for both types
    for kL, kH in [(0.3, 0.5), (0.3, 0.6), (0.35, 0.8)]:
        rho_max = C_Q * (kH - kL) / (kH + kL)
        for rho in np.linspace(0.1, 1.0, 4) * rho_max:
            ce = C_Q + rho
            if ce / K > kL:
                continue
            menu = ((ce / kL, 1.0), (ce / kH, 1.0))
            rows.append({"source": "cor6", "kappa_L": kL, "kappa_H": kH, "lambda_A": ce, "f_L": 0.5,
                         "m_L": ce / kL, "a_L": 1.0, "m_H": ce / kH, "a_H": 1.0,
                         "binding_L_at_k": False, **audit(menu, kL, kH)})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "continuous_implementation.csv")
    df.to_csv(csv_path, index=False)

    n = len(df)
    max_gain = float(df[["gain_L", "gain_H"]].max().max())
    min_attain = float(df["attain"].min())
    max_jump = float(df["max_jump"].max())
    max_ratio = float(df.loc[df["max_jump"] > 1e-12, "jump_ratio"].max())
    b = df[df["binding_L_at_k"]]
    c6 = df[df["source"] == "cor6"]
    iv_max = float(b["iv_err"].max()) if len(b) else 0.0
    thr_fail_b = int((b["thr_gain_L"] > 1e-6).sum())
    thr_fail_c6 = int((c6["thr_gain_L"] > 1e-6).sum())
    passed = bool(max_gain <= 1e-9 and min_attain >= -1e-3 and max_ratio <= 0.2
                  and iv_max <= 1e-3 and thr_fail_b == len(b) and thr_fail_c6 == len(c6) and len(b) > 0)
    verdict = "PASS" if passed else "FAIL"

    md = f"""# Continuous Implementation of Screening Menus (Lemma 4)

**Verdict:** **{verdict}**

- Menus audited: {n} ({n - len(c6)} solver menus, seed {seed}; {len(c6)} Corollary 6 menus).
- (ii) Largest gain from any prompt under alpha(m): {max_gain:.1e} (no type gains).
- (i) Smallest alpha(m_theta) - a_theta: {min_attain:.1e} (solver tolerance); largest step of alpha on a grid of spacing {K/20000:.1e}: {max_jump:.1e}, and refining the grid tenfold shrinks every menu's largest step by a factor of at least {1/max_ratio:.1f} (continuous; a jump would not shrink).
- (iv) Menus with binding IC_L and type L at (k, 0): {len(b)}. Largest |alpha - [Lambda - kappa_L(k+m)/2]/s| above m_H: {iv_max:.1e}.
- (v) A minimum-effort threshold a(m) = a_H 1{{m >= m_H}} lets type L gain in {thr_fail_b} of {len(b)} binding menus and in {thr_fail_c6} of {len(c6)} Corollary 6 menus (largest gain {float(pd.concat([b, c6])['thr_gain_L'].max()):.3f}).

Artifacts: `continuous_implementation.csv`, `continuous_implementation.md`.
"""
    md_path = os.path.join(output_dir, "continuous_implementation.md")
    with open(md_path, "w") as f:
        f.write(md)

    return {"verdict": verdict, "passed": passed, "n": n, "n_binding": int(len(b)), "n_cor6": int(len(c6)),
            "max_gain": max_gain, "min_attain": min_attain, "max_jump": max_jump, "max_ratio": max_ratio, "iv_max": iv_max,
            "thr_fail_binding": thr_fail_b, "thr_fail_cor6": thr_fail_c6,
            "csv_path": csv_path, "md_path": md_path}


if __name__ == "__main__":
    res = run_verification()
    print(f"Continuous implementation: {res['verdict']} ({res['n']} menus, {res['n_binding']} binding)")
