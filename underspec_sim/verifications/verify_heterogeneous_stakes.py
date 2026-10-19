r"""
underspec_sim.verifications.verify_heterogeneous_stakes:
Heterogeneous attribute stakes under an attribute-blind ask rate (Proposition 8).

Attributes x in [0, k] are ordered by decreasing stake
    Lambda(x) = Lambda_k + (Lambda_0 - Lambda_k) (1 - x/k)^p,
users specify the m most important ones, and the assistant asks at one rate a about every
open attribute:
    U(m, a; kappa) = V - int_m^k [(1 - a) Lambda(x) + a c_Q] dx - kappa m^2 / 2.
With G(kappa) = max_m U(m, 1) - max_m U(m, 0), the proposition states that when
Lambda(k) < c_Q < mean(Lambda), G is zero up to Lambda(k)/k, negative up to a unique
kappa_hat > c_Q / x*, and positive beyond (diligent users guess, terse users ask). The script
checks the sign pattern and the envelope identity G' = (m_0^2 - m_1^2)/2 on random stake
profiles, and reports the benchmark example used in the paper (k = 10, c_Q = 2, Lambda(x) = 9 - 0.8x).

Outputs:
- outputs/heterogeneous_stakes.csv
- outputs/heterogeneous_stakes.md
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

V, K, C_Q = 100.0, 10.0, 2.0
M_GRID = np.linspace(0.0, K, 8001)


class Profile:
    def __init__(self, lam0: float, lamk: float, p: float):
        self.lam0, self.lamk, self.p = lam0, lamk, p

    def stake(self, x):
        return self.lamk + (self.lam0 - self.lamk) * (1.0 - x / K) ** self.p

    def tail(self, m):
        """int_m^k Lambda(x) dx."""
        return self.lamk * (K - m) + (self.lam0 - self.lamk) * K / (self.p + 1.0) * (1.0 - m / K) ** (self.p + 1.0)

    @property
    def mean(self):
        return self.tail(0.0) / K

    @property
    def x_star(self):
        return K * (1.0 - ((C_Q - self.lamk) / (self.lam0 - self.lamk)) ** (1.0 / self.p))

    def best(self, kappa, a):
        u = V - ((1.0 - a) * self.tail(M_GRID) + a * C_Q * (K - M_GRID)) - 0.5 * kappa * M_GRID ** 2
        i = int(np.argmax(u))
        return float(u[i]), float(M_GRID[i])

    def best_targeted(self, kappa):
        c = np.minimum(self.stake(M_GRID), C_Q)
        tail = np.concatenate([np.cumsum(((c[:-1] + c[1:]) / 2.0 * np.diff(M_GRID))[::-1])[::-1], [0.0]])
        return float((V - tail - 0.5 * kappa * M_GRID ** 2).max())

    def gain(self, kappa):
        return self.best(kappa, 1.0)[0] - self.best(kappa, 0.0)[0]

    def kappa_hat(self):
        lo, hi = C_Q / self.x_star, 50.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.gain(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)


def sign_pattern_ok(prof: Profile) -> Dict[str, Any]:
    kap = np.linspace(0.02, 8.0, 400)
    G = np.array([prof.gain(x) for x in kap])
    lo = prof.lamk / K
    tol = 1e-6
    zero_below = bool(np.all(np.abs(G[kap <= lo]) < 1e-3))
    region = G[(kap > lo * 1.05)]
    signs = np.sign(region[np.abs(region) > tol])
    one_change = bool(len(signs) > 0 and signs[0] < 0 and np.sum(np.diff(signs) != 0) == 1)
    kh = prof.kappa_hat()
    # envelope identity at interior points
    errs = []
    for x in np.linspace(lo + 0.05, 4.0, 25):
        h = 1e-3
        fd = (prof.gain(x + h) - prof.gain(x - h)) / (2 * h)
        m0, m1 = prof.best(x, 0.0)[1], prof.best(x, 1.0)[1]
        errs.append(abs(fd - (m0 ** 2 - m1 ** 2) / 2.0))
    return {"zero_below": zero_below, "one_sign_change": one_change, "kappa_hat": kh,
            "kappa_hat_above_bound": bool(kh > C_Q / prof.x_star),
            "envelope_err": float(np.max(errs))}


def run_verification(output_dir: str = "outputs", n_profiles: int = 40, seed: int = 13) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if output_dir == "outputs":
        output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n_profiles:
        prof = Profile(float(rng.uniform(3.0, 15.0)), float(rng.uniform(0.0, 1.9)), float(rng.uniform(0.5, 3.0)))
        if not (prof.lamk < C_Q < prof.mean):
            continue
        rows.append({"Lambda_0": prof.lam0, "Lambda_k": prof.lamk, "p": prof.p, "mean_Lambda": prof.mean,
                     "x_star": prof.x_star, **sign_pattern_ok(prof)})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "heterogeneous_stakes.csv")
    df.to_csv(csv_path, index=False)

    # Benchmark example: Lambda(x) = 9 - 0.8 x (p = 1, Lambda_0 = 9, Lambda_k = 1)
    bench = Profile(9.0, 1.0, 1.0)
    kh = bench.kappa_hat()
    kL, kH = 0.3, 0.5
    uL = [bench.best(kL, a)[0] for a in (0.0, 1.0)]
    uH = [bench.best(kH, a)[0] for a in (0.0, 1.0)]
    first_best = 0.5 * max(uL) + 0.5 * max(uH)
    pooling = max(0.5 * bench.best(kL, a)[0] + 0.5 * bench.best(kH, a)[0] for a in np.linspace(0, 1, 101))
    targeted = 0.5 * bench.best_targeted(kL) + 0.5 * bench.best_targeted(kH)
    gap, target_gain = first_best - pooling, targeted - first_best
    sorting_ok = bool(uL[0] > uL[1] and uH[1] > uH[0])

    n_ok = int((df["zero_below"] & df["one_sign_change"] & df["kappa_hat_above_bound"]).sum())
    env_max = float(df["envelope_err"].max())
    passed = bool(n_ok == len(df) and env_max < 0.05 and sorting_ok and gap > 0 and target_gain > 0)
    verdict = "PASS" if passed else "FAIL"

    md = f"""# Heterogeneous Attribute Stakes (Proposition 8)

**Verdict:** **{verdict}**

- Random stake profiles with Lambda(k) < c_Q < mean(Lambda): {len(df)} (seed {seed}).
- Sign pattern of G(kappa) (zero up to Lambda(k)/k, negative, then positive with one sign change at kappa_hat > c_Q/x*): **{n_ok} of {len(df)}**.
- Envelope identity G'(kappa) = (m_0^2 - m_1^2)/2: largest finite-difference error {env_max:.3f}.
- Benchmark (k = 10, c_Q = 2, Lambda(x) = 9 - 0.8x, mean 5): kappa_hat = {kh:.3f}. With kappa_L = {kL}, kappa_H = {kH} and equal shares, the diligent type guesses and the terse type asks ({sorting_ok}); zero-bias pooling gap = {gap:.3f}; targeted asking (ask iff Lambda(x) > c_Q) adds {target_gain:.3f} over the uniform first best.

Artifacts: `heterogeneous_stakes.csv`, `heterogeneous_stakes.md`.
"""
    md_path = os.path.join(output_dir, "heterogeneous_stakes.md")
    with open(md_path, "w") as f:
        f.write(md)

    return {"verdict": verdict, "passed": passed, "n_profiles": int(len(df)), "n_ok": n_ok,
            "envelope_err": env_max, "kappa_hat": kh, "gap": gap, "target_gain": target_gain,
            "sorting_ok": sorting_ok, "csv_path": csv_path, "md_path": md_path}


if __name__ == "__main__":
    res = run_verification()
    print(f"Heterogeneous stakes: {res['verdict']} ({res['n_ok']}/{res['n_profiles']} profiles; "
          f"kappa_hat={res['kappa_hat']:.3f}, gap={res['gap']:.3f})")
