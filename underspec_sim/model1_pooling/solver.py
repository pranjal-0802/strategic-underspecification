r"""
underspec_sim.model1_pooling.solver: Proposition 4 Stackelberg Pooling Equilibrium.
Computes:
(a) Signs the second-order condition: \bar{Q} = (\mu_A * s - 2 * b) * \bar{\gamma} / 2,
    where s \equiv \Lambda - c_Q and b \equiv \lambda_A - c_Q.
(b) If \bar{Q} < 0 (concave, so stationary point is a local maximum):
    uses closed-form interior stationary point a^{SE} = - \bar{L} / (2 * \bar{Q}),
    where \bar{L} = (\mu_A * s - b) * \bar{R}_0.
    Clipped to [0, 1] if unconstrained peak lies outside the interval.
(c) If \bar{Q} >= 0 (convex/linear, including unbiased \mu_A=1, \lambda_A=c_Q):
    evaluates \Pi(0) and \Pi(1) directly, returning whichever is larger:
    \Pi(1) - \Pi(0) = \bar{L} + \bar{Q}
    (which equals (\Lambda - c_Q)*[\bar{R}_0 + \bar{\gamma} / 2] in the unbiased case).
    Flags regime: "corner".
"""

from dataclasses import dataclass
from typing import Sequence, Optional, Union
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.best_response import beta_func, gamma_func
from underspec_sim.model1_pooling.payoff import leader_payoff_pooling


@dataclass
class PoolingEquilibriumResult:
    """Detailed results from pooling equilibrium solve."""
    a_SE: float              # True optimal ask rate on [0, 1]
    regime: str              # "interior" or "corner"
    a_SE_closed: float       # Raw stationary point formula from Proposition 4
    a_SE_grid: float         # Numerical fine grid argmax on [0, 1]
    a_SE_grid_unbounded: float
    pi_0: float              # Leader payoff at a=0
    pi_1: float              # Leader payoff at a=1
    pi_diff: float           # Pi(1) - Pi(0)
    matches: bool
    is_concave: bool
    soc_value: float         # Q_bar (< 0 means strictly concave)
    Q_bar: float
    L_bar: float
    R0_bar: float
    gamma_bar: float
    discrepancy: float
    note: str
    b: float = 0.0           # lambda_A - c_Q
    s: float = 0.0           # Lambda - c_Q
    Delta: float = 0.0       # lambda_A - mu_A * Lambda (legacy)
    C0: float = 0.0          # mu_A * Lambda (legacy)

    def __float__(self) -> float:
        return self.a_SE


def solve_pooling_equilibrium(
    F_samples: Sequence[float],
    mu_A: Optional[float] = None,
    lambda_A: Optional[float] = None,
    c_Q: Optional[float] = None,
    params: Optional[ModelParams] = None,
    grid_points: int = 10001,
    tolerance: float = 1e-2,
    clip_m: bool = True,
) -> PoolingEquilibriumResult:
    r"""
    Solves for the leader's optimal pooling ask rate a^{SE} on [0, 1].

    Parameters:
    - clip_m: if True, enforces physical constraint m^*(\kappa; a) \in [0, k]
              when evaluating expected leader payoffs.
              if False, uses the unconstrained quadratic approximation.
    """
    if params is None:
        params = ModelParams()
    mu = params.mu_A if mu_A is None else mu_A
    lam = params.lambda_A if lambda_A is None else lambda_A
    cq = params.c_Q if c_Q is None else c_Q

    effective_params = params.model_copy(update={"mu_A": mu, "lambda_A": lam, "c_Q": cq})
    Lambda = effective_params.Lambda
    k = effective_params.k

    kappa_arr = np.asarray(F_samples, dtype=float)
    betas = beta_func(kappa_arr, effective_params)
    gammas = gamma_func(kappa_arr, effective_params)

    R0_bar = k - float(np.mean(betas))
    gamma_bar = float(np.mean(gammas))

    s = Lambda - cq
    b = lam - cq

    Q_bar = (mu * s - 2.0 * b) * gamma_bar / 2.0
    L_bar = (mu * s - b) * R0_bar
    is_concave = Q_bar < -1e-12

    from underspec_sim.model1_pooling.payoff import leader_payoff_pooling_clipped

    # Payoffs at boundaries
    if clip_m:
        pi_0 = float(leader_payoff_pooling_clipped(0.0, F_samples, mu, lam, cq, effective_params))
        pi_1 = float(leader_payoff_pooling_clipped(1.0, F_samples, mu, lam, cq, effective_params))
    else:
        pi_0 = float(leader_payoff_pooling(0.0, F_samples, mu, lam, cq, effective_params))
        pi_1 = float(leader_payoff_pooling(1.0, F_samples, mu, lam, cq, effective_params))
    pi_diff = pi_1 - pi_0

    # Grid search on [0, 1]
    a_grid = np.linspace(0.0, 1.0, grid_points)
    if clip_m:
        pi_grid = leader_payoff_pooling_clipped(a_grid, F_samples, mu, lam, cq, effective_params)
    else:
        pi_grid = leader_payoff_pooling(a_grid, F_samples, mu, lam, cq, effective_params)
    max_val = float(np.max(pi_grid))
    if np.isclose(pi_grid[-1], max_val, atol=1e-8):
        best_idx = len(a_grid) - 1
    else:
        best_idx = int(np.argmax(pi_grid))
    a_SE_grid = float(a_grid[best_idx])

    # Unbounded grid search over [-5, 5]
    a_grid_unb = np.linspace(-5.0, 5.0, grid_points * 2)
    pi_grid_unb = leader_payoff_pooling(a_grid_unb, F_samples, mu, lam, cq, effective_params)
    best_unb_idx = int(np.argmax(pi_grid_unb))
    a_SE_grid_unb = float(a_grid_unb[best_unb_idx])

    # Raw Proposition 4 closed form formula (unconstrained quadratic)
    if abs(Q_bar) < 1e-12:
        a_SE_closed = float("nan")
    else:
        a_SE_closed = - L_bar / (2.0 * Q_bar)

    # Determine optimum a_SE on [0, 1]
    if clip_m:
        a_SE = a_SE_grid
        if a_SE <= 1e-4 or a_SE >= 1.0 - 1e-4:
            regime = "corner"
            note = f"Clipped domain optimum is corner a_SE = {a_SE:.4f}."
        else:
            regime = "interior"
            note = f"Clipped domain optimum is interior a_SE = {a_SE:.4f}."
        if is_concave:
            a_target = float(np.clip(a_SE_closed, 0.0, 1.0))
        else:
            a_target = 1.0 if pi_1 >= pi_0 else 0.0
        discrepancy = abs(a_SE_grid - a_target)
        matches = discrepancy <= tolerance
    else:
        if is_concave:
            if 0.0 <= a_SE_closed <= 1.0:
                a_SE = a_SE_closed
                regime = "interior"
                note = f"Concave regime (Q_bar={Q_bar:.4f} < 0): interior optimum a_SE = {a_SE:.4f}."
            else:
                a_SE = 1.0 if a_SE_closed > 1.0 else 0.0
                regime = "corner"
                note = f"Concave regime (Q_bar={Q_bar:.4f} < 0): stationary point {a_SE_closed:.4f} outside [0, 1], clipped to boundary {a_SE:.1f}."
        else:
            regime = "corner"
            a_SE = 1.0 if pi_1 >= pi_0 else 0.0
            note = (
                f"Convex/linear regime (Q_bar={Q_bar:.4f} >= 0): "
                f"optimum is corner a_SE={a_SE:.1f}."
            )
        discrepancy = abs(a_SE - a_SE_grid)
        matches = discrepancy <= tolerance

    return PoolingEquilibriumResult(
        a_SE=a_SE,
        regime=regime,
        a_SE_closed=a_SE_closed,
        a_SE_grid=a_SE_grid,
        a_SE_grid_unbounded=a_SE_grid_unb,
        pi_0=pi_0,
        pi_1=pi_1,
        pi_diff=pi_diff,
        matches=matches,
        is_concave=is_concave,
        soc_value=Q_bar,
        Q_bar=Q_bar,
        L_bar=L_bar,
        R0_bar=R0_bar,
        gamma_bar=gamma_bar,
        discrepancy=discrepancy,
        note=note,
        b=b,
        s=s,
        Delta=float(lam - mu * Lambda),
        C0=float(mu * Lambda),
    )
