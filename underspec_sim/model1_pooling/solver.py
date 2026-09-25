r"""
underspec_sim.model1_pooling.solver: Proposition 4 Stackelberg Pooling Equilibrium.
Computes:
(a) Signs the second-order condition: \Delta * \bar{\gamma} = (\lambda_A - \mu_A \Lambda) * \bar{\gamma}.
(b) If \Delta * \bar{\gamma} > 0 (concave):
    uses closed-form interior stationary point a^{SE} = - (C_0*\bar{\gamma} + \Delta*\bar{R}_0)/(2*\Delta*\bar{\gamma})
    clipped to [0, 1] if unconstrained peak lies outside the interval.
(c) If \Delta * \bar{\gamma} <= 0 (convex/linear, including unbiased \mu_A=1, \lambda_A=c_Q):
    evaluates \Pi(0) and \Pi(1) directly, returning whichever is larger:
    \Pi(1) - \Pi(0) = - [C_0*\bar{\gamma} + \Delta*(\bar{R}_0 + \bar{\gamma})]
    (which equals (\Lambda - c_Q)*[\bar{R}_0 - c_Q * \mathbb{E}[1/\kappa]] in the unbiased case).
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
    soc_value: float         # Delta * gamma_bar
    C0: float
    Delta: float
    R0_bar: float
    gamma_bar: float
    discrepancy: float
    note: str

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
) -> PoolingEquilibriumResult:
    r"""
    Solves for the leader's optimal pooling ask rate a^{SE} on [0, 1].
    
    Implements:
    (a) Compute Delta * gamma_bar and check its sign.
    (b) If positive, use closed-form interior a_SE (Proposition 4).
    (c) If non-positive, evaluate Pi(0) and Pi(1) directly, returning the larger.
        Sets regime to "corner" vs "interior".
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

    C0 = mu * Lambda
    Delta = lam - mu * Lambda
    soc_val = Delta * gamma_bar
    is_concave = soc_val > 0.0

    # Payoffs at boundaries
    pi_0 = float(leader_payoff_pooling(0.0, F_samples, mu, lam, cq, effective_params))
    pi_1 = float(leader_payoff_pooling(1.0, F_samples, mu, lam, cq, effective_params))
    pi_diff = pi_1 - pi_0

    # Grid search on [0, 1]
    a_grid = np.linspace(0.0, 1.0, grid_points)
    pi_grid = leader_payoff_pooling(a_grid, F_samples, mu, lam, cq, effective_params)
    best_idx = int(np.argmax(pi_grid))
    a_SE_grid = float(a_grid[best_idx])

    # Unbounded grid search over [-5, 5]
    a_grid_unb = np.linspace(-5.0, 5.0, grid_points * 2)
    pi_grid_unb = leader_payoff_pooling(a_grid_unb, F_samples, mu, lam, cq, effective_params)
    best_unb_idx = int(np.argmax(pi_grid_unb))
    a_SE_grid_unb = float(a_grid_unb[best_unb_idx])

    # Raw Proposition 4 closed form formula
    if abs(soc_val) < 1e-12:
        a_SE_closed = float("nan")
    else:
        a_SE_closed = - (C0 * gamma_bar + Delta * R0_bar) / (2.0 * soc_val)

    # Determine true optimum a_SE on [0, 1]
    if is_concave:
        # Concave: interior peak exists if in [0, 1]
        if 0.0 <= a_SE_closed <= 1.0:
            a_SE = a_SE_closed
            regime = "interior"
            note = f"Concave regime (Delta*gamma={soc_val:.4f} > 0): interior optimum a_SE = {a_SE:.4f}."
        else:
            # Clipped to boundary
            a_SE = 1.0 if a_SE_closed > 1.0 else 0.0
            regime = "corner"
            note = f"Concave regime (Delta*gamma={soc_val:.4f} > 0): stationary point {a_SE_closed:.4f} outside [0, 1], clipped to boundary {a_SE:.1f}."
    else:
        # Non-positive: strictly convex or linear on [0, 1], optimum is at a corner
        regime = "corner"
        if pi_1 >= pi_0:
            a_SE = 1.0
        else:
            a_SE = 0.0
        note = (
            f"Convex/linear regime (Delta*gamma={soc_val:.4f} <= 0): "
            f"stationary point {a_SE_closed:.4f} is a minimum; "
            f"optimum is corner a_SE={a_SE:.1f} (Pi(1)-Pi(0)={pi_diff:.4f})."
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
        soc_value=soc_val,
        C0=C0,
        Delta=Delta,
        R0_bar=R0_bar,
        gamma_bar=gamma_bar,
        discrepancy=discrepancy,
        note=note,
    )
