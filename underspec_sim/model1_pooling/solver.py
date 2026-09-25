r"""
underspec_sim.model1_pooling.solver: Proposition 4 Stackelberg Pooling Equilibrium.
Computes:
(a) Closed-form solution from Proposition 4:
    a^{SE} = - (C_0 * \bar{\gamma} + \Delta * \bar{R}_0) / (2 * \Delta * \bar{\gamma})
(b) Fine grid-search of \Pi(a) over a \in [0, 1].
(c) Sanity cross-check verifying if grid-search argmax matches the closed form.
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
    a_SE_closed: float
    a_SE_grid: float
    a_SE_grid_unbounded: float
    matches: bool
    is_concave: bool
    soc_value: float  # Delta * gamma_bar
    C0: float
    Delta: float
    R0_bar: float
    gamma_bar: float
    discrepancy: float
    note: str

    def __float__(self) -> float:
        return self.a_SE_closed


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
    Solves for the leader's optimal pooling ask rate a^{SE}.
    
    Implements:
    (a) Closed-form formula from Proposition 4:
        a^{SE} = - (C_0 * \bar{\gamma} + \Delta * \bar{R}_0) / (2 * \Delta * \bar{\gamma})
    (b) Grid-search of \Pi(a) over a \in [0, 1] at fine resolution.
    (c) Verifies if the closed form matches the grid-search argmax.
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

    # (a) Closed form formula
    if abs(soc_val) < 1e-12:
        # Denominator is zero (linear payoff in a)
        a_SE_closed = float("nan")
    else:
        a_SE_closed = - (C0 * gamma_bar + Delta * R0_bar) / (2.0 * soc_val)

    # (b) Grid search on [0, 1]
    a_grid = np.linspace(0.0, 1.0, grid_points)
    pi_grid = leader_payoff_pooling(a_grid, F_samples, mu, lam, cq, params)
    best_idx = int(np.argmax(pi_grid))
    a_SE_grid = float(a_grid[best_idx])

    # Unbounded grid search over [-5, 5]
    a_grid_unb = np.linspace(-5.0, 5.0, grid_points * 2)
    pi_grid_unb = leader_payoff_pooling(a_grid_unb, F_samples, mu, lam, cq, params)
    best_unb_idx = int(np.argmax(pi_grid_unb))
    a_SE_grid_unb = float(a_grid_unb[best_unb_idx])

    # Check match
    if np.isnan(a_SE_closed):
        matches = False
        discrepancy = float("inf")
        note = "Linear payoff in a: denominator 2*Delta*gamma_bar is zero."
    elif not is_concave:
        matches = False
        discrepancy = abs(a_SE_closed - a_SE_grid)
        note = (
            f"SOC failed (Delta*gamma_bar = {soc_val:.4f} <= 0). "
            f"Pi(a) is strictly CONVEX, so critical point {a_SE_closed:.4f} is a MINIMUM, "
            f"and grid search maximizes at boundary {a_SE_grid:.4f}."
        )
    elif a_SE_closed < 0.0 or a_SE_closed > 1.0:
        matches = False
        discrepancy = abs(a_SE_closed - a_SE_grid)
        note = (
            f"Unconstrained peak {a_SE_closed:.4f} is outside [0, 1]. "
            f"Constrained argmax on [0, 1] is at boundary {a_SE_grid:.4f}."
        )
    else:
        discrepancy = abs(a_SE_closed - a_SE_grid)
        matches = discrepancy <= tolerance
        note = (
            f"Closed form {a_SE_closed:.4f} matches grid search {a_SE_grid:.4f} "
            f"(discrepancy {discrepancy:.6f})."
        )

    return PoolingEquilibriumResult(
        a_SE_closed=a_SE_closed,
        a_SE_grid=a_SE_grid,
        a_SE_grid_unbounded=a_SE_grid_unb,
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
