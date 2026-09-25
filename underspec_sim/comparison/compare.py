r"""
underspec_sim.comparison.compare: Section 7 Regime Comparison.
Compares Model I (Pooling) and Model II (Screening):
- Verifies Corollary 4: Menus weakly dominate pooling (\Pi_{II} >= \Pi_I).
- Decomposes welfare gaps into heterogeneity-driven and bias-driven components.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Union
import numpy as np
import pandas as pd
from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import leader_payoff_per_type
from underspec_sim.core.best_response import user_best_response
from underspec_sim.model1_pooling.solver import solve_pooling_equilibrium
from underspec_sim.model2_screening.solver import solve_menu, MenuSolutionResult


@dataclass
class RegimeComparisonResult:
    leader_payoff_pooling: float
    leader_payoff_screening: float
    dominance_holds: bool
    payoff_gap: float  # Pi_screening - Pi_pooling
    a_SE: float
    menu: Dict[str, Tuple[float, float]]
    heterogeneity: float
    bias: float
    message: str


def compare_regimes(
    two_type_or_samples: Union[Tuple[float, float, float, float], Sequence[float]],
    mu_A: Optional[float] = None,
    lambda_A: Optional[float] = None,
    c_Q: Optional[float] = None,
    params: Optional[ModelParams] = None,
    unconstrained_m: bool = False,
    tolerance: float = 1e-4,
) -> RegimeComparisonResult:
    r"""
    Compare leader payoff under Model I (pooling) and Model II (screening menu).
    Supports either:
    - (kappa_L, kappa_H, f_L, f_H) 4-tuple
    - or sequence of kappa samples (where two extreme deciles or means represent types)
    """
    if params is None:
        params = ModelParams()
    mu = params.mu_A if mu_A is None else mu_A
    lam = params.lambda_A if lambda_A is None else lambda_A
    cq = params.c_Q if c_Q is None else c_Q
    effective_params = params.model_copy(update={"mu_A": mu, "lambda_A": lam, "c_Q": cq})

    if isinstance(two_type_or_samples, tuple) and len(two_type_or_samples) == 4:
        kap_L, kap_H, f_L, f_H = two_type_or_samples
        F_samples = [kap_L] * int(f_L * 1000) + [kap_H] * int(f_H * 1000)
    else:
        samples = list(two_type_or_samples)
        kap_L = float(np.percentile(samples, 25))
        kap_H = float(np.percentile(samples, 75))
        f_L, f_H = 0.5, 0.5
        F_samples = samples

    # Model I: Pooling solve
    pool_res = solve_pooling_equilibrium(
        F_samples=F_samples,
        mu_A=mu,
        lambda_A=lam,
        c_Q=cq,
        params=effective_params,
    )
    a_SE = pool_res.a_SE_grid

    # Compute leader pooling payoff on the two types
    mL_pool = user_best_response(kap_L, a_SE, effective_params, clip=(not unconstrained_m))
    mH_pool = user_best_response(kap_H, a_SE, effective_params, clip=(not unconstrained_m))

    pi_L_pool = leader_payoff_per_type(mL_pool, a_SE, kap_L, effective_params)
    pi_H_pool = leader_payoff_per_type(mH_pool, a_SE, kap_H, effective_params)
    w_L = f_L / (f_L + f_H)
    w_H = f_H / (f_L + f_H)
    pi_pooling = float(w_L * pi_L_pool + w_H * pi_H_pool)

    # Model II: Screening menu solve
    screen_res = solve_menu(
        kappa_L=kap_L,
        kappa_H=kap_H,
        f_L=f_L,
        f_H=f_H,
        mu_A=mu,
        lambda_A=lam,
        c_Q=cq,
        params=effective_params,
        unconstrained_m=unconstrained_m,
    )
    pi_screening = screen_res.leader_payoff

    gap = pi_screening - pi_pooling
    dominance = gap >= -tolerance

    het = kap_H - kap_L
    bias = lam - cq

    msg = (
        f"Regime Comparison: Model I Pi={pi_pooling:.4f} (a_SE={a_SE:.4f}), "
        f"Model II Pi={pi_screening:.4f}. Gap={gap:.4f}. Dominance: {dominance}."
    )

    return RegimeComparisonResult(
        leader_payoff_pooling=pi_pooling,
        leader_payoff_screening=pi_screening,
        dominance_holds=dominance,
        payoff_gap=gap,
        a_SE=a_SE,
        menu=screen_res.menu,
        heterogeneity=het,
        bias=bias,
        message=msg,
    )


def sweep_regime_comparison(
    kappa_L: float = 0.30,
    kappa_H_values: Optional[Sequence[float]] = None,
    bias_values: Optional[Sequence[float]] = None,
    params: Optional[ModelParams] = None,
    unconstrained_m: bool = False,
) -> pd.DataFrame:
    r"""
    Sweeps regime comparison across a 2D grid of heterogeneity and bias.
    """
    if params is None:
        params = ModelParams()

    if kappa_H_values is None:
        kappa_H_values = [0.35, 0.40, 0.45, 0.50, 0.60]
    if bias_values is None:
        bias_values = [0.0, 1.0, 2.0, 3.0, 4.0]

    records = []
    c_Q = params.c_Q

    for kap_H in kappa_H_values:
        for b in bias_values:
            lam = c_Q + b
            comp = compare_regimes(
                two_type_or_samples=(kappa_L, kap_H, 0.5, 0.5),
                mu_A=1.0,
                lambda_A=lam,
                c_Q=c_Q,
                params=params,
                unconstrained_m=unconstrained_m,
            )
            records.append({
                "kappa_L": kappa_L,
                "kappa_H": kap_H,
                "heterogeneity": comp.heterogeneity,
                "bias": comp.bias,
                "lambda_A": lam,
                "pi_pooling": comp.leader_payoff_pooling,
                "pi_screening": comp.leader_payoff_screening,
                "payoff_gap": comp.payoff_gap,
                "dominance_holds": comp.dominance_holds,
                "a_SE": comp.a_SE,
                "a_L_screen": comp.menu["L"][1],
                "a_H_screen": comp.menu["H"][1],
            })

    return pd.DataFrame(records)
