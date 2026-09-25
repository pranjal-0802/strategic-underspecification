r"""
underspec_sim.model2_screening.solver: Constrained optimization of screening menu.
Solves the full constrained mechanism design problem:
\max_{(m_L, a_L), (m_H, a_H)} f_L * \Pi_{\kappa_L}(m_L, a_L) + f_H * \Pi_{\kappa_H}(m_H, a_H)
subject to:
- IC_L: U(m_L, a_L; \kappa_L) >= U(m_H, a_H; \kappa_L)
- IC_H: U(m_H, a_H; \kappa_H) >= U(m_L, a_L; \kappa_H)
- IR_L: U(m_L, a_L; \kappa_L) >= \underline{U}
- IR_H: U(m_H, a_H; \kappa_H) >= \underline{U}
- Bounds: m_L, m_H in [0, k], a_L, a_H in [0, 1]

Explicitly evaluates constraint activity at the solution rather than assuming
which constraints bind.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.optimize import minimize
from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import user_utility, leader_payoff_per_type
from underspec_sim.core.first_best import first_best, biased_first_best


@dataclass
class MenuSolutionResult:
    m_L: float
    a_L: float
    m_H: float
    a_H: float
    menu: Dict[str, Tuple[float, float]]
    leader_payoff: float
    user_utility_L: float
    user_utility_H: float
    IC_L_slack: float
    IC_H_slack: float
    IR_L_slack: float
    IR_H_slack: float
    active_constraints: List[str]
    matches_paper_active_set: bool
    success: bool
    message: str

    def as_tuple_dict(self) -> Dict[str, Tuple[float, float]]:
        return {"L": (self.m_L, self.a_L), "H": (self.m_H, self.a_H)}


def solve_menu(
    kappa_L: float,
    kappa_H: float,
    f_L: float,
    f_H: float,
    mu_A: Optional[float] = None,
    lambda_A: Optional[float] = None,
    c_Q: Optional[float] = None,
    params: Optional[ModelParams] = None,
    unconstrained_m: bool = False,
    active_tol: float = 1e-4,
    method: str = "SLSQP",
) -> MenuSolutionResult:
    r"""
    Solves the leader's Model II screening problem using numerical constrained optimization.
    Returns the optimal menu {(m_L, a_L), (m_H, a_H)} and constraint activity report.
    """
    if params is None:
        params = ModelParams()
    mu = params.mu_A if mu_A is None else mu_A
    lam = params.lambda_A if lambda_A is None else lambda_A
    cq = params.c_Q if c_Q is None else c_Q

    effective_params = params.model_copy(update={"mu_A": mu, "lambda_A": lam, "c_Q": cq})
    k = effective_params.k
    U_bar = effective_params.U_bar

    # Normalize shares
    total_f = f_L + f_H
    w_L = f_L / total_f
    w_H = f_H / total_f

    # Objective: maximize expected leader payoff => minimize negative
    # Mild tie-breaker (1e-8 * aL) ensures that when m_L = k (so k - m_L = 0 and payoff is
    # mathematically invariant to a_L), the solver selects a_L = 0 (matching a_L^B = 0 per Prop 6)
    # rather than wandering arbitrarily on the plateau.
    def objective(x: np.ndarray) -> float:
        mL, aL, mH, aH = x
        pi_L = leader_payoff_per_type(mL, aL, kappa_L, effective_params)
        pi_H = leader_payoff_per_type(mH, aH, kappa_H, effective_params)
        return -(w_L * pi_L + w_H * pi_H) + 1e-8 * aL

    # Constraints
    def ic_L_con(x: np.ndarray) -> float:
        # U(m_L, a_L; kappa_L) - U(m_H, a_H; kappa_L) >= 0
        return user_utility(x[0], x[1], kappa_L, effective_params) - user_utility(x[2], x[3], kappa_L, effective_params)

    def ic_H_con(x: np.ndarray) -> float:
        # U(m_H, a_H; kappa_H) - U(m_L, a_L; kappa_H) >= 0
        return user_utility(x[2], x[3], kappa_H, effective_params) - user_utility(x[0], x[1], kappa_H, effective_params)

    def ir_L_con(x: np.ndarray) -> float:
        # U(m_L, a_L; kappa_L) - U_bar >= 0
        return user_utility(x[0], x[1], kappa_L, effective_params) - U_bar

    def ir_H_con(x: np.ndarray) -> float:
        # U(m_H, a_H; kappa_H) - U_bar >= 0
        return user_utility(x[2], x[3], kappa_H, effective_params) - U_bar

    constraints = [
        {"type": "ineq", "fun": ic_L_con},
        {"type": "ineq", "fun": ic_H_con},
        {"type": "ineq", "fun": ir_L_con},
        {"type": "ineq", "fun": ir_H_con},
    ]

    m_upper = None if unconstrained_m else k
    bounds = [
        (0.0, m_upper),
        (0.0, 1.0),
        (0.0, m_upper),
        (0.0, 1.0),
    ]

    # Generate initial guesses
    m_FB_L, a_FB_L = first_best(kappa_L, effective_params, clip=(not unconstrained_m))
    m_FB_H, a_FB_H = first_best(kappa_H, effective_params, clip=(not unconstrained_m))

    m_B_L, a_B_L = biased_first_best(kappa_L, effective_params, clip=(not unconstrained_m))
    m_B_H, a_B_H = biased_first_best(kappa_H, effective_params, clip=(not unconstrained_m))

    candidates_x0 = [
        [float(m_FB_L), float(a_FB_L), float(m_FB_H), float(a_FB_H)],
        [float(m_B_L), float(a_B_L), float(m_B_H), float(a_B_H)],
        [k * 0.8, 0.0, k * 0.4, 1.0],
        [k * 0.5, 0.5, k * 0.5, 0.5],
    ]

    best_res = None
    best_val = float("inf")

    for x0 in candidates_x0:
        # Clip x0 to bounds
        for i in range(4):
            low, high = bounds[i]
            if high is not None:
                x0[i] = min(high, max(low, x0[i]))
            else:
                x0[i] = max(low, x0[i])

        try:
            res = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            # Check feasibility
            is_feasible = (
                ic_L_con(res.x) >= -active_tol
                and ic_H_con(res.x) >= -active_tol
                and ir_L_con(res.x) >= -active_tol
                and ir_H_con(res.x) >= -active_tol
            )
            if is_feasible and res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is None or not best_res.success:
        # Fallback to standard SLSQP with default starting point
        res = minimize(
            objective,
            [float(m_B_L), float(a_B_L), float(m_B_H), float(a_B_H)],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        best_res = res

    opt_x = best_res.x
    mL, aL, mH, aH = float(opt_x[0]), float(opt_x[1]), float(opt_x[2]), float(opt_x[3])

    ic_L_val = float(ic_L_con(opt_x))
    ic_H_val = float(ic_H_con(opt_x))
    ir_L_val = float(ir_L_con(opt_x))
    ir_H_val = float(ir_H_con(opt_x))

    active_cons = []
    if abs(ic_L_val) <= active_tol:
        active_cons.append("IC_L")
    if abs(ic_H_val) <= active_tol:
        active_cons.append("IC_H")
    if abs(ir_L_val) <= active_tol:
        active_cons.append("IR_L")
    if abs(ir_H_val) <= active_tol:
        active_cons.append("IR_H")

    # Paper's corrected active set in proof sketch: IC_L alone binds (IR_H, IC_H, IR_L slack)
    matches_assumed = ("IC_L" in active_cons) and ("IR_H" not in active_cons)

    total_payoff = -float(best_res.fun)
    u_L = float(user_utility(mL, aL, kappa_L, effective_params))
    u_H = float(user_utility(mH, aH, kappa_H, effective_params))

    return MenuSolutionResult(
        m_L=mL,
        a_L=aL,
        m_H=mH,
        a_H=aH,
        menu={"L": (mL, aL), "H": (mH, aH)},
        leader_payoff=total_payoff,
        user_utility_L=u_L,
        user_utility_H=u_H,
        IC_L_slack=ic_L_val,
        IC_H_slack=ic_H_val,
        IR_L_slack=ir_L_val,
        IR_H_slack=ir_H_val,
        active_constraints=active_cons,
        matches_paper_active_set=matches_assumed,
        success=bool(best_res.success),
        message=str(best_res.message),
    )
