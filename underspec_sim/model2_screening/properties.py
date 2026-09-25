"""
underspec_sim.model2_screening.properties:
- Proposition 5 verification (No distortion without bias, slack IC constraints)
- Proposition 6 verification (Distortion under bias, active constraint set reporting)
- Heterogeneity vs Bias sweep (2D grid testing separability / monotonicity)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Sequence
import numpy as np
import pandas as pd
from underspec_sim.core.params import ModelParams
from underspec_sim.core.first_best import first_best, biased_first_best
from underspec_sim.model2_screening.solver import solve_menu, MenuSolutionResult


@dataclass
class NoDistortionCheckResult:
    passed: bool
    res: MenuSolutionResult
    m_FB_L: float
    a_FB_L: float
    m_FB_H: float
    a_FB_H: float
    diff_L: float
    diff_H: float
    IC_L_slack: float
    IC_H_slack: float
    ic_slack_condition: bool
    message: str


def test_no_distortion_without_bias(
    kappa_L: float = 0.30,
    kappa_H: float = 0.50,
    f_L: float = 0.5,
    f_H: float = 0.5,
    params: Optional[ModelParams] = None,
    unconstrained_m: bool = True,
    tol: float = 1e-2,
) -> NoDistortionCheckResult:
    r"""
    Proposition 5: At \mu_A = 1, \lambda_A = c_Q (unbiased):
    Confirm the solver returns (m_\kappa, a_\kappa) = first_best(\kappa) for both types
    to numerical tolerance, and confirm IC constraints are strictly slack.
    """
    if params is None:
        params = ModelParams()

    # Unbiased parameters
    c_Q = params.c_Q
    unbiased_params = params.model_copy(update={"mu_A": 1.0, "lambda_A": c_Q})

    # Expected first best
    m_FB_L, a_FB_L = first_best(kappa_L, unbiased_params, clip=(not unconstrained_m))
    m_FB_H, a_FB_H = first_best(kappa_H, unbiased_params, clip=(not unconstrained_m))

    res = solve_menu(
        kappa_L=kappa_L,
        kappa_H=kappa_H,
        f_L=f_L,
        f_H=f_H,
        mu_A=1.0,
        lambda_A=c_Q,
        c_Q=c_Q,
        params=unbiased_params,
        unconstrained_m=unconstrained_m,
    )

    diff_L = float(np.hypot(res.m_L - m_FB_L, res.a_L - a_FB_L))
    diff_H = float(np.hypot(res.m_H - m_FB_H, res.a_H - a_FB_H))

    matches_fb = (diff_L <= tol) and (diff_H <= tol)
    ic_slack = (res.IC_L_slack > 1e-4) and (res.IC_H_slack > 1e-4)

    passed = matches_fb and ic_slack
    if passed:
        msg = (
            f"PASS (Prop 5): Menu matches First-Best ((mL,aL)=({res.m_L:.3f},{res.a_L:.3f}) vs "
            f"FB ({m_FB_L:.3f},{a_FB_L:.3f}), (mH,aH)=({res.m_H:.3f},{res.a_H:.3f}) vs FB ({m_FB_H:.3f},{a_FB_H:.3f})). "
            f"IC constraints strictly slack (IC_L={res.IC_L_slack:.4f}, IC_H={res.IC_H_slack:.4f})."
        )
    else:
        msg = (
            f"FAIL (Prop 5): matches_fb={matches_fb} (diff_L={diff_L:.4f}, diff_H={diff_H:.4f}), "
            f"ic_slack={ic_slack} (IC_L={res.IC_L_slack:.4f}, IC_H={res.IC_H_slack:.4f})."
        )

    return NoDistortionCheckResult(
        passed=passed,
        res=res,
        m_FB_L=float(m_FB_L),
        a_FB_L=float(a_FB_L),
        m_FB_H=float(m_FB_H),
        a_FB_H=float(a_FB_H),
        diff_L=diff_L,
        diff_H=diff_H,
        IC_L_slack=res.IC_L_slack,
        IC_H_slack=res.IC_H_slack,
        ic_slack_condition=ic_slack,
        message=msg,
    )


@dataclass
class DistortionUnderBiasResult:
    passed: bool
    res: MenuSolutionResult
    m_B_L: float
    a_B_L: float
    m_B_H: float
    a_B_H: float
    a_L_at_corner: bool
    a_H_strictly_below_biased_fb: bool
    distortion_size: float
    active_constraints: List[str]
    matches_paper_assumed_active: bool
    message: str


def test_distortion_under_bias(
    kappa_L: float = 0.30,
    kappa_H: float = 0.50,
    f_L: float = 0.5,
    f_H: float = 0.5,
    lambda_A: float = 4.0,
    params: Optional[ModelParams] = None,
    unconstrained_m: bool = False,
) -> DistortionUnderBiasResult:
    r"""
    Proposition 6: At \lambda_A > \mu_A * c_Q (under-asking bias):
    - a_L stays at its biased-first-best corner (a_L^{SB} = a_L^B = 0)
    - a_H comes in strictly below its own biased-first-best (a_H^{SB} < a_H^B <= 1)
    - Report active constraints (Paper assumed: IC_L binding, IR_H binding, IC_H and IR_L slack).
    """
    if params is None:
        params = ModelParams()
    effective_params = params.model_copy(update={"mu_A": 1.0, "lambda_A": lambda_A})

    m_B_L, a_B_L = biased_first_best(kappa_L, effective_params, clip=(not unconstrained_m))
    m_B_H, a_B_H = biased_first_best(kappa_H, effective_params, clip=(not unconstrained_m))

    res = solve_menu(
        kappa_L=kappa_L,
        kappa_H=kappa_H,
        f_L=f_L,
        f_H=f_H,
        mu_A=1.0,
        lambda_A=lambda_A,
        c_Q=params.c_Q,
        params=effective_params,
        unconstrained_m=unconstrained_m,
    )

    a_L_at_corner = abs(res.a_L - a_B_L) < 1e-3
    a_H_strictly_below = (res.a_H < a_B_H - 1e-3)
    distortion_size = float(a_B_H - res.a_H)

    # Core proposition claim: a_L at corner and a_H distorted downward
    passed = a_L_at_corner and a_H_strictly_below

    active_set_str = ", ".join(res.active_constraints) if res.active_constraints else "None"
    assumed_active_str = "IC_L, IR_H"

    msg = (
        f"{'PASS' if passed else 'FAIL'} (Prop 6): a_L={res.a_L:.4f} (biased FB={a_B_L:.4f}), "
        f"a_H={res.a_H:.4f} (biased FB={a_B_H:.4f}, distortion={distortion_size:.4f}). "
        f"Observed active constraints: [{active_set_str}] "
        f"(Paper proof sketch assumed: [{assumed_active_str}]). "
    )
    if not res.matches_paper_active_set:
        msg += (
            f"\nNOTE on Proof Sketch: The observed active set [{active_set_str}] differs from paper's "
            f"relaxed assumption [{assumed_active_str}]. Without cash transfers, IR_H is slack (slack={res.IR_H_slack:.2f}), "
            f"while IC_L binds (slack={res.IC_L_slack:.2e})."
        )

    return DistortionUnderBiasResult(
        passed=passed,
        res=res,
        m_B_L=float(m_B_L),
        a_B_L=float(a_B_L),
        m_B_H=float(m_B_H),
        a_B_H=float(a_B_H),
        a_L_at_corner=a_L_at_corner,
        a_H_strictly_below_biased_fb=a_H_strictly_below,
        distortion_size=distortion_size,
        active_constraints=res.active_constraints,
        matches_paper_assumed_active=res.matches_paper_active_set,
        message=msg,
    )


def sweep_distortion_vs_heterogeneity(
    params: Optional[ModelParams] = None,
    kappa_L: float = 0.30,
    kappa_H_values: Optional[Sequence[float]] = None,
    bias_values: Optional[Sequence[float]] = None,
    unconstrained_m: bool = False,
) -> pd.DataFrame:
    r"""
    2D sweep of heterogeneity (\kappa_H - \kappa_L) and bias (\lambda_A - c_Q).
    Returns DataFrame recording:
    - delta_kappa = kappa_H - kappa_L
    - bias = lambda_A - c_Q
    - a_H_B, a_H_SB, distortion = a_H_B - a_H_SB
    - active constraints
    """
    if params is None:
        params = ModelParams()

    if kappa_H_values is None:
        kappa_H_values = [0.35, 0.40, 0.45, 0.50, 0.60]
    if bias_values is None:
        bias_values = [1.0, 2.0, 3.0, 4.0]

    records = []
    c_Q = params.c_Q

    for kap_H in kappa_H_values:
        delta_kap = kap_H - kappa_L
        for b in bias_values:
            lam = c_Q + b
            eff_params = params.model_copy(update={"mu_A": 1.0, "lambda_A": lam})

            m_B_H, a_B_H = biased_first_best(kap_H, eff_params, clip=(not unconstrained_m))
            sol = solve_menu(
                kappa_L=kappa_L,
                kappa_H=kap_H,
                f_L=0.5,
                f_H=0.5,
                mu_A=1.0,
                lambda_A=lam,
                c_Q=c_Q,
                params=eff_params,
                unconstrained_m=unconstrained_m,
            )
            distortion = float(a_B_H - sol.a_H)

            records.append({
                "kappa_L": kappa_L,
                "kappa_H": kap_H,
                "delta_kappa": delta_kap,
                "bias": b,
                "lambda_A": lam,
                "a_L": sol.a_L,
                "m_L": sol.m_L,
                "a_H": sol.a_H,
                "m_H": sol.m_H,
                "a_H_B": float(a_B_H),
                "distortion": distortion,
                "IC_L_slack": sol.IC_L_slack,
                "IC_H_slack": sol.IC_H_slack,
                "IR_L_slack": sol.IR_L_slack,
                "IR_H_slack": sol.IR_H_slack,
                "active_constraints": "+".join(sol.active_constraints),
            })

    return pd.DataFrame(records)
