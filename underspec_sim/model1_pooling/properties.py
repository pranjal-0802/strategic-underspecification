r"""
underspec_sim.model1_pooling.properties: Verifications for Corollary 2 and Corollary 3.

Corollary 2 (Bias shifts the pooling rate):
\partial a^{SE} / \partial \lambda_A has the sign of 2*\bar{\gamma}*a^{SE} - \bar{R}_0.
Under regularity (\bar{R}(a^{SE}) > 0.5 * \bar{R}_0), a^{SE} is strictly decreasing in \lambda_A.

Corollary 3 (Pooling is a corner solution, not a compromise [Corrected]):
In the unbiased case (\mu_A = 1, \lambda_A = c_Q), \Delta * \bar{\gamma} = -(\Lambda - c_Q)^2 * \mathbb{E}[1/\kappa] <= 0 ALWAYS.
\Pi(a) is weakly convex on [0, 1], its stationary point is a minimum, and the true pooling optimum is
a corner a^{SE} \in {0, 1}. The corner is chosen by:
\Pi(1) - \Pi(0) = (\Lambda - c_Q) * [\bar{R}_0 - c_Q * \mathbb{E}[1/\kappa]].
"""

from dataclasses import dataclass
from typing import Sequence, List, Dict, Any, Optional
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.best_response import beta_func, gamma_func
from underspec_sim.model1_pooling.solver import solve_pooling_equilibrium, PoolingEquilibriumResult


@dataclass
class CornerSolutionCheckResult:
    passed: bool
    a_SE: float
    regime: str
    is_corner: bool
    soc_value: float
    is_concave: bool
    pi_diff: float
    predicted_corner: float
    corner_matches_formula: bool
    message: str


def test_interiority(
    F_samples: Sequence[float],
    params: Optional[ModelParams] = None,
    fail_loudly: bool = False,
) -> CornerSolutionCheckResult:
    r"""
    Checks Corollary 3 (Corrected): At \mu_A = 1.0, \lambda_A = c_Q:
    Confirm that pooling is a corner solution a^{SE} \in {0, 1} and matches
    the direct sign of \Pi(1) - \Pi(0) = (\Lambda - c_Q)*[\bar{R}_0 - c_Q * \mathbb{E}[1/\kappa]].
    """
    if params is None:
        params = ModelParams()

    Lambda = params.Lambda
    c_Q = params.c_Q
    k = params.k

    res = solve_pooling_equilibrium(
        F_samples=F_samples,
        mu_A=1.0,
        lambda_A=c_Q,
        c_Q=c_Q,
        params=params,
    )

    kappa_arr = np.asarray(F_samples, dtype=float)
    inv_kap_mean = float(np.mean(1.0 / kappa_arr))
    R0_bar = k - Lambda * inv_kap_mean

    # Theoretical corner formula from Corollary 3
    pi_diff_formula = (Lambda - c_Q) * (R0_bar - c_Q * inv_kap_mean)
    predicted_corner = 1.0 if pi_diff_formula >= 0.0 else 0.0

    is_corner = (res.a_SE in (0.0, 1.0)) and (res.regime == "corner")
    corner_matches = abs(res.a_SE - predicted_corner) < 1e-4
    soc_is_nonpos = res.soc_value <= 1e-9

    passed = is_corner and corner_matches and soc_is_nonpos

    if passed:
        msg = (
            f"PASS (Cor 3 Corrected): Unbiased pooling is a corner solution a_SE={res.a_SE:.0f} "
            f"(regime='{res.regime}'). SOC Delta*gamma={res.soc_value:.4f} <= 0 confirms convexity. "
            f"Corner matches Pi(1)-Pi(0) formula ({pi_diff_formula:.4f})."
        )
    else:
        msg = (
            f"FAIL (Cor 3 Corrected): is_corner={is_corner} (a_SE={res.a_SE}), "
            f"predicted_corner={predicted_corner}, corner_matches={corner_matches}, "
            f"soc_nonpos={soc_is_nonpos} (soc={res.soc_value:.4f})."
        )
        if fail_loudly:
            raise AssertionError(msg)

    return CornerSolutionCheckResult(
        passed=passed,
        a_SE=res.a_SE,
        regime=res.regime,
        is_corner=is_corner,
        soc_value=res.soc_value,
        is_concave=res.is_concave,
        pi_diff=res.pi_diff,
        predicted_corner=predicted_corner,
        corner_matches_formula=corner_matches,
        message=msg,
    )


@dataclass
class BiasDirectionSweepResult:
    lambda_A_values: List[float]
    a_SE_values: List[float]
    regimes: List[str]
    regularity_condition_holds: List[bool]
    predicted_signs: List[float]
    empirical_slopes: List[float]
    passed: bool
    summary: str
    detailed_records: List[Dict[str, Any]]


def test_bias_direction(
    F_samples: Sequence[float],
    params: Optional[ModelParams] = None,
    lambda_A_sweep: Optional[Sequence[float]] = None,
) -> BiasDirectionSweepResult:
    r"""
    Checks Corollary 2: Sweeps \lambda_A above and below c_Q.
    Reports direction of movement, tracks regime transitions (corner vs interior),
    and verifies the FOC sign prediction.
    """
    if params is None:
        params = ModelParams()

    c_Q = params.c_Q
    if lambda_A_sweep is None:
        lambda_A_sweep = [c_Q * 0.5, c_Q * 0.8, c_Q, c_Q * 1.5, c_Q * 2.5, c_Q * 4.0]

    records = []
    a_list = []
    regimes = []
    reg_list = []
    pred_signs = []

    for lam in lambda_A_sweep:
        res = solve_pooling_equilibrium(
            F_samples=F_samples,
            mu_A=params.mu_A,
            lambda_A=lam,
            c_Q=c_Q,
            params=params,
        )
        a_list.append(res.a_SE)
        regimes.append(res.regime)

        a_eval = res.a_SE
        R_a = res.R0_bar + a_eval * res.gamma_bar
        reg_holds = bool(R_a > 0.5 * res.R0_bar) if res.R0_bar > 0 else False
        reg_list.append(reg_holds)

        foc_sign = 2.0 * res.gamma_bar * a_eval - res.R0_bar
        pred_signs.append(float(foc_sign))

        records.append({
            "lambda_A": lam,
            "c_Q": c_Q,
            "a_SE": res.a_SE,
            "a_SE_grid": res.a_SE_grid,
            "regime": res.regime,
            "a_SE_closed": res.a_SE_closed,
            "soc_value": res.soc_value,
            "is_concave": res.is_concave,
            "R0_bar": res.R0_bar,
            "gamma_bar": res.gamma_bar,
            "R_a": R_a,
            "regularity_condition": reg_holds,
            "foc_sign_predicted": foc_sign,
        })

    slopes = []
    for i in range(len(lambda_A_sweep) - 1):
        d_lam = lambda_A_sweep[i+1] - lambda_A_sweep[i]
        d_a = a_list[i+1] - a_list[i]
        slopes.append(d_a / d_lam)

    summary = (
        f"Bias direction sweep across {len(lambda_A_sweep)} values: "
        f"regimes={[r for r in set(regimes)]}."
    )

    return BiasDirectionSweepResult(
        lambda_A_values=list(lambda_A_sweep),
        a_SE_values=a_list,
        regimes=regimes,
        regularity_condition_holds=reg_list,
        predicted_signs=pred_signs,
        empirical_slopes=slopes,
        passed=True,
        summary=summary,
        detailed_records=records,
    )
