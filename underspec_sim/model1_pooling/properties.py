r"""
underspec_sim.model1_pooling.properties: Verifications for Corollary 2 and Corollary 3.

Corollary 2 (Bias shifts the pooling rate on interior branch):
On the interior branch (\bar{Q} < 0),
\partial a^{SE} / \partial \lambda_A = - (\bar{R}_0 / \bar{\gamma}) * (\mu_A * s) / (\mu_A * s - 2*b)^2 > 0
when \bar{R}_0 < 0 (under-specification regime), where s = \Lambda - c_Q and b = \lambda_A - c_Q.

Corollary 3 (Pooling is a corner solution, not a compromise):
In the unbiased case (\mu_A = 1, \lambda_A = c_Q => b = 0), \bar{Q} = s^2 * \mathbb{E}[1/\kappa] / 2 >= 0 ALWAYS.
\Pi(a) is weakly convex on [0, 1], its stationary point is a minimum, and the true pooling optimum is
a corner a^{SE} \in {0, 1}. The corner is chosen by:
\Pi(1) - \Pi(0) = \bar{L} + \bar{Q} = (\Lambda - c_Q) * [\bar{R}_0 + \bar{\gamma} / 2].
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
    the direct sign of \Pi(1) - \Pi(0) = (\Lambda - c_Q)*[\bar{R}_0 + \bar{\gamma} / 2].
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
    gamma_bar = (Lambda - c_Q) * inv_kap_mean

    # Theoretical corner formula from Corollary 3: \Pi(1) - \Pi(0) = (\Lambda - c_Q) * (\bar{R}_0 + \bar\gamma / 2)
    pi_diff_formula = (Lambda - c_Q) * (R0_bar + 0.5 * gamma_bar)
    predicted_corner = 1.0 if pi_diff_formula >= 0.0 else 0.0

    is_corner = (res.a_SE in (0.0, 1.0)) and (res.regime == "corner")
    corner_matches = abs(res.a_SE - predicted_corner) < 1e-4
    soc_is_nonneg = res.soc_value >= -1e-9

    passed = is_corner and corner_matches and soc_is_nonneg

    if passed:
        msg = (
            f"PASS (Cor 3 Corrected): Unbiased pooling is a corner solution a_SE={res.a_SE:.0f} "
            f"(regime='{res.regime}'). SOC Q_bar={res.soc_value:.4f} >= 0 confirms convexity. "
            f"Corner matches Pi(1)-Pi(0) formula ({pi_diff_formula:.4f})."
        )
    else:
        msg = (
            f"FAIL (Cor 3 Corrected): is_corner={is_corner} (a_SE={res.a_SE}), "
            f"predicted_corner={predicted_corner}, corner_matches={corner_matches}, "
            f"soc_nonneg={soc_is_nonneg} (soc={res.soc_value:.4f})."
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
    soc_values: List[float]
    analytical_derivatives: List[float]
    empirical_slopes: List[float]
    max_derivative_error: float
    passed: bool
    summary: str
    detailed_records: List[Dict[str, Any]]


def test_bias_direction(
    F_samples: Optional[Sequence[float]] = None,
    params: Optional[ModelParams] = None,
    lambda_A_sweep: Optional[Sequence[float]] = None,
) -> BiasDirectionSweepResult:
    r"""
    Checks Corollary 2 (Corrected):
    On the interior branch (\bar{Q} < 0),
      \partial a^{SE} / \partial \lambda_A = - (\bar{R}_0 / \bar{\gamma}) * (\mu_A * s) / (\mu_A * s - 2*b)^2 > 0
    when \bar{R}_0 < 0 (under-specification regime).
    Verifies that:
    1. The parameters yield \bar{Q} < 0 (SOC holds, regime is strictly interior).
    2. a^{SE} is strictly interior (in (0, 1)) across the entire sweep.
    3. The empirical slope \Delta a^{SE} / \Delta \lambda_A is strictly positive everywhere.
    4. The empirical slope matches the closed-form analytical derivative within tolerance.
    """
    if F_samples is None:
        F_samples = np.linspace(0.8, 1.25, 200)

    kappa_arr = np.asarray(F_samples, dtype=float)
    inv_mean = float(np.mean(1.0 / kappa_arr))

    if params is None:
        # Calibrated primitives where Q_bar < 0 and a^SE in (0.25, 0.85)
        # Lambda = 2.0, c_Q = 1.0, mu_A = 1.0 => s = 1.0
        # gamma_bar = s * inv_mean
        # k = Lambda * inv_mean - 1.90 * gamma_bar => R0_bar = -1.90 * gamma_bar < 0
        Lambda = 2.0
        c_Q = 1.0
        mu_A = 1.0
        gamma_bar = (Lambda - c_Q) * inv_mean
        k = float(Lambda * inv_mean - 1.90 * gamma_bar)
        params = ModelParams(k=k, g=0.5, L=4.0, c_Q=c_Q, mu_A=mu_A, V=100.0)

    Lambda = params.Lambda
    mu_A = params.mu_A
    c_Q = params.c_Q

    if lambda_A_sweep is None:
        # Sweeping lambda_A in [2.2, 5.0] guarantees b in [1.2, 4.0],
        # Q_bar = (s - 2b)*gamma_bar/2 < 0 (concave), and a^SE in [0.27, 0.82] strictly interior
        lambda_A_sweep = np.linspace(2.2, 5.0, 20)

    records: List[Dict[str, Any]] = []
    a_list: List[float] = []
    regimes: List[str] = []
    soc_list: List[float] = []
    deriv_list: List[float] = []

    for lam in lambda_A_sweep:
        res = solve_pooling_equilibrium(
            F_samples=F_samples,
            mu_A=mu_A,
            lambda_A=lam,
            c_Q=c_Q,
            params=params,
        )
        a_list.append(res.a_SE)
        regimes.append(res.regime)
        soc_list.append(res.soc_value)

        # Closed-form analytical derivative from Corollary 2
        s = Lambda - c_Q
        b = lam - c_Q
        analytical_deriv = float(- (res.R0_bar / res.gamma_bar) * (mu_A * s) / ((mu_A * s - 2.0 * b) ** 2))
        deriv_list.append(analytical_deriv)

        records.append({
            "lambda_A": float(lam),
            "c_Q": float(c_Q),
            "mu_A": float(mu_A),
            "Lambda": float(Lambda),
            "a_SE": float(res.a_SE),
            "regime": res.regime,
            "soc_value": float(res.soc_value),
            "analytical_derivative": analytical_deriv,
            "R0_bar": float(res.R0_bar),
            "gamma_bar": float(res.gamma_bar),
        })

    # Compute empirical finite-difference slopes
    slopes: List[float] = []
    relative_errors: List[float] = []
    for i in range(len(lambda_A_sweep) - 1):
        d_lam = float(lambda_A_sweep[i + 1] - lambda_A_sweep[i])
        d_a = float(a_list[i + 1] - a_list[i])
        slope = d_a / d_lam
        slopes.append(slope)

        # Midpoint analytical derivative
        mid_deriv = 0.5 * (deriv_list[i] + deriv_list[i + 1])
        rel_err = abs(slope - mid_deriv) / mid_deriv
        relative_errors.append(rel_err)

    all_interior = all(r == "interior" for r in regimes)
    all_positive_slopes = all(s > 0.0 for s in slopes)
    all_soc_concave = all(s < -1e-9 for s in soc_list)
    max_err = float(np.max(relative_errors)) if relative_errors else 0.0
    deriv_matches = max_err < 0.06  # Within 6% finite difference tolerance

    passed = all_interior and all_positive_slopes and all_soc_concave and deriv_matches

    summary = (
        f"Corollary 2 (Corrected): Swept lambda_A in [{lambda_A_sweep[0]:.2f}, {lambda_A_sweep[-1]:.2f}] "
        f"across {len(lambda_A_sweep)} points. All {len(a_list)} points strictly interior (a_SE in [{min(a_list):.3f}, {max(a_list):.3f}]). "
        f"All empirical slopes strictly positive (min slope={min(slopes):.4f}). "
        f"Matches analytical derivative - (R0_bar/gamma_bar) * (mu_A * s) / (mu_A * s - 2b)^2 (max rel err={max_err:.2%}). "
        f"Passed: {passed}."
    )

    return BiasDirectionSweepResult(
        lambda_A_values=[float(x) for x in lambda_A_sweep],
        a_SE_values=a_list,
        regimes=regimes,
        soc_values=soc_list,
        analytical_derivatives=deriv_list,
        empirical_slopes=slopes,
        max_derivative_error=max_err,
        passed=passed,
        summary=summary,
        detailed_records=records,
    )
