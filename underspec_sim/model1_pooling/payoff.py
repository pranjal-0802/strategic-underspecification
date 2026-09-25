r"""
underspec_sim.model1_pooling.payoff: Leader pooling payoff implementation.
Matches Section 5.2 in strategic_underspecification.tex:
- R(a, \kappa) = k - m^*(\kappa; a)
- \bar{R}_0 = k - \mathbb{E}[\beta(\kappa)]
- \bar{\gamma} = \mathbb{E}[\gamma(\kappa)]
- \bar{R}(a) = \bar{R}_0 + a * \bar{\gamma}
- C(a) = \mu_A * \Lambda + a * (\lambda_A - \mu_A * \Lambda) = C_0 + a * \Delta
- \Pi(a) = \mu_A * V - C(a) * \bar{R}(a)
         = \mu_A * V - C_0 * \bar{R}_0 - a * [C_0 * \bar{\gamma} + \Delta * \bar{R}_0] - a^2 * \Delta * \bar{\gamma}
"""

from typing import Union, Sequence, Optional
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.best_response import beta_func, gamma_func


def leader_payoff_pooling(
    a: Union[float, np.ndarray],
    F_samples: Sequence[float],
    mu_A: Optional[float] = None,
    lambda_A: Optional[float] = None,
    c_Q: Optional[float] = None,
    params: Optional[ModelParams] = None,
) -> Union[float, np.ndarray]:
    r"""
    Implements \Pi(a) = \mu_A * V - C(a) * \bar{R}(a) (the exact quadratic form in the paper).

    Parameters:
    - a: ask rate scalar or array in [0, 1]
    - F_samples: sample draws of user cost types \kappa ~ F
    - mu_A: leader downstream welfare weight (defaults to params.mu_A)
    - lambda_A: leader friction cost per question (defaults to params.lambda_A)
    - c_Q: user cost per question (defaults to params.c_Q)
    - params: base ModelParams
    """
    if params is None:
        params = ModelParams()
    mu = params.mu_A if mu_A is None else mu_A
    lam = params.lambda_A if lambda_A is None else lambda_A
    cq = params.c_Q if c_Q is None else c_Q

    effective_params = params.model_copy(update={"mu_A": mu, "lambda_A": lam, "c_Q": cq})
    Lambda = effective_params.Lambda
    k = effective_params.k
    V = effective_params.V

    kappa_arr = np.asarray(F_samples, dtype=float)
    betas = beta_func(kappa_arr, effective_params)
    gammas = gamma_func(kappa_arr, effective_params)

    R0_bar = k - float(np.mean(betas))
    gamma_bar = float(np.mean(gammas))

    C0 = mu * Lambda
    Delta = lam - mu * Lambda

    # Quadratic form: \Pi(a) = \mu V - C0*R0_bar - a*(C0*gamma_bar + Delta*R0_bar) - a^2 * Delta*gamma_bar
    linear_coeff = C0 * gamma_bar + Delta * R0_bar
    quad_coeff = Delta * gamma_bar

    return mu * V - C0 * R0_bar - a * linear_coeff - (a ** 2) * quad_coeff
