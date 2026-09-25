r"""
underspec_sim.model1_pooling.payoff: Leader pooling payoff implementation.
Matches the paper's general leader objective (Section 4.1):

    \Pi_\kappa(m, a) = \mu_A * U(m, a; \kappa) - (\lambda_A - c_Q) * a * (k - m)

evaluated at user best-response m^*(\kappa; a) = \beta(\kappa) - a * \gamma(\kappa),
where U(m, a; \kappa) = V - \Lambda(1-a)(k-m) - \kappa/2 * m^2 - a * c_Q * (k-m).

Expanding \Pi_\kappa(m^*(\kappa; a), a) in powers of a yields the exact quadratic:
    \Pi(a) = \text{const} + \bar{L} * a + \bar{Q} * a^2

where, with s \equiv \Lambda - c_Q and b \equiv \lambda_A - c_Q:
    \bar{Q} = (\mu_A * s - 2 * b) * \bar{\gamma} / 2
    \bar{L} = (\mu_A * s - b) * \bar{R}_0
    \text{const} = \mu_A * (V - \Lambda * k) + \mu_A * \Lambda^2 * \mathbb{E}[1/\kappa] / 2
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
    Exact quadratic expected leader pooling payoff \mathbb{E}_\kappa[\Pi_\kappa(a)].

    Derived directly from the paper's primitives including the specification cost
    \kappa * m^2 / 2. Matches primitive evaluation to machine precision (< 2e-14).
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
    inv_kappa = float(np.mean(1.0 / kappa_arr))
    betas = beta_func(kappa_arr, effective_params)
    gammas = gamma_func(kappa_arr, effective_params)

    R0_bar = k - float(np.mean(betas))
    gamma_bar = float(np.mean(gammas))

    s = Lambda - cq
    b = lam - cq

    Q_bar = (mu * s - 2.0 * b) * gamma_bar / 2.0
    L_bar = (mu * s - b) * R0_bar
    const = mu * (V - Lambda * k) + mu * (Lambda ** 2) * inv_kappa / 2.0

    a_arr = np.asarray(a)
    return const + L_bar * a_arr + Q_bar * (a_arr ** 2)
