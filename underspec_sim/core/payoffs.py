r"""
underspec_sim.core.payoffs: User and Leader payoff functions.
Matches equations in paper/strategic_underspecification.tex:
- eq (210): U(m, a; \kappa) = V - \Lambda(1 - a)(k - m) - 0.5 * \kappa * m^2 - a * c_Q * (k - m)
- eq (179): \Pi_\kappa(m, a) = \mu_A * U(m, a; \kappa) - (\lambda_A - c_Q) * a * (k - m)
- eq (87): exact conjunctive benchmark U_exact(m; a, g) = V * q(a, g)^{k-m} - 0.5*\kappa*m^2 - a*c_Q*(k-m)
"""

from typing import Union
import numpy as np
from underspec_sim.core.params import ModelParams


def resolution_probability(a: Union[float, np.ndarray], g: float) -> Union[float, np.ndarray]:
    r"""q(a, g) = a + (1 - a)g: probability an unspecified attribute is resolved correctly."""
    return a + (1.0 - a) * g


def user_utility(
    m: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    params: ModelParams,
) -> Union[float, np.ndarray]:
    r"""
    User payoff under the linear-risk approximation (eq 210 in the paper):
    U(m, a; \kappa) = V - \Lambda(1 - a)(k - m) - 0.5 * \kappa * m^2 - a * c_Q * (k - m)
    """
    k = params.k
    Lambda = params.Lambda
    c_Q = params.c_Q
    V = params.V

    return V - Lambda * (1.0 - a) * (k - m) - 0.5 * kappa * (m ** 2) - a * c_Q * (k - m)


def user_utility_exact_conjunctive(
    m: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    params: ModelParams,
) -> Union[float, np.ndarray]:
    r"""
    Exact conjunctive user payoff (Section 2.1, eq 87 in paper):
    U_{conj}(m; a, g) = V * q(a, g)^{k - m} - 0.5 * \kappa * m^2 - a * c_Q * (k - m)
    """
    k = params.k
    c_Q = params.c_Q
    V = params.V
    q = resolution_probability(a, params.g)

    return V * (q ** (k - m)) - 0.5 * kappa * (m ** 2) - a * c_Q * (k - m)


def leader_payoff_per_type(
    m: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    params: ModelParams,
) -> Union[float, np.ndarray]:
    r"""
    Leader objective per type \kappa (eq 179 in the paper):
    \Pi_\kappa(m, a) = \mu_A * U(m, a; \kappa) - (\lambda_A - c_Q) * a * (k - m)
    Notice that when \mu_A = 1 and \lambda_A = c_Q, \Pi_\kappa(m, a) == U(m, a; \kappa).
    """
    U = user_utility(m, a, kappa, params)
    friction_gap = params.lambda_A - params.c_Q
    k = params.k
    return params.mu_A * U - friction_gap * a * (k - m)
