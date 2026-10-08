r"""
underspec_sim.core.first_best: Proposition 3 (First-Best Benchmark).
Characterizes the fully-informed, fully-aligned social optimum:
(m_FB(\kappa), a_FB(\kappa)) = \arg\max_{m, a} U(m, a; \kappa)

Proposition 3 (bang-bang threshold in \kappa):
U is linear in a for fixed m, so a_FB(\kappa) \in {0, 1}.
Define \kappa^* = (\Lambda + c_Q) / (2k).
If \Lambda > c_Q:
- For \kappa > \kappa^*: a_FB = 1, m_FB = c_Q / \kappa
- For \kappa < \kappa^*: a_FB = 0, m_FB = \Lambda / \kappa
- For \kappa == \kappa^*: indifferent
"""

from typing import Tuple, Union
import numpy as np
from underspec_sim.core.params import ModelParams


def first_best(
    kappa: Union[float, np.ndarray],
    params: ModelParams,
    clip: bool = True,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""
    Compute first-best policy bundle (m_FB, a_FB).

    Under the linear-risk specification with \Lambda > c_Q and physical constraint m \in [0, k]:
    \partial U / \partial a = (\Lambda - c_Q)(k - m) >= 0 everywhere on [0, k].
    Asking weakly dominates guessing for every type \kappa:
    - For \kappa > c_Q / k: m_FB = c_Q / \kappa < k, and a_FB = 1 is strictly optimal.
    - For \kappa <= c_Q / k: m_FB = k, all attributes are specified upstream (k - m = 0),
      and asking is payoff-irrelevant (conventionally a_FB = 1).
    When clip=False (unconstrained theoretical benchmark), the unclipped formula
    recovers the legacy \kappa^* threshold (\Lambda + c_Q) / (2k).
    """
    Lambda = params.Lambda
    c_Q = params.c_Q
    k = params.k
    kappa_star = params.kappa_star

    if isinstance(kappa, np.ndarray):
        a_FB = np.zeros_like(kappa, dtype=float)
        m_FB = np.zeros_like(kappa, dtype=float)

        if clip:
            # Physical domain m in [0, k]: when Lambda > c_Q, a=1 weakly dominates for all types
            if Lambda > c_Q:
                a_FB[:] = 1.0
                m_FB = np.clip(c_Q / kappa, 0.0, k)
            else:
                a_FB[:] = 0.0
                m_FB = np.clip(Lambda / kappa, 0.0, k)
        else:
            # Unconstrained algebraic benchmark
            if Lambda > c_Q:
                mask_ask = kappa >= kappa_star
                mask_guess = kappa < kappa_star
                a_FB[mask_ask] = 1.0
                m_FB[mask_ask] = c_Q / kappa[mask_ask]
                a_FB[mask_guess] = 0.0
                m_FB[mask_guess] = Lambda / kappa[mask_guess]
            else:
                a_FB[:] = 0.0
                m_FB[:] = Lambda / kappa

        return m_FB, a_FB

    else:
        kap = float(kappa)
        if clip:
            if Lambda > c_Q:
                a_FB = 1.0
                m_FB = float(np.clip(c_Q / kap, 0.0, k))
            else:
                a_FB = 0.0
                m_FB = float(np.clip(Lambda / kap, 0.0, k))
        else:
            if Lambda > c_Q:
                if kap >= kappa_star:
                    a_FB = 1.0
                    m_FB = c_Q / kap
                else:
                    a_FB = 0.0
                    m_FB = Lambda / kap
            else:
                a_FB = 0.0
                m_FB = Lambda / kap

        return m_FB, a_FB


def biased_first_best(
    kappa: Union[float, np.ndarray],
    params: ModelParams,
    clip: bool = False,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""
    Compute leader's biased preferred bundle (m_B, a_B) = argmax_{m,a} \Pi_\kappa(m, a)
    per Section 6.3 in the paper.
    """
    kappa_star_B = params.kappa_star_biased
    Lambda = params.Lambda
    c_Q_eff = params.c_Q + (params.lambda_A - params.c_Q) / params.mu_A
    k = params.k

    if isinstance(kappa, np.ndarray):
        a_B = np.zeros_like(kappa, dtype=float)
        m_B = np.zeros_like(kappa, dtype=float)

        if Lambda > c_Q_eff:
            mask_ask = kappa > kappa_star_B
            a_B[mask_ask] = 1.0
            m_B[mask_ask] = c_Q_eff / kappa[mask_ask]

            mask_guess = kappa <= kappa_star_B
            a_B[mask_guess] = 0.0
            m_B[mask_guess] = Lambda / kappa[mask_guess]
        else:
            a_B[:] = 0.0
            m_B[:] = Lambda / kappa

        if clip:
            m_B = np.clip(m_B, 0.0, k)
        return m_B, a_B
    else:
        kap = float(kappa)
        if Lambda > c_Q_eff:
            if kap > kappa_star_B:
                a_B = 1.0
                m_B = c_Q_eff / kap
            else:
                a_B = 0.0
                m_B = Lambda / kap
        else:
            a_B = 0.0
            m_B = Lambda / kap

        if clip:
            m_B = float(np.clip(m_B, 0.0, k))
        return m_B, a_B


def first_best_exact(
    kappa: float,
    params: ModelParams,
    m_grid_points: int = 2001,
) -> Tuple[float, float, float]:
    r"""
    Compute the true first-best bundle (m_FB, a_FB, U_FB) under the exact
    conjunctive model U_{conj}(m; a, g) = V * q(a, g)^{k-m} - 0.5 * \kappa * m^2 - a * c_Q * (k-m)
    by searching over m in [0, k] and a in {0, 1}.

    In this exact specification, a genuine threshold in \kappa emerges where
    asking vs guessing trades off based on whether the marginal accuracy recovery
    V(1-g)q^{k-m-1} justifies the cognitive questioning cost c_Q.
    """
    from underspec_sim.core.payoffs import user_utility_exact_conjunctive
    k = params.k
    m_grid = np.linspace(0.0, k, m_grid_points)

    u_a0 = user_utility_exact_conjunctive(m_grid, 0.0, kappa, params)
    best_idx_0 = int(np.argmax(u_a0))
    best_m_0 = float(m_grid[best_idx_0])
    best_u_0 = float(u_a0[best_idx_0])

    u_a1 = user_utility_exact_conjunctive(m_grid, 1.0, kappa, params)
    best_idx_1 = int(np.argmax(u_a1))
    best_m_1 = float(m_grid[best_idx_1])
    best_u_1 = float(u_a1[best_idx_1])

    if best_u_1 >= best_u_0:
        return best_m_1, 1.0, best_u_1
    else:
        return best_m_0, 0.0, best_u_0
