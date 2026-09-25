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
    clip: bool = False,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""
    Compute first-best policy bundle (m_FB, a_FB) per Proposition 3.

    Parameters:
    - kappa: user specification cost type
    - params: model primitives
    - clip: if True, clips m_FB to [0, k]. The paper's closed-form assumes clip=False.

    Returns:
    - (m_FB, a_FB)
    """
    kappa_star = params.kappa_star
    Lambda = params.Lambda
    c_Q = params.c_Q
    k = params.k

    if isinstance(kappa, np.ndarray):
        a_FB = np.zeros_like(kappa, dtype=float)
        m_FB = np.zeros_like(kappa, dtype=float)

        if Lambda > c_Q:
            mask_ask = kappa > kappa_star
            mask_guess = kappa < kappa_star
            # At tie kappa == kappa_star, payoff at 0 and 1 are identical
            a_FB[mask_ask] = 1.0
            m_FB[mask_ask] = c_Q / kappa[mask_ask]

            a_FB[mask_guess] = 0.0
            m_FB[mask_guess] = Lambda / kappa[mask_guess]

            tie_mask = ~mask_ask & ~mask_guess
            a_FB[tie_mask] = 1.0
            m_FB[tie_mask] = c_Q / kappa[tie_mask]
        else:
            # When Lambda <= c_Q, asking is more expensive than guessing loss
            a_FB[:] = 0.0
            m_FB[:] = Lambda / kappa

        if clip:
            m_FB = np.clip(m_FB, 0.0, k)
        return m_FB, a_FB

    else:
        kap = float(kappa)
        if Lambda > c_Q:
            if kap > kappa_star:
                a_FB = 1.0
                m_FB = c_Q / kap
            elif kap < kappa_star:
                a_FB = 0.0
                m_FB = Lambda / kap
            else:
                # Indifferent at exact threshold
                a_FB = 1.0
                m_FB = c_Q / kap
        else:
            a_FB = 0.0
            m_FB = Lambda / kap

        if clip:
            m_FB = float(np.clip(m_FB, 0.0, k))
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
