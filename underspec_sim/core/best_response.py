r"""
underspec_sim.core.best_response: User best-response function against ask rate a.
Matches Section 5.1 in strategic_underspecification.tex:
m^*(\kappa; a) = \beta(\kappa) - a * \gamma(\kappa)
where:
\beta(\kappa) = \Lambda / \kappa
\gamma(\kappa) = (\Lambda - c_Q) / \kappa
\partial m^* / \partial a = -\gamma(\kappa) = (c_Q - \Lambda) / \kappa

Includes both unconstrained-continuous version and clipped-to-[0, k] version,
logging when clipping binds as the closed forms in the paper assume interior solutions.
"""

import logging
from typing import Union, Tuple, Optional
import numpy as np
from underspec_sim.core.params import ModelParams

logger = logging.getLogger(__name__)


def beta_func(kappa: Union[float, np.ndarray], params: ModelParams) -> Union[float, np.ndarray]:
    r"""\beta(\kappa) = \Lambda / \kappa."""
    return params.Lambda / kappa


def gamma_func(kappa: Union[float, np.ndarray], params: ModelParams) -> Union[float, np.ndarray]:
    r"""\gamma(\kappa) = (\Lambda - c_Q) / \kappa."""
    return (params.Lambda - params.c_Q) / kappa


def user_best_response(
    kappa: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    params: ModelParams,
    clip: bool = True,
    log_clipping: bool = True,
) -> Union[float, np.ndarray]:
    r"""
    Compute user best-response specification level m* against ask rate a.

    Unconstrained formula (Section 5.1):
        m^*(\kappa; a) = \beta(\kappa) - a * \gamma(\kappa)
                       = (\Lambda(1 - a) + a * c_Q) / \kappa

    If clip=True, clips m* to [0, k] and logs if clipping bound.
    """
    beta = beta_func(kappa, params)
    gamma = gamma_func(kappa, params)
    m_unconstrained = beta - a * gamma

    if not clip:
        return m_unconstrained

    # Clipped to [0, k]
    if isinstance(m_unconstrained, np.ndarray):
        m_clipped = np.clip(m_unconstrained, 0.0, params.k)
        lower_bound_active = np.any(m_unconstrained < 0.0)
        upper_bound_active = np.any(m_unconstrained > params.k)
        if log_clipping and (lower_bound_active or upper_bound_active):
            logger.info(
                f"Clipping bound active: {np.sum(m_unconstrained < 0)} values < 0, "
                f"{np.sum(m_unconstrained > params.k)} values > k (k={params.k})"
            )
        return m_clipped
    else:
        m_val = float(m_unconstrained)
        if m_val < 0.0:
            if log_clipping:
                logger.info(f"Clipping bound active (lower): m* unconstrained = {m_val:.4f} < 0, clipped to 0.0")
            return 0.0
        elif m_val > params.k:
            if log_clipping:
                logger.info(f"Clipping bound active (upper): m* unconstrained = {m_val:.4f} > k={params.k}, clipped to {params.k}")
            return float(params.k)
        return m_val


def user_best_response_with_diagnostics(
    kappa: float,
    a: float,
    params: ModelParams,
) -> Tuple[float, float, bool]:
    """
    Returns (m_clipped, m_unconstrained, is_clipped_binding).
    """
    m_unconstrained = float(user_best_response(kappa, a, params, clip=False, log_clipping=False))
    m_clipped = float(np.clip(m_unconstrained, 0.0, params.k))
    is_clipped = abs(m_unconstrained - m_clipped) > 1e-9
    return m_clipped, m_unconstrained, is_clipped
