"""
Core primitives, payoffs, best response, and first-best benchmark.
"""

from underspec_sim.core.params import ModelParams
from underspec_sim.core.payoffs import (
    resolution_probability,
    user_utility,
    user_utility_exact_conjunctive,
    leader_payoff_per_type,
)
from underspec_sim.core.best_response import (
    user_best_response,
    user_best_response_with_diagnostics,
    beta_func,
    gamma_func,
)
from underspec_sim.core.first_best import first_best, biased_first_best

__all__ = [
    "ModelParams",
    "resolution_probability",
    "user_utility",
    "user_utility_exact_conjunctive",
    "leader_payoff_per_type",
    "user_best_response",
    "user_best_response_with_diagnostics",
    "beta_func",
    "gamma_func",
    "first_best",
    "biased_first_best",
]
