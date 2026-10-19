"""
Model III: screening a continuum of user types (Section 8).
"""

from underspec_sim.model3_continuum.standard import (
    TypeDistribution,
    standard_threshold,
    standard_value,
    standard_on_types,
    tail_condition,
    solve_discrete_menu,
    recover_wedge,
)

__all__ = [
    "TypeDistribution",
    "standard_threshold",
    "standard_value",
    "standard_on_types",
    "tail_condition",
    "solve_discrete_menu",
    "recover_wedge",
]
