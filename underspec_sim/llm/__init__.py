"""
LLM integration and simulation module.
"""

from underspec_sim.llm.llm_client import LLMClient, LLMResponse
from underspec_sim.llm.simulated_user import run_llm_trial, SimulationTrialResult, TASK_ATTRIBUTES

__all__ = [
    "LLMClient",
    "LLMResponse",
    "run_llm_trial",
    "SimulationTrialResult",
    "TASK_ATTRIBUTES",
]
