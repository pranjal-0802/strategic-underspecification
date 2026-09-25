r"""
underspec_sim.llm.simulated_user: LLM-driven simulated users with heterogeneous \kappa types.
Interacts with an assistant implementing Model I (pooling a^{SE}) or Model II (screening menu).
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
from underspec_sim.core.params import ModelParams
from underspec_sim.core.best_response import user_best_response
from underspec_sim.llm.llm_client import LLMClient


TASK_ATTRIBUTES = [
    "timeout_ms",
    "retry_policy",
    "currency_code",
    "logging_format",
    "auth_mechanism",
    "idempotency_key",
    "caching_strategy",
    "rate_limit_rpm",
    "error_response_schema",
    "webhook_callback_url",
]


@dataclass
class SimulationTrialResult:
    trial_id: int
    kappa: float
    policy_regime: str  # "pooling" or "screening"
    policy_ask_rate: float
    prescribed_m: float
    realized_m: int
    unspecified_count: int
    questions_asked: int
    attributes_guessed: int
    correct_guesses: int
    task_success: bool
    user_utility_realized: float
    leader_payoff_realized: float
    user_prompt_text: str
    assistant_response_text: str


def run_llm_trial(
    trial_id: int,
    kappa: float,
    policy_regime: str,
    ask_rate: float,
    prescribed_m: float,
    params: ModelParams,
    client: LLMClient,
    rng: Optional[np.random.Generator] = None,
) -> SimulationTrialResult:
    """
    Execute a single LLM trial with a simulated user of cost type kappa.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = int(params.k)
    m_target = int(min(k, max(0, round(prescribed_m))))

    # 1. User generates specification prompt
    simulated_attrs = {TASK_ATTRIBUTES[i]: f"val_{i+1}" for i in range(m_target)}
    user_system = (
        "You are a software engineer with specific time/effort cost. "
        "Write a concise or thorough prompt for a coding assistant based on your terseness preference."
    )
    user_instruction = (
        f"You are requesting a payment API implementation with up to {k} configuration requirements. "
        f"Your effort cost is kappa={kappa:.2f}. "
        f"Write a specification explicitly articulating {m_target} of the {k} requirements."
    )

    user_resp = client.complete(
        user_prompt=user_instruction,
        system_prompt=user_system,
        metadata={
            "role": "simulated_user",
            "trial_id": trial_id,
            "kappa": kappa,
            "m_target": m_target,
            "attributes": simulated_attrs,
        },
    )

    # In dry-run or real LLM, we measure revealed attributes
    realized_m = m_target
    unspecified = k - realized_m

    # 2. Assistant applies ask/guess policy
    questions_asked = 0
    correct_guesses = 0
    wrong_guesses = 0

    for _ in range(unspecified):
        if rng.random() < ask_rate:
            # Asked: resolved correctly with cost c_Q to user and lambda_A to leader
            questions_asked += 1
        else:
            # Guessed: succeeds with probability g
            if rng.random() < params.g:
                correct_guesses += 1
            else:
                wrong_guesses += 1

    assistant_action = "ask" if questions_asked > 0 else "guess"
    assistant_system = "You are a coding assistant adhering to a committed clarification policy."
    assistant_instruction = (
        f"The user wrote:\n{user_resp.text}\n"
        f"Your ask rate is a={ask_rate:.2f}. Unspecified attributes: {unspecified}."
    )

    assistant_resp = client.complete(
        user_prompt=assistant_instruction,
        system_prompt=assistant_system,
        metadata={
            "role": "assistant",
            "trial_id": trial_id,
            "action": assistant_action,
            "unresolved_count": unspecified,
        },
    )

    # 3. Payoffs
    task_success = (wrong_guesses == 0)
    # Effort cost = 0.5 * kappa * m^2
    effort_cost = 0.5 * kappa * (realized_m ** 2)
    # Question friction cost = c_Q * questions_asked
    friction_cost_user = params.c_Q * questions_asked
    # Loss = L * wrong_guesses
    loss_user = params.L * wrong_guesses
    user_utility = params.V - effort_cost - friction_cost_user - loss_user

    # Leader payoff
    friction_cost_leader = params.lambda_A * questions_asked
    loss_leader = params.mu_A * (params.L * wrong_guesses)
    leader_payoff = params.mu_A * params.V - effort_cost * params.mu_A - friction_cost_leader - loss_leader

    return SimulationTrialResult(
        trial_id=trial_id,
        kappa=kappa,
        policy_regime=policy_regime,
        policy_ask_rate=ask_rate,
        prescribed_m=prescribed_m,
        realized_m=realized_m,
        unspecified_count=unspecified,
        questions_asked=questions_asked,
        attributes_guessed=unspecified - questions_asked,
        correct_guesses=correct_guesses,
        task_success=task_success,
        user_utility_realized=user_utility,
        leader_payoff_realized=leader_payoff,
        user_prompt_text=user_resp.text,
        assistant_response_text=assistant_resp.text,
    )
