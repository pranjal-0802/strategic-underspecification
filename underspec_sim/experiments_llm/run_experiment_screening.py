"""
underspec_sim.experiments_llm.run_experiment_screening:
Executes LLM-driven experiment under Model II Screening policy menu {(m_L, a_L), (m_H, a_H)}.
Simulated users self-select into their optimal menu option and specify accordingly.
"""

from typing import List
import numpy as np
import pandas as pd
from underspec_sim.core.params import ModelParams
from underspec_sim.model2_screening.solver import solve_menu
from underspec_sim.llm.llm_client import LLMClient
from underspec_sim.llm.simulated_user import run_llm_trial, SimulationTrialResult
from underspec_sim.experiments_llm.config import LLMExperimentConfig


def run_screening_llm_experiment(
    config: LLMExperimentConfig,
    params: ModelParams,
    kappa_L: float = 0.30,
    kappa_H: float = 0.50,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    client = LLMClient(
        model=config.model,
        db_path=config.db_path,
        dry_run=config.dry_run,
    )

    # Solve menu
    menu_sol = solve_menu(
        kappa_L=kappa_L,
        kappa_H=kappa_H,
        f_L=0.5,
        f_H=0.5,
        params=params,
        unconstrained_m=False,
    )

    trials_per_type = config.num_trials // 2
    results: List[SimulationTrialResult] = []
    trial_id = 0

    # Type L trials
    for _ in range(trials_per_type):
        trial_id += 1
        res_L = run_llm_trial(
            trial_id=trial_id,
            kappa=kappa_L,
            policy_regime="screening",
            ask_rate=menu_sol.a_L,
            prescribed_m=menu_sol.m_L,
            params=params,
            client=client,
            rng=rng,
        )
        results.append(res_L)

    # Type H trials
    for _ in range(trials_per_type):
        trial_id += 1
        res_H = run_llm_trial(
            trial_id=trial_id,
            kappa=kappa_H,
            policy_regime="screening",
            ask_rate=menu_sol.a_H,
            prescribed_m=menu_sol.m_H,
            params=params,
            client=client,
            rng=rng,
        )
        results.append(res_H)

    df = pd.DataFrame([vars(r) for r in results])
    return df
