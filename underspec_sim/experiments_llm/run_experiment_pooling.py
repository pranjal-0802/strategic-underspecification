"""
underspec_sim.experiments_llm.run_experiment_pooling:
Executes LLM-driven experiment under Model I Pooling policy (a^{SE}).
Simulated users of heterogeneous kappa types best-respond to the single ask rate.
"""

from typing import List
import numpy as np
import pandas as pd
from underspec_sim.core.params import ModelParams
from underspec_sim.core.best_response import user_best_response
from underspec_sim.model1_pooling.solver import solve_pooling_equilibrium
from underspec_sim.llm.llm_client import LLMClient
from underspec_sim.llm.simulated_user import run_llm_trial, SimulationTrialResult
from underspec_sim.experiments_llm.config import LLMExperimentConfig


def run_pooling_llm_experiment(
    config: LLMExperimentConfig,
    params: ModelParams,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    client = LLMClient(
        model=config.model,
        db_path=config.db_path,
        dry_run=config.dry_run,
    )

    # Solve pooling policy on population
    F_samples = np.linspace(0.2, 0.6, 100)
    pool_res = solve_pooling_equilibrium(F_samples, params=params)
    a_SE = pool_res.a_SE_grid

    # Simulated user types
    test_kappas = [0.25, 0.35, 0.55]  # Low, medium, high cost
    results: List[SimulationTrialResult] = []

    trial_counter = 0
    for kap in test_kappas:
        # Best response specification effort m*(kap; a_SE)
        m_star = float(user_best_response(kap, a_SE, params, clip=True))
        for _ in range(config.num_trials // len(test_kappas)):
            trial_counter += 1
            res = run_llm_trial(
                trial_id=trial_counter,
                kappa=kap,
                policy_regime="pooling",
                ask_rate=a_SE,
                prescribed_m=m_star,
                params=params,
                client=client,
                rng=rng,
            )
            results.append(res)

    df = pd.DataFrame([vars(r) for r in results])
    return df
