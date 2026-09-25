#!/usr/bin/env python3
"""
underspec_sim.experiments_llm.run_all_llm_experiments:
CLI orchestrator for LLM-driven experiments.
Runs simulated users interacting with the assistant under Model I and Model II.
Logs requests to SQLite runs.db and outputs summary CSVs and charts.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from underspec_sim.core.params import ModelParams
from underspec_sim.experiments_llm.config import LLMExperimentConfig
from underspec_sim.experiments_llm.run_experiment_pooling import run_pooling_llm_experiment
from underspec_sim.experiments_llm.run_experiment_screening import run_screening_llm_experiment


def main():
    parser = argparse.ArgumentParser(description="Run LLM-driven simulated user experiments.")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model name")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run with deterministic stub (free)")
    parser.add_argument("--live", action="store_true", help="Run with live Anthropic API (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--num-trials", type=int, default=24, help="Number of simulated user trials")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    is_dry_run = False if args.live else args.dry_run

    config = LLMExperimentConfig(
        model=args.model,
        dry_run=is_dry_run,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
    )
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"STARTING LLM EXPERIMENTS (Mode: {'DRY RUN' if config.dry_run else 'LIVE ANTHROPIC API'})")
    print(f"Model: {config.model} | Trials: {config.num_trials} | SQLite DB: {config.db_path}")
    print("=" * 80)

    params = ModelParams(k=10.0, g=0.5, L=10.0, c_Q=2.0, V=100.0, mu_A=1.0, lambda_A=4.0)

    # 1. Run Pooling experiment
    print("\n[1/2] Running Model I (Pooling) LLM Trials...")
    df_pool = run_pooling_llm_experiment(config, params)
    pool_csv = os.path.join(config.output_dir, "llm_experiment_pooling.csv")
    df_pool.to_csv(pool_csv, index=False)
    print(f"  Completed {len(df_pool)} pooling trials. Saved to {pool_csv}")

    # 2. Run Screening experiment
    print("\n[2/2] Running Model II (Screening) LLM Trials...")
    df_screen = run_screening_llm_experiment(config, params)
    screen_csv = os.path.join(config.output_dir, "llm_experiment_screening.csv")
    df_screen.to_csv(screen_csv, index=False)
    print(f"  Completed {len(df_screen)} screening trials. Saved to {screen_csv}")

    # Combine for analysis
    df_all = pd.concat([df_pool, df_screen], ignore_index=True)

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Task success rate by regime
    success_rates = df_all.groupby(["policy_regime", "kappa"])["task_success"].mean().unstack(level=0)
    success_rates.plot(kind="bar", ax=ax1, color=["crimson", "forestgreen"], alpha=0.85)
    ax1.set_ylabel("Empirical Task Success Rate")
    ax1.set_xlabel("User Cost Type kappa")
    ax1.set_title("Realized Task Success: Pooling vs Screening")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Leader Payoff by regime
    payoffs = df_all.groupby(["policy_regime", "kappa"])["leader_payoff_realized"].mean().unstack(level=0)
    payoffs.plot(kind="bar", ax=ax2, color=["crimson", "forestgreen"], alpha=0.85)
    ax2.set_ylabel("Empirical Leader Payoff")
    ax2.set_xlabel("User Cost Type kappa")
    ax2.set_title("Realized Leader Payoff: Pooling vs Screening")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(config.output_dir, "llm_experiment_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 80)
    print("LLM EXPERIMENT SUMMARY RESULTS:")
    print("-" * 80)
    summary_table = df_all.groupby("policy_regime").agg(
        total_trials=("trial_id", "count"),
        avg_realized_m=("realized_m", "mean"),
        avg_questions_asked=("questions_asked", "mean"),
        task_success_rate=("task_success", "mean"),
        avg_user_utility=("user_utility_realized", "mean"),
        avg_leader_payoff=("leader_payoff_realized", "mean"),
    ).reset_index()
    print(summary_table.to_string(index=False))
    print("=" * 80)
    print(f"Chart saved to: {plot_path}")
    print(f"All LLM requests audited in: {config.db_path}\n")


if __name__ == "__main__":
    main()
