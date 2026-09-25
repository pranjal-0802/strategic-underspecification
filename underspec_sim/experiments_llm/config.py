"""
underspec_sim.experiments_llm.config: Experiment configuration for LLM-driven path.
"""

from pydantic import BaseModel, Field


class LLMExperimentConfig(BaseModel):
    model: str = Field(default="claude-sonnet-4-6", description="Anthropic model name")
    dry_run: bool = Field(default=True, description="Run with deterministic stub free of API cost")
    db_path: str = Field(default="runs.db", description="SQLite request log database path")
    output_dir: str = Field(default="outputs", description="Output directory for CSVs and plots")
    num_trials: int = Field(default=20, gt=0, description="Number of simulated user trials per condition")
    seed: int = Field(default=42, description="Random seed for reproducibility")
