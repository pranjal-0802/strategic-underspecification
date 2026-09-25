"""
underspec_sim.llm.llm_client: LLM client with retry, SQLite request logging,
and deterministic dry-run stub.
- Reads ANTHROPIC_API_KEY from environment; never hardcoded.
- In dry-run mode, generates deterministic responses without requiring an API key.
- Logs all calls to SQLite database (runs.db).
"""

import os
import time
import json
import sqlite3
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_seconds: float
    is_dry_run: bool
    raw_response: Optional[Any] = None


class LLMClient:
    """
    Unified client for Anthropic API with SQLite audit logging and deterministic stub.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        db_path: str = "runs.db",
        dry_run: bool = False,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.model = model
        self.db_path = db_path
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.dry_run and not self.api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not found in environment. Switching to dry_run=True mode."
            )
            self.dry_run = True

        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite request log table in runs.db."""
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT,
                    user_prompt TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    is_dry_run INTEGER NOT NULL,
                    latency_seconds REAL NOT NULL,
                    metadata_json TEXT
                )
                """
            )
        conn.close()

    def _log_run(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        response_text: str,
        latency: float,
        is_dry_run: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert execution record into runs.db."""
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute(
                """
                INSERT INTO llm_runs (
                    timestamp, model, system_prompt, user_prompt,
                    response_text, is_dry_run, latency_seconds, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    self.model,
                    system_prompt or "",
                    user_prompt,
                    response_text,
                    1 if is_dry_run else 0,
                    latency,
                    json.dumps(metadata or {}),
                ),
            )
        conn.close()

    def _deterministic_stub(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate deterministic mocked responses for testing without API keys.
        Conditioned on metadata like kappa, declared attributes, role, etc.
        """
        meta = metadata or {}
        role = meta.get("role", "user")
        kappa = meta.get("kappa", 0.4)
        m_target = meta.get("m_target", 5)
        attributes = meta.get("attributes", {})

        if role == "simulated_user":
            # Generate a specification text describing m_target attributes
            spec_lines = [f"# Coding Task Specification (Type kappa={kappa:.2f})"]
            if kappa > 0.45:
                spec_lines.append("I need a function to process payments. Make it fast.")
            else:
                spec_lines.append(
                    f"I need a complete, production payment processing service with exactly {m_target} explicit requirements:"
                )
                for i in range(1, int(m_target) + 1):
                    attr_name = f"attribute_{i}"
                    val = attributes.get(attr_name, f"config_val_{i}")
                    spec_lines.append(f"- Requirement {i}: {attr_name} must be set to {val}.")
            return "\n".join(spec_lines)

        elif role == "assistant":
            # Assistant decides whether to ask clarifying questions or guess
            action = meta.get("action", "guess")
            unresolved = meta.get("unresolved_count", 2)
            if action == "ask":
                return (
                    f"Before implementing, I have {unresolved} clarifying questions regarding the edge cases "
                    f"and schema formatting."
                )
            else:
                return (
                    f"Understood. Implementing the solution directly with silent defaults for any unmentioned attributes."
                )

        # Fallback generic stub
        return f"[Deterministic Dry-Run Response for prompt: '{user_prompt[:40]}...']"

    def complete(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Call Anthropic LLM (or deterministic stub if dry_run=True), logging to SQLite.
        """
        t0 = time.time()

        if self.dry_run:
            time.sleep(0.01)  # small simulated latency
            text = self._deterministic_stub(user_prompt, system_prompt, metadata)
            latency = time.time() - t0
            self._log_run(system_prompt, user_prompt, text, latency, is_dry_run=True, metadata=metadata)
            return LLMResponse(
                text=text,
                model=self.model,
                latency_seconds=latency,
                is_dry_run=True,
            )

        # Real Anthropic API path
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required for non-dry-run calls. Install with pip install anthropic.")

        client = anthropic.Anthropic(api_key=self.api_key)

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                messages = [{"role": "user", "content": user_prompt}]
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                resp = client.messages.create(**kwargs)
                response_text = resp.content[0].text
                latency = time.time() - t0
                self._log_run(system_prompt, user_prompt, response_text, latency, is_dry_run=False, metadata=metadata)
                return LLMResponse(
                    text=response_text,
                    model=self.model,
                    latency_seconds=latency,
                    is_dry_run=False,
                    raw_response=resp,
                )
            except Exception as e:
                last_exc = e
                wait_time = self.backoff_factor ** attempt
                logger.warning(f"Anthropic API call failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Anthropic API call failed after {self.max_retries} attempts: {last_exc}")
