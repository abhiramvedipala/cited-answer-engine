"""Thin wrapper over the Anthropic SDK that logs tokens, latency and cost per call."""

import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import anthropic

# USD per million tokens (input, output). Haiku 5.5 rate is for prompts <= 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
LOG = Path(os.getenv("LLM_LOG", "logs/llm_calls.jsonl"))


def model_name() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1e6


_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def stream(system: str, user: str, purpose: str, max_tokens: int = 1024) -> Iterator[str | dict]:
    """Yields text deltas, then one final dict with usage, latency and cost."""
    model = model_name()
    t0 = time.perf_counter()
    with client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "disabled"},  # short grounded answers; keeps latency and cost low
        messages=[{"role": "user", "content": user}],
    ) as s:
        yield from s.text_stream
        final = s.get_final_message()
    u = final.usage
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        "model": model,
        "purpose": purpose,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "latency_s": round(time.perf_counter() - t0, 3),
        "cost_usd": cost(model, u.input_tokens, u.output_tokens),
        "stop_reason": final.stop_reason,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    yield rec


def complete(system: str, user: str, purpose: str, max_tokens: int = 1024) -> tuple[str, dict]:
    parts, meta = [], {}
    for x in stream(system, user, purpose, max_tokens):
        if isinstance(x, dict):
            meta = x
        else:
            parts.append(x)
    return "".join(parts), meta
