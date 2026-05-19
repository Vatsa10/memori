"""Role-based model routing for the eval harness.

Each role maps to the most cost-effective model that meets the quality bar
for that job. Env overrides let you swap models without code changes.

Roles:
    generation         — chat replies in eval probes
    reasoning          — same as generation but enables thinking mode
    extraction         — Pydantic-structured fact extraction
    smart_ops_judge    — ADD/UPDATE/MERGE/DELETE/NOOP decisions
    summary_rollup     — turn/session/day/month rollups
    vision             — image / PDF figure ingestion
    eval_judge         — grades benchmark answers (kept constant across runs)
    eval_judge_premium — sanity-check sample with a stronger judge
"""

import os
from typing import Callable, Optional

from dotenv import load_dotenv

# Load .env from repo root if present
load_dotenv()


_DEFAULTS = {
    "generation":         "deepseek/deepseek-v4-flash",
    "reasoning":          "deepseek/deepseek-v4-flash",
    "extraction":         "deepseek/deepseek-v4-flash",
    "smart_ops_judge":    "deepseek/deepseek-v4-flash",
    "summary_rollup":     "gemini/gemini-2.5-flash-lite",
    "vision":             "gemini/gemini-2.5-flash",
    "eval_judge":         "openai/gpt-5.4-nano",
    "eval_judge_premium": "openai/gpt-5.4",
}


def get_model(role: str) -> str:
    """Return the model id for a given role. Env override: EVAL_MODEL_<ROLE>."""
    if role not in _DEFAULTS:
        raise ValueError(f"unknown role {role!r}; known: {sorted(_DEFAULTS)}")
    env_key = f"EVAL_MODEL_{role.upper()}"
    return os.environ.get(env_key) or _DEFAULTS[role]


def required_keys_for(model: str) -> list[str]:
    if model.startswith("openai/"):
        return ["OPENAI_API_KEY"]
    if model.startswith("gemini/"):
        return ["GEMINI_API_KEY"]
    if model.startswith("deepseek/"):
        return ["DEEPSEEK_API_KEY"]
    return []


def assert_keys_present(roles: list[str]) -> None:
    """Raise if any role's provider is missing its API key."""
    missing = set()
    for role in roles:
        for key in required_keys_for(get_model(role)):
            if not os.environ.get(key):
                missing.add(key)
    if missing:
        raise RuntimeError(
            f"Missing API keys for eval roles: {sorted(missing)}. "
            "Set them in .env or environment."
        )


def get_llm_fn(*, with_reasoning: bool = False) -> Callable:
    """Return an async llm_fn(model, messages, **kwargs) -> str.

    If with_reasoning=True, forwards `reasoning_effort` to the provider when
    supported (DeepSeek V4 thinking mode).
    """
    from litellm import acompletion

    async def _call(model: str, messages: list[dict], **kwargs) -> str:
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", 0.1),
        }
        if with_reasoning and model.startswith("deepseek/"):
            params["reasoning_effort"] = kwargs.pop("reasoning_effort", "medium")
        params.update(kwargs)
        response = await acompletion(**params)
        return response.choices[0].message.content or ""

    return _call


def summary() -> dict[str, str]:
    """Return the active model assignment for logging/manifests."""
    return {role: get_model(role) for role in _DEFAULTS}
