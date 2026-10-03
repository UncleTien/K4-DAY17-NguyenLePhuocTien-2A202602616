from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_config(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
) -> ProviderConfig:
    provider = normalize_provider(provider)

    api_key = None
    base_url = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_API_KEY"
        )

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")

    elif provider == "ollama":
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and build shared lab config."""

    root = (
        base_dir
        or Path(__file__).resolve().parent.parent
    ).resolve()

    load_dotenv(root / ".env")

    data_dir = root / "data"
    state_dir = root / "state"

    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(
        os.getenv("LLM_PROVIDER", "ollama")
    )

    default_models = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "qwen2.5:3b",
        "openrouter": "openai/gpt-4o-mini",
    }

    model_name = os.getenv(
        "LLM_MODEL",
        default_models[provider],
    )

    model = _provider_config(
        provider=provider,
        model_name=model_name,
        temperature=float(
            os.getenv("LLM_TEMPERATURE", "0")
        ),
    )

    judge_provider = normalize_provider(
        os.getenv("JUDGE_PROVIDER", provider)
    )

    judge_model_name = os.getenv(
        "JUDGE_MODEL",
        default_models[judge_provider],
    )

    judge_model = _provider_config(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=int(
            os.getenv(
                "COMPACT_THRESHOLD_TOKENS",
                "1200",
            )
        ),
        compact_keep_messages=int(
            os.getenv(
                "COMPACT_KEEP_MESSAGES",
                "6",
            )
        ),
        model=model,
        judge_model=judge_model,
    )