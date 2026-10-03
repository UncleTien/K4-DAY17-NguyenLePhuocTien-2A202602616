from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider names and common aliases."""

    provider = value.strip().lower()

    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "open-router": "openrouter",
        "open_router": "openrouter",
        "openai-compatible": "custom",
    }

    provider = aliases.get(provider, provider)

    supported = {
        "openai",
        "custom",
        "gemini",
        "anthropic",
        "ollama",
        "openrouter",
    }

    if provider not in supported:
        raise ValueError(
            f"Unsupported provider: {value!r}. "
            f"Supported providers: {', '.join(sorted(supported))}"
        )

    return provider


def build_chat_model(config: ProviderConfig):
    """Build a LangChain chat model for the selected provider."""

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("CUSTOM_BASE_URL is required for custom provider.")

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "not-required",
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }

        if config.base_url:
            kwargs["base_url"] = config.base_url

        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise ImportError(
                "OpenRouter provider requires langchain-openrouter."
            ) from exc

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")