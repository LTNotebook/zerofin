"""LLM provider factory — returns the right LangChain model based on config.

The rest of the codebase calls get_llm() and doesn't care whether it's
talking to Groq, DeepSeek, or OpenRouter behind the scenes. Switching
providers is a config change, not a code change.

Adding a new provider:
    1. Add its API key to config.py and .env
    2. Add an entry to PROVIDERS below
    That's it. No new if/else branches needed.

Provider wrappers:
    - "openai": ChatOpenAI — for OpenAI-compatible APIs (Groq, OpenRouter, DeepSeek direct)
    - "deepseek": ChatDeepSeek — DeepSeek's dedicated LangChain package
    - "anthropic": ChatAnthropic — for Anthropic-compatible APIs (MiniMax via OpenRouter)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from zerofin.config import settings

logger = logging.getLogger(__name__)

ProviderName = Literal["deepseek", "groq", "openrouter", "openrouter_anthropic"]


@dataclass(frozen=True)
class ProviderConfig:
    """Everything needed to connect to an LLM provider."""

    default_model: str
    base_url: str | None = None
    # "openai", "deepseek", or "anthropic"
    wrapper: str = "openai"


# Registry — one entry per provider. Add new providers here.
PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        default_model="deepseek-chat",
        wrapper="deepseek",
    ),
    "groq": ProviderConfig(
        default_model="meta-llama/llama-4-scout-17b-16e-instruct",
        base_url="https://api.groq.com/openai/v1",
    ),
    "openrouter": ProviderConfig(
        default_model="deepseek/deepseek-chat",
        base_url="https://openrouter.ai/api/v1",
    ),
    # OpenRouter's Anthropic-compatible endpoint.
    # Use this for models that work better with Anthropic message format
    # (e.g., MiniMax M2.7).
    "openrouter_anthropic": ProviderConfig(
        default_model="minimax/minimax-m2.7",
        base_url="https://openrouter.ai/api",
        wrapper="anthropic",
    ),
}

# Maps provider name → settings attribute for the API key
API_KEY_MAP: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openrouter_anthropic": "OPENROUTER_API_KEY",
}


def get_available_providers() -> list[str]:
    """Return provider names that have an API key configured."""
    available = []
    for name in PROVIDERS:
        key_attr = API_KEY_MAP[name]
        if getattr(settings, key_attr, ""):
            available.append(name)
    return available


def get_llm(
    provider: ProviderName | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> BaseChatModel:
    """Return a LangChain chat model.

    Args:
        provider: Override the config provider. If None, uses settings.LLM_PROVIDER.
        model: Override the model name. If None, uses settings.LLM_MODEL or the
            provider's default.
        temperature: Controls randomness. 0.0 = deterministic.

    Returns:
        A LangChain chat model ready to use with .invoke(), .batch(),
        or .with_structured_output().
    """
    provider_name = (provider or settings.LLM_PROVIDER).lower()

    if provider_name not in PROVIDERS:
        available = list(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: '{provider_name}'. Must be one of: {available}"
        )

    config = PROVIDERS[provider_name]
    model_name = model or settings.LLM_MODEL or config.default_model

    # Get the API key from settings
    key_attr = API_KEY_MAP[provider_name]
    api_key = getattr(settings, key_attr, "")
    if not api_key:
        raise ValueError(
            f"No API key for provider '{provider_name}'. "
            f"Set {key_attr} in your .env file."
        )

    logger.info("LLM: provider=%s, model=%s", provider_name, model_name)

    if config.wrapper == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if config.wrapper == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            base_url=config.base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # OpenAI-compatible wrapper (groq, openrouter, and any future providers)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        base_url=config.base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
