"""Tests for the LLM provider registry.

No live API calls — tests configuration logic with mocked constructors.

Run with: pytest tests/test_provider.py
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from zerofin.ai.provider import PROVIDERS, get_available_providers, get_llm


# =====================================================================
# Provider registry tests
# =====================================================================


class TestProviderRegistry:
    """Tests for the provider configuration."""

    def test_all_expected_providers_exist(self) -> None:
        """Registry contains all configured providers."""
        expected = {"deepseek", "groq", "openrouter", "openrouter_anthropic"}
        assert set(PROVIDERS.keys()) == expected

    def test_each_provider_has_default_model(self) -> None:
        """Every provider has a non-empty default model."""
        for name, config in PROVIDERS.items():
            assert config.default_model, f"{name} has no default model"


# =====================================================================
# get_llm validation tests
# =====================================================================


class TestGetLlmValidation:
    """Tests for get_llm error handling — no actual LLM calls."""

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm(provider="nonexistent_provider")

    def test_missing_api_key_raises(self) -> None:
        """Provider with empty API key raises ValueError."""
        with patch("zerofin.ai.provider.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "deepseek"
            mock_settings.LLM_MODEL = ""
            mock_settings.DEEPSEEK_API_KEY = ""
            with pytest.raises(ValueError, match="No API key"):
                get_llm(provider="deepseek")

    def test_provider_override_works(self) -> None:
        """Provider parameter overrides config setting."""
        with patch("zerofin.ai.provider.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "deepseek"
            mock_settings.LLM_MODEL = ""
            mock_settings.GROQ_API_KEY = ""
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                get_llm(provider="groq")


# =====================================================================
# get_available_providers tests
# =====================================================================


class TestGetAvailableProviders:
    """Tests for discovering which providers have keys configured."""

    def test_returns_providers_with_keys(self) -> None:
        """Only providers with non-empty API keys are returned."""
        with patch("zerofin.ai.provider.settings") as mock_settings:
            mock_settings.DEEPSEEK_API_KEY = "sk-test"
            mock_settings.GROQ_API_KEY = ""
            mock_settings.OPENROUTER_API_KEY = "sk-or-test"
            available = get_available_providers()
            assert "deepseek" in available
            assert "openrouter" in available
            assert "openrouter_anthropic" in available  # same key as openrouter
            assert "groq" not in available

    def test_no_keys_returns_empty(self) -> None:
        """No API keys configured returns empty list."""
        with patch("zerofin.ai.provider.settings") as mock_settings:
            mock_settings.DEEPSEEK_API_KEY = ""
            mock_settings.GROQ_API_KEY = ""
            mock_settings.OPENROUTER_API_KEY = ""
            assert get_available_providers() == []
