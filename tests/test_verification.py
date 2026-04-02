"""Tests for the LLM verification pipeline models and routing logic.

No live API calls — tests pure logic only (validators, routing decisions).

Run with: pytest tests/test_verification.py
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zerofin.ai.verification import VerificationResult, needs_second_pass

# =====================================================================
# VerificationResult validator tests
# =====================================================================


class TestVerificationResultValidators:
    """Tests for the Pydantic validators on VerificationResult."""

    def test_lowercase_verdict(self) -> None:
        """Uppercase verdicts from DeepSeek get lowercased."""
        result = VerificationResult(
            mechanism="test",
            alternative_explanations="test",
            verdict="LIKELY_PLAUSIBLE",
            confidence=0.7,
            reasoning="test",
            relationship_category="sector_peer",
        )
        assert result.verdict == "likely_plausible"

    def test_lowercase_category(self) -> None:
        """Uppercase categories get lowercased."""
        result = VerificationResult(
            mechanism="test",
            alternative_explanations="test",
            verdict="uncertain",
            confidence=0.5,
            reasoning="test",
            relationship_category="MACRO_SENSITIVITY",
        )
        assert result.relationship_category == "macro_sensitivity"

    def test_confidence_clamped_above_one(self) -> None:
        """Confidence above 1.0 gets clamped to 1.0."""
        result = VerificationResult(
            mechanism="test",
            alternative_explanations="test",
            verdict="confirmed_plausible",
            confidence=1.5,
            reasoning="test",
            relationship_category="sector_peer",
        )
        assert result.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        """Confidence below 0.0 gets clamped to 0.0."""
        result = VerificationResult(
            mechanism="test",
            alternative_explanations="test",
            verdict="confirmed_spurious",
            confidence=-0.5,
            reasoning="test",
            relationship_category="none",
        )
        assert result.confidence == 0.0

    def test_confidence_normal_range_unchanged(self) -> None:
        """Confidence within 0-1 range passes through unchanged."""
        result = VerificationResult(
            mechanism="test",
            alternative_explanations="test",
            verdict="likely_plausible",
            confidence=0.72,
            reasoning="test",
            relationship_category="supply_chain",
        )
        assert result.confidence == 0.72

    def test_invalid_verdict_rejected(self) -> None:
        """Invalid verdict string raises ValidationError."""
        with pytest.raises(ValidationError):
            VerificationResult(
                mechanism="test",
                alternative_explanations="test",
                verdict="maybe_plausible",
                confidence=0.5,
                reasoning="test",
                relationship_category="sector_peer",
            )

    def test_invalid_category_rejected(self) -> None:
        """Invalid category string raises ValidationError."""
        with pytest.raises(ValidationError):
            VerificationResult(
                mechanism="test",
                alternative_explanations="test",
                verdict="uncertain",
                confidence=0.5,
                reasoning="test",
                relationship_category="invalid_category",
            )

    def test_all_valid_verdicts(self) -> None:
        """All five verdict values are accepted."""
        for verdict in [
            "confirmed_plausible",
            "likely_plausible",
            "uncertain",
            "likely_spurious",
            "confirmed_spurious",
        ]:
            result = VerificationResult(
                mechanism="test",
                alternative_explanations="test",
                verdict=verdict,
                confidence=0.5,
                reasoning="test",
                relationship_category="none",
            )
            assert result.verdict == verdict

    def test_all_valid_categories(self) -> None:
        """All seven category values are accepted."""
        for category in [
            "sector_peer",
            "supply_chain",
            "macro_sensitivity",
            "competition",
            "risk_factor",
            "etf_composition",
            "none",
        ]:
            result = VerificationResult(
                mechanism="test",
                alternative_explanations="test",
                verdict="uncertain",
                confidence=0.5,
                reasoning="test",
                relationship_category=category,
            )
            assert result.relationship_category == category


# =====================================================================
# needs_second_pass routing tests
# =====================================================================


def _make_result(verdict: str, confidence: float) -> VerificationResult:
    """Helper to create a VerificationResult with minimal fields."""
    return VerificationResult(
        mechanism="test",
        alternative_explanations="test",
        verdict=verdict,
        confidence=confidence,
        reasoning="test",
        relationship_category="none",
    )


class TestNeedsSecondPass:
    """Tests for the routing logic that decides what goes to Sonnet."""

    def test_confirmed_plausible_skips(self) -> None:
        """Confirmed plausible never goes to pass 2."""
        assert needs_second_pass(_make_result("confirmed_plausible", 0.95)) is False
        assert needs_second_pass(_make_result("confirmed_plausible", 0.50)) is False

    def test_confirmed_spurious_skips(self) -> None:
        """Confirmed spurious never goes to pass 2."""
        assert needs_second_pass(_make_result("confirmed_spurious", 0.15)) is False
        assert needs_second_pass(_make_result("confirmed_spurious", 0.90)) is False

    def test_uncertain_always_routes(self) -> None:
        """Uncertain always goes to pass 2."""
        assert needs_second_pass(_make_result("uncertain", 0.50)) is True
        assert needs_second_pass(_make_result("uncertain", 0.30)) is True
        assert needs_second_pass(_make_result("uncertain", 0.90)) is True

    def test_likely_plausible_below_threshold_routes(self) -> None:
        """Likely plausible below threshold goes to pass 2."""
        assert needs_second_pass(_make_result("likely_plausible", 0.60)) is True
        assert needs_second_pass(_make_result("likely_plausible", 0.50)) is True

    def test_likely_plausible_above_threshold_skips(self) -> None:
        """Likely plausible above threshold skips pass 2."""
        assert needs_second_pass(_make_result("likely_plausible", 0.70)) is False
        assert needs_second_pass(_make_result("likely_plausible", 0.90)) is False

    def test_likely_plausible_at_threshold_skips(self) -> None:
        """Likely plausible exactly at threshold skips pass 2."""
        assert needs_second_pass(_make_result("likely_plausible", 0.65)) is False

    def test_likely_spurious_below_threshold_routes(self) -> None:
        """Likely spurious below threshold goes to pass 2."""
        assert needs_second_pass(_make_result("likely_spurious", 0.25)) is True
        assert needs_second_pass(_make_result("likely_spurious", 0.50)) is True

    def test_likely_spurious_above_threshold_skips(self) -> None:
        """Likely spurious above threshold skips pass 2."""
        assert needs_second_pass(_make_result("likely_spurious", 0.70)) is False
        assert needs_second_pass(_make_result("likely_spurious", 0.90)) is False

    def test_likely_spurious_at_threshold_skips(self) -> None:
        """Likely spurious exactly at threshold skips pass 2."""
        assert needs_second_pass(_make_result("likely_spurious", 0.65)) is False
