"""Tests for the entity mention identification pipeline.

No live API or database calls — tests pure logic only.

Run with: pytest tests/test_mentions.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zerofin.ai.mentions import (
    MentionResult,
    find_mentions,
    format_entity_list,
    validate_mention_ids,
)

# =====================================================================
# format_entity_list tests
# =====================================================================


class TestFormatEntityList:
    """Tests for entity list formatting."""

    def test_formats_name_and_id(self) -> None:
        entities = [
            {"id": "NVDA", "name": "NVIDIA Corporation", "label": "Company"},
        ]
        result = format_entity_list(entities)
        assert "NVIDIA Corporation (NVDA)" in result

    def test_multiple_entities(self) -> None:
        entities = [
            {"id": "NVDA", "name": "NVIDIA Corporation", "label": "Company"},
            {"id": "AAPL", "name": "Apple Inc.", "label": "Company"},
        ]
        result = format_entity_list(entities)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_empty_list(self) -> None:
        result = format_entity_list([])
        assert result == ""


# =====================================================================
# validate_mention_ids tests
# =====================================================================


class TestValidateMentionIds:
    """Tests for ID validation against entity list."""

    def test_all_valid_ids_pass(self) -> None:
        result = MentionResult(mentioned_ids=["NVDA", "AAPL", "TSLA"])
        valid_ids = {"NVDA", "AAPL", "TSLA", "MSFT"}
        validated = validate_mention_ids(result, valid_ids)
        assert len(validated.mentioned_ids) == 3

    def test_invalid_ids_removed(self) -> None:
        result = MentionResult(mentioned_ids=["NVDA", "FAKE", "AAPL"])
        valid_ids = {"NVDA", "AAPL"}
        validated = validate_mention_ids(result, valid_ids)
        assert validated.mentioned_ids == ["NVDA", "AAPL"]

    def test_all_invalid_returns_empty(self) -> None:
        result = MentionResult(mentioned_ids=["FAKE1", "FAKE2"])
        valid_ids = {"NVDA", "AAPL"}
        validated = validate_mention_ids(result, valid_ids)
        assert validated.mentioned_ids == []

    def test_empty_input_returns_empty(self) -> None:
        result = MentionResult(mentioned_ids=[])
        valid_ids = {"NVDA", "AAPL"}
        validated = validate_mention_ids(result, valid_ids)
        assert validated.mentioned_ids == []

    def test_format_error_id_removed(self) -> None:
        """IDs returned in wrong format get filtered out."""
        result = MentionResult(
            mentioned_ids=["NVDA", "WTI Crude Oil Futures (CL=F)", "CL=F"]
        )
        valid_ids = {"NVDA", "CL=F"}
        validated = validate_mention_ids(result, valid_ids)
        assert validated.mentioned_ids == ["NVDA", "CL=F"]


# =====================================================================
# MentionResult model tests
# =====================================================================


class TestMentionResult:
    """Tests for the MentionResult Pydantic model."""

    def test_empty_list_valid(self) -> None:
        result = MentionResult(mentioned_ids=[])
        assert result.mentioned_ids == []

    def test_list_of_strings_valid(self) -> None:
        result = MentionResult(mentioned_ids=["NVDA", "CL=F", "^GSPC"])
        assert len(result.mentioned_ids) == 3

    def test_duplicate_ids_preserved(self) -> None:
        """Model doesn't deduplicate — that's the caller's job."""
        result = MentionResult(mentioned_ids=["NVDA", "NVDA"])
        assert len(result.mentioned_ids) == 2


# =====================================================================
# find_mentions tests (with mocked chain)
# =====================================================================


class TestFindMentions:
    """Tests for find_mentions with a mocked LLM chain."""

    def _mock_chain(self, mentioned_ids: list[str]) -> MagicMock:
        chain = MagicMock()
        chain.invoke.return_value = MentionResult(
            mentioned_ids=mentioned_ids,
        )
        return chain

    def test_returns_llm_result(self) -> None:
        chain = self._mock_chain(["NVDA", "AAPL"])
        result = find_mentions("test article", "entity list", chain=chain)
        assert result.mentioned_ids == ["NVDA", "AAPL"]

    def test_returns_empty_on_error(self) -> None:
        chain = MagicMock()
        chain.invoke.side_effect = Exception("API timeout")
        result = find_mentions("test article", "entity list", chain=chain)
        assert result.mentioned_ids == []

    def test_validates_ids_when_valid_set_provided(self) -> None:
        chain = self._mock_chain(["NVDA", "FAKE", "AAPL"])
        result = find_mentions(
            "test article",
            "entity list",
            chain=chain,
            valid_ids={"NVDA", "AAPL"},
        )
        assert result.mentioned_ids == ["NVDA", "AAPL"]

    def test_skips_validation_when_no_valid_set(self) -> None:
        chain = self._mock_chain(["NVDA", "FAKE"])
        result = find_mentions("test article", "entity list", chain=chain)
        # No validation — both IDs pass through
        assert result.mentioned_ids == ["NVDA", "FAKE"]

    def test_passes_correct_args_to_chain(self) -> None:
        chain = self._mock_chain([])
        find_mentions("my article text", "my entity list", chain=chain)
        chain.invoke.assert_called_once_with({
            "entity_list": "my entity list",
            "article_text": "my article text",
        })
