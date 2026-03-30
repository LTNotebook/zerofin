"""Tests for the entity mention identification pipeline.

No live API or database calls — tests pure logic only.

Run with: pytest tests/test_mentions.py
"""

from __future__ import annotations

from zerofin.ai.mentions import (
    MentionResult,
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
