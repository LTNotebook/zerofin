"""Tests for the article extraction pipeline models and post-processing.

No live API calls — tests pure logic only (validators, dedup, validation).

Run with: pytest tests/test_extraction.py
"""

from __future__ import annotations

import pytest

from zerofin.ai.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    deduplicate_entities,
    deduplicate_relationships,
    validate_relationships,
)

# =====================================================================
# ExtractedEntity validator tests
# =====================================================================


class TestEntityTypeValidator:
    """Tests for entity type casing normalization."""

    def test_exact_match(self) -> None:
        entity = ExtractedEntity(
            text="Apple",
            reasoning="test",
            entity_type="Company",
            canonical_name="Apple Inc.",
        )
        assert entity.entity_type == "Company"

    def test_lowercase_normalized(self) -> None:
        entity = ExtractedEntity(
            text="Apple",
            reasoning="test",
            entity_type="company",
            canonical_name="Apple Inc.",
        )
        assert entity.entity_type == "Company"

    def test_uppercase_normalized(self) -> None:
        entity = ExtractedEntity(
            text="oil",
            reasoning="test",
            entity_type="COMMODITY",
            canonical_name="Crude Oil",
        )
        assert entity.entity_type == "Commodity"

    def test_centralbank_camelcase(self) -> None:
        entity = ExtractedEntity(
            text="the Fed",
            reasoning="test",
            entity_type="centralbank",
            canonical_name="Federal Reserve",
        )
        assert entity.entity_type == "CentralBank"

    def test_governmentbody_camelcase(self) -> None:
        entity = ExtractedEntity(
            text="SEC",
            reasoning="test",
            entity_type="governmentbody",
            canonical_name="SEC",
        )
        assert entity.entity_type == "GovernmentBody"


# =====================================================================
# ExtractedRelationship validator tests
# =====================================================================


class TestRelationshipConfidenceValidator:
    """Tests for confidence clamping and threshold enforcement."""

    def test_valid_confidence_passes(self) -> None:
        rel = ExtractedRelationship(
            subject="A",
            object="B",
            reasoning="test",
            relationship_type="CAUSES",
            confidence=0.85,
            evidence="test",
        )
        assert rel.confidence == 0.85

    def test_confidence_above_one_clamped(self) -> None:
        rel = ExtractedRelationship(
            subject="A",
            object="B",
            reasoning="test",
            relationship_type="CAUSES",
            confidence=1.5,
            evidence="test",
        )
        assert rel.confidence == 1.0

    def test_confidence_below_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="below the minimum threshold"):
            ExtractedRelationship(
                subject="A",
                object="B",
                reasoning="test",
                relationship_type="CAUSES",
                confidence=0.5,
                evidence="test",
            )

    def test_confidence_at_threshold_passes(self) -> None:
        rel = ExtractedRelationship(
            subject="A",
            object="B",
            reasoning="test",
            relationship_type="CAUSES",
            confidence=0.70,
            evidence="test",
        )
        assert rel.confidence == 0.70


class TestRelationshipTypeValidator:
    """Tests for relationship type normalization."""

    def test_uppercase_preserved(self) -> None:
        rel = ExtractedRelationship(
            subject="A",
            object="B",
            reasoning="test",
            relationship_type="CAUSES",
            confidence=0.9,
            evidence="test",
        )
        assert rel.relationship_type == "CAUSES"

    def test_lowercase_uppercased(self) -> None:
        rel = ExtractedRelationship(
            subject="A",
            object="B",
            reasoning="test",
            relationship_type="supplies_to",
            confidence=0.9,
            evidence="test",
        )
        assert rel.relationship_type == "SUPPLIES_TO"

    def test_spaces_converted_to_underscores(self) -> None:
        rel = ExtractedRelationship(
            subject="A",
            object="B",
            reasoning="test",
            relationship_type="supplies to",
            confidence=0.9,
            evidence="test",
        )
        assert rel.relationship_type == "SUPPLIES_TO"


# =====================================================================
# Deduplication tests
# =====================================================================


class TestDeduplicateEntities:
    """Tests for entity deduplication by canonical name."""

    def test_no_duplicates_unchanged(self) -> None:
        entities = [
            ExtractedEntity(
                text="oil", reasoning="test",
                entity_type="Commodity", canonical_name="Crude Oil",
            ),
            ExtractedEntity(
                text="gold", reasoning="test",
                entity_type="Commodity", canonical_name="Gold",
            ),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_duplicate_canonical_keeps_first(self) -> None:
        entities = [
            ExtractedEntity(
                text="oil prices", reasoning="test",
                entity_type="Commodity", canonical_name="Crude Oil",
            ),
            ExtractedEntity(
                text="crude oil", reasoning="test",
                entity_type="Commodity", canonical_name="Crude Oil",
            ),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1
        assert result[0].text == "oil prices"

    def test_case_insensitive_dedup(self) -> None:
        entities = [
            ExtractedEntity(
                text="Apple", reasoning="test",
                entity_type="Company", canonical_name="Apple Inc.",
            ),
            ExtractedEntity(
                text="apple", reasoning="test",
                entity_type="Company", canonical_name="apple inc.",
            ),
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1


class TestDeduplicateRelationships:
    """Tests for relationship deduplication by triple."""

    def test_no_duplicates_unchanged(self) -> None:
        rels = [
            ExtractedRelationship(
                subject="A", object="B", reasoning="test",
                relationship_type="CAUSES", confidence=0.9,
                evidence="test",
            ),
            ExtractedRelationship(
                subject="A", object="C", reasoning="test",
                relationship_type="CAUSES", confidence=0.9,
                evidence="test",
            ),
        ]
        result = deduplicate_relationships(rels)
        assert len(result) == 2

    def test_duplicate_triple_keeps_first(self) -> None:
        rels = [
            ExtractedRelationship(
                subject="A", object="B", reasoning="first",
                relationship_type="CAUSES", confidence=0.9,
                evidence="first",
            ),
            ExtractedRelationship(
                subject="A", object="B", reasoning="second",
                relationship_type="CAUSES", confidence=0.8,
                evidence="second",
            ),
        ]
        result = deduplicate_relationships(rels)
        assert len(result) == 1
        assert result[0].evidence == "first"


# =====================================================================
# Validation tests
# =====================================================================


class TestValidateRelationships:
    """Tests for post-processing validation rules."""

    def _make_entities(self) -> list[ExtractedEntity]:
        return [
            ExtractedEntity(
                text="NVIDIA", reasoning="test",
                entity_type="Company", canonical_name="NVIDIA Corporation",
            ),
            ExtractedEntity(
                text="oil", reasoning="test",
                entity_type="Commodity", canonical_name="Crude Oil",
            ),
            ExtractedEntity(
                text="TSMC", reasoning="test",
                entity_type="Company", canonical_name="TSMC",
            ),
        ]

    def test_valid_relationship_passes(self) -> None:
        entities = self._make_entities()
        rels = [
            ExtractedRelationship(
                subject="TSMC", object="NVIDIA Corporation",
                reasoning="test", relationship_type="SUPPLIES_TO",
                confidence=0.9, evidence="test",
            ),
        ]
        result = validate_relationships(rels, entities)
        assert len(result) == 1

    def test_supplies_to_commodity_rejected(self) -> None:
        entities = self._make_entities()
        rels = [
            ExtractedRelationship(
                subject="NVIDIA Corporation", object="Crude Oil",
                reasoning="test", relationship_type="SUPPLIES_TO",
                confidence=0.9, evidence="test",
            ),
        ]
        result = validate_relationships(rels, entities)
        assert len(result) == 0

    def test_nonexistent_subject_rejected(self) -> None:
        entities = self._make_entities()
        rels = [
            ExtractedRelationship(
                subject="Fake Company", object="NVIDIA Corporation",
                reasoning="test", relationship_type="CAUSES",
                confidence=0.9, evidence="test",
            ),
        ]
        result = validate_relationships(rels, entities)
        assert len(result) == 0

    def test_nonexistent_object_rejected(self) -> None:
        entities = self._make_entities()
        rels = [
            ExtractedRelationship(
                subject="NVIDIA Corporation", object="Nonexistent",
                reasoning="test", relationship_type="CAUSES",
                confidence=0.9, evidence="test",
            ),
        ]
        result = validate_relationships(rels, entities)
        assert len(result) == 0

    def test_non_supplies_to_commodity_allowed(self) -> None:
        """Other relationship types CAN have Commodity as object."""
        entities = self._make_entities()
        rels = [
            ExtractedRelationship(
                subject="NVIDIA Corporation", object="Crude Oil",
                reasoning="test", relationship_type="DISRUPTS",
                confidence=0.9, evidence="test",
            ),
        ]
        result = validate_relationships(rels, entities)
        assert len(result) == 1
