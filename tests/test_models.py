"""
Tests for Pydantic validation models.

These tests make sure our bouncers (DataPointCreate, EntityCreate) are
doing their job — letting good data through and rejecting bad data.

Run with:
    pytest tests/test_models.py

Or run all tests:
    pytest
"""

from __future__ import annotations

from decimal import Decimal

import pendulum
import pytest
from pydantic import ValidationError

from zerofin.models.entities import DataPointCreate, EntityCreate

# =====================================================================
# DataPointCreate tests — the bouncer for PostgreSQL
# =====================================================================


class TestDataPointCreate:
    """Tests for the DataPointCreate model."""

    def test_valid_data_point_accepted(self) -> None:
        """Good data should pass through without errors."""
        data = DataPointCreate(
            entity_type="asset",
            entity_id="NVDA",
            metric="close_price",
            value=Decimal("172.93"),
            timestamp=pendulum.now("UTC"),
            source="yfinance",
            unit="USD",
        )

        # If we got here without an error, the data was accepted
        assert data.entity_id == "NVDA"
        assert data.value == Decimal("172.93")
        assert data.source == "yfinance"

    def test_entity_id_converted_to_uppercase(self) -> None:
        """Lowercase IDs should be auto-converted to uppercase.

        This prevents 'nvda' and 'NVDA' from being treated as
        different entities — they're the same stock.
        """
        data = DataPointCreate(
            entity_type="asset",
            entity_id="nvda",  # lowercase on purpose
            metric="close_price",
            value=Decimal("172.93"),
            timestamp=pendulum.now("UTC"),
            source="yfinance",
        )

        assert data.entity_id == "NVDA"  # should be uppercase now

    def test_invalid_entity_type_rejected(self) -> None:
        """An entity type we don't recognize should be rejected.

        We only accept 'asset' and 'indicator'. Anything else
        means something is wrong with the data.
        """
        with pytest.raises(ValidationError) as error_info:
            DataPointCreate(
                entity_type="banana",  # not a valid type
                entity_id="NVDA",
                metric="close_price",
                value=Decimal("172.93"),
                timestamp=pendulum.now("UTC"),
                source="yfinance",
            )

        # Verify that the error is specifically on the entity_type field,
        # not some other field that happened to fail at the same time.
        errors = error_info.value.errors()
        assert any(e["loc"] == ("entity_type",) for e in errors)

    def test_invalid_source_rejected(self) -> None:
        """A data source we don't recognize should be rejected."""
        with pytest.raises(ValidationError) as error_info:
            DataPointCreate(
                entity_type="asset",
                entity_id="NVDA",
                metric="close_price",
                value=Decimal("172.93"),
                timestamp=pendulum.now("UTC"),
                source="made_up_source",  # not in our valid sources list
            )

        errors = error_info.value.errors()
        assert any(e["loc"] == ("source",) for e in errors)

    def test_empty_entity_id_rejected(self) -> None:
        """An empty entity ID should be rejected — we need to know WHAT this data is for."""
        with pytest.raises(ValidationError) as error_info:
            DataPointCreate(
                entity_type="asset",
                entity_id="",  # empty — not allowed
                metric="close_price",
                value=Decimal("100.00"),
                timestamp=pendulum.now("UTC"),
                source="yfinance",
            )

        errors = error_info.value.errors()
        assert any(e["loc"] == ("entity_id",) for e in errors)

    def test_empty_metric_rejected(self) -> None:
        """An empty metric should be rejected — we need to know WHAT was measured."""
        with pytest.raises(ValidationError) as error_info:
            DataPointCreate(
                entity_type="asset",
                entity_id="NVDA",
                metric="",  # empty — not allowed
                value=Decimal("100.00"),
                timestamp=pendulum.now("UTC"),
                source="yfinance",
            )

        errors = error_info.value.errors()
        assert any(e["loc"] == ("metric",) for e in errors)

    def test_revision_fields_work(self) -> None:
        """Data revisions should be trackable.

        Economic data gets revised all the time — CPI might be
        reported as 2.4% then corrected to 2.5% a month later.
        """
        data = DataPointCreate(
            entity_type="indicator",
            entity_id="CPI",
            metric="value",
            value=Decimal("2.5"),
            timestamp=pendulum.now("UTC"),
            source="fred",
            is_revised=True,
            revision_of=42,  # ID of the original data point
        )

        assert data.is_revised is True
        assert data.revision_of == 42

    def test_unit_is_optional(self) -> None:
        """Unit can be left out — some metrics don't have units."""
        data = DataPointCreate(
            entity_type="indicator",
            entity_id="UNEMPLOYMENT",
            metric="rate",
            value=Decimal("3.7"),
            timestamp=pendulum.now("UTC"),
            source="fred",
            # no unit — that's fine
        )

        assert data.unit is None


# =====================================================================
# EntityCreate tests — the bouncer for Neo4j
# =====================================================================


class TestEntityCreate:
    """Tests for the EntityCreate model."""

    def test_valid_entity_accepted(self) -> None:
        """Good entity data should pass through without errors."""
        entity = EntityCreate(
            id="NVDA",
            label="Company",
            name="NVIDIA Corporation",
            description="Semiconductor company, GPU maker",
        )

        assert entity.id == "NVDA"
        assert entity.label == "Company"
        assert entity.name == "NVIDIA Corporation"

    def test_id_converted_to_uppercase(self) -> None:
        """Lowercase IDs should be auto-converted to uppercase."""
        entity = EntityCreate(
            id="nvda",  # lowercase on purpose
            label="Company",
            name="NVIDIA Corporation",
        )

        assert entity.id == "NVDA"

    def test_invalid_label_rejected(self) -> None:
        """A label that's not one of our 12 entity types should be rejected."""
        with pytest.raises(ValidationError) as error_info:
            EntityCreate(
                id="NVDA",
                label="Spaceship",  # not a valid entity type
                name="NVIDIA Corporation",
            )

        errors = error_info.value.errors()
        assert any(e["loc"] == ("label",) for e in errors)

    def test_all_12_labels_accepted(self) -> None:
        """Every one of our 12 entity types should be accepted."""
        valid_labels = [
            "Asset",
            "Indicator",
            "Sector",
            "Event",
            "Company",
            "Index",
            "Commodity",
            "Currency",
            "Country",
            "CentralBank",
            "GovernmentBody",
            "Person",
        ]

        for label in valid_labels:
            entity = EntityCreate(
                id=f"TEST_{label.upper()}",
                label=label,
                name=f"Test {label}",
            )
            assert entity.label == label

    def test_empty_id_rejected(self) -> None:
        """An empty ID should be rejected."""
        with pytest.raises(ValidationError) as error_info:
            EntityCreate(
                id="",  # empty — not allowed
                label="Company",
                name="NVIDIA Corporation",
            )

        errors = error_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)

    def test_empty_name_rejected(self) -> None:
        """An empty name should be rejected."""
        with pytest.raises(ValidationError) as error_info:
            EntityCreate(
                id="NVDA",
                label="Company",
                name="",  # empty — not allowed
            )

        errors = error_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_metadata_is_optional(self) -> None:
        """Metadata can be left out — not every entity needs extra info."""
        entity = EntityCreate(
            id="NVDA",
            label="Company",
            name="NVIDIA Corporation",
            # no metadata — that's fine
        )

        assert entity.metadata is None

    def test_to_graph_properties_basic(self) -> None:
        """to_graph_properties() should return a clean dict for Neo4j."""
        entity = EntityCreate(
            id="NVDA",
            label="Company",
            name="NVIDIA Corporation",
            description="GPU maker",
        )

        props = entity.to_graph_properties()

        assert props["id"] == "NVDA"
        assert props["name"] == "NVIDIA Corporation"
        assert props["description"] == "GPU maker"

    def test_to_graph_properties_merges_metadata(self) -> None:
        """Metadata fields should be merged into top-level properties.

        This way Neo4j can query on metadata fields directly,
        like MATCH (n:Company {sector: 'Technology'}).
        """
        entity = EntityCreate(
            id="NVDA",
            label="Company",
            name="NVIDIA Corporation",
            metadata={"sector": "Technology", "subtype": "stock"},
        )

        props = entity.to_graph_properties()

        # Metadata should be merged into top-level
        assert props["sector"] == "Technology"
        assert props["subtype"] == "stock"
        # Label is NOT in properties — it's the node label in Neo4j
        assert "label" not in props
