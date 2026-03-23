"""
Pydantic models — data validation for everything entering the system.

These models act as bouncers at the door. Before any data goes into
PostgreSQL or Neo4j, it passes through one of these models. If the
data is bad (missing fields, wrong types, impossible values), the
model rejects it with a clear error message.

Two models for Phase 1:
- DataPointCreate: validates numbers going into PostgreSQL (prices, indicators)
- EntityCreate: validates entities going into Neo4j (companies, sectors, etc.)
"""

from __future__ import annotations

from decimal import Decimal

import pendulum
from pydantic import BaseModel, Field, field_validator

# The 12 entity types from the FinDKG ontology — same list as in graph.py.
# If you add a type here, add it in graph.py too.
VALID_ENTITY_TYPES = [
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

# What kind of thing is the data point about?
# "asset" = stocks, ETFs, commodities with prices
# "indicator" = economic measurements like CPI, GDP, unemployment
VALID_DATA_POINT_TYPES = ["asset", "indicator"]

# Where did the data come from? We track this so we know
# which source to blame if something looks wrong.
VALID_SOURCES = ["yfinance", "fred", "newsapi", "rss", "manual", "test_script"]


class DataPointCreate(BaseModel):
    """Validates a single data point before it goes into PostgreSQL.

    A data point is any number with a date — a stock price, an economic
    indicator reading, a volume count, etc. This model makes sure the
    data looks right before we store it.

    Example usage:
        data = DataPointCreate(
            entity_type="asset",
            entity_id="NVDA",
            metric="close_price",
            value=Decimal("172.93"),
            timestamp=pendulum.now("UTC"),
            source="yfinance",
            unit="USD",
        )
        # If any field is invalid, Pydantic raises a ValidationError
        # with a clear message saying what's wrong.
    """

    # What kind of thing is this? (asset, indicator)
    entity_type: str = Field(
        description="Category: 'asset' for stocks/commodities, 'indicator' for economic data"
    )

    # Which specific thing? (NVDA, CPI, GOLD)
    entity_id: str = Field(
        min_length=1,
        description="Unique identifier like a ticker symbol (NVDA) or indicator code (CPI)",
    )

    # What measurement is this? (close_price, volume, value, yoy_change)
    metric: str = Field(
        min_length=1,
        description="What's being measured: close_price, volume, value, etc.",
    )

    # The actual number
    value: Decimal = Field(
        description="The numeric value of this data point",
    )

    # What unit is the value in? (USD, percent, index_points)
    # Optional because some metrics don't have units
    unit: str | None = Field(
        default=None,
        description="Unit of measurement: USD, percent, index_points, etc.",
    )

    # When is this data FROM? (not when we collected it — when it happened)
    # For a stock price, this is the market close time.
    # For CPI, this is the release date.
    timestamp: pendulum.DateTime = Field(
        description="When this data point is from (not when we collected it)",
    )

    # Where did we get this data?
    source: str = Field(
        description="Data source: yfinance, fred, manual, etc.",
    )

    # Is this a correction of a previously reported number?
    is_revised: bool = Field(
        default=False,
        description="True if this replaces a previously reported value",
    )

    # If revised, which data point does this replace?
    revision_of: int | None = Field(
        default=None,
        description="ID of the original data point this revises (if applicable)",
    )

    # --- Validators ---
    # These run automatically when you create a DataPointCreate.
    # If any check fails, Pydantic raises an error instead of
    # letting bad data through.

    @field_validator("entity_type")
    @classmethod
    def entity_type_must_be_valid(cls, value: str) -> str:
        """Make sure entity_type is one we recognize."""
        if value not in VALID_DATA_POINT_TYPES:
            raise ValueError(f"entity_type must be one of {VALID_DATA_POINT_TYPES}, got '{value}'")
        return value

    @field_validator("source")
    @classmethod
    def source_must_be_valid(cls, value: str) -> str:
        """Make sure the data source is one we recognize."""
        if value not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got '{value}'")
        return value

    @field_validator("entity_id")
    @classmethod
    def entity_id_must_be_uppercase(cls, value: str) -> str:
        """Normalize entity IDs to uppercase to prevent duplicates.

        Without this, 'nvda' and 'NVDA' would be treated as different
        entities. We force uppercase so there's only one way to refer
        to each entity.
        """
        return value.upper()

    model_config = {
        "arbitrary_types_allowed": True,  # Allows pendulum.DateTime
    }


class EntityCreate(BaseModel):
    """Validates an entity before it goes into Neo4j as a node.

    An entity is anything that exists in the financial world — a company,
    a sector, a commodity, a country, etc. This model makes sure we have
    the minimum required info before creating a graph node.

    Example usage:
        entity = EntityCreate(
            id="NVDA",
            label="Company",
            name="NVIDIA Corporation",
            description="Semiconductor company, GPU maker",
            metadata={"sector": "Technology", "subtype": "stock"},
        )
    """

    # Unique identifier — ticker symbol, indicator code, or slug
    id: str = Field(
        min_length=1,
        description="Unique ID: ticker (NVDA), code (CPI), or slug (energy_sector)",
    )

    # Which of the 12 entity types is this?
    label: str = Field(
        description="Node label in Neo4j — must be one of the 12 FinDKG entity types",
    )

    # Human-readable name
    name: str = Field(
        min_length=1,
        description="Display name: 'NVIDIA Corporation', 'Consumer Price Index', etc.",
    )

    # Optional longer description
    description: str | None = Field(
        default=None,
        description="What this entity is, in plain English",
    )

    # Flexible extra info — different entity types need different fields.
    # A Company might have sector and market_cap.
    # An Indicator might have unit and release_schedule.
    # Rather than making separate models for each type, we use a flexible dict.
    metadata: dict | None = Field(
        default=None,
        description="Extra properties specific to this entity type",
    )

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, value: str) -> str:
        """Make sure the label is one of our 12 entity types."""
        if value not in VALID_ENTITY_TYPES:
            raise ValueError(f"label must be one of {VALID_ENTITY_TYPES}, got '{value}'")
        return value

    @field_validator("id")
    @classmethod
    def id_must_be_uppercase(cls, value: str) -> str:
        """Normalize IDs to uppercase to prevent duplicates."""
        return value.upper()

    def to_graph_properties(self) -> dict:
        """Convert this model into a dict ready for Neo4j.

        Neo4j nodes store flat properties, so we merge the metadata
        dict into the top-level properties. This way you can query
        on any metadata field directly in Cypher.
        """
        # Start with the required fields
        properties = {
            "id": self.id,
            "name": self.name,
        }

        # Add description if provided
        if self.description:
            properties["description"] = self.description

        # Merge metadata into top-level properties
        # So {"sector": "Technology"} becomes a direct property on the node
        if self.metadata:
            properties.update(self.metadata)

        return properties
