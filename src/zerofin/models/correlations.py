"""
Pydantic models for the correlation engine.

These validate correlation results before they go into Neo4j.
Same idea as DataPointCreate and EntityCreate — nothing gets
stored without passing through validation first.

Two models:
- CorrelationCandidate: one discovered correlation between two entities
- CorrelationRunSummary: stats from a full correlation engine run
"""

from __future__ import annotations

from typing import Any

import pendulum
from pydantic import BaseModel, Field, field_validator, model_validator

from zerofin.config import settings

# Which tier does this correlation fall into?
# These match the thresholds in config.py
TIER_STORE = "store"            # |r| >= 0.3 — weak, worth watching
TIER_ACTIONABLE = "actionable"  # |r| >= 0.5 — moderate, use in analysis
TIER_STRONG = "strong"          # |r| >= 0.7 — high-confidence
VALID_TIERS = [TIER_STORE, TIER_ACTIONABLE, TIER_STRONG]

# How was this correlation calculated?
VALID_METHODS = ["pearson", "spearman"]

# What kind of data was correlated?
VALID_ENTITY_TYPES = ["asset", "indicator"]


class CorrelationCandidate(BaseModel):
    """Validates a single correlation before it goes into Neo4j.

    A correlation candidate says "entity A and entity B moved together
    (or opposite) over a given time window, with this much statistical
    confidence." The correlation engine discovers these; the review
    queue decides which ones to promote to active relationships.

    Example:
        candidate = CorrelationCandidate(
            entity_a_type="asset",
            entity_a_id="NVDA",
            entity_b_type="asset",
            entity_b_id="AMD",
            lag_days=0,
            correlation=0.82,
            p_value=0.0001,
            method="pearson",
            observation_count=252,
            window_days=252,
            window_end=pendulum.now("UTC"),
        )
    """

    # --- The two entities being compared ---
    entity_a_type: str = Field(description="'asset' or 'indicator'")
    entity_a_id: str = Field(
        min_length=1, description="Ticker or indicator code (e.g. NVDA, DGS10)"
    )
    entity_b_type: str = Field(description="'asset' or 'indicator'")
    entity_b_id: str = Field(min_length=1, description="Ticker or indicator code")

    # --- The correlation result ---
    # Signed value: +0.82 means they move together, -0.75 means they move opposite
    correlation: float = Field(ge=-1.0, le=1.0, description="Pearson or Spearman r value")
    # How likely is this correlation to be random noise? Lower = more real.
    p_value: float = Field(ge=0.0, le=1.0, description="Statistical significance")
    # Which method produced this result
    method: str = Field(description="'pearson' or 'spearman'")

    # --- Context ---
    # Does entity A lead entity B by N days? 0 = same day.
    lag_days: int = Field(ge=0, description="Lead-lag in trading days")
    # How many overlapping data points were used
    observation_count: int = Field(gt=0, description="Number of overlapping data points")
    # Which window size produced this (21, 63, or 252)
    window_days: int = Field(gt=0, description="Rolling window size in trading days")
    # When the window ended (so we know how fresh this is)
    window_end: pendulum.DateTime = Field(description="End date of the analysis window")

    # --- Derived fields (set automatically by validators) ---
    # Absolute strength: 0.0 to 1.0 (ignores direction)
    strength: float = Field(default=0.0, description="Absolute correlation strength")
    # "positive" or "negative"
    direction: str = Field(default="", description="Correlation direction")
    # Which tier: store, actionable, or strong
    tier: str = Field(default="", description="Strength tier for filtering")

    # --- Validators ---

    @field_validator("entity_a_type", "entity_b_type")
    @classmethod
    def entity_type_must_be_valid(cls, value: str) -> str:
        if value not in VALID_ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {VALID_ENTITY_TYPES}, got '{value}'")
        return value

    @field_validator("entity_a_id", "entity_b_id")
    @classmethod
    def entity_id_must_be_uppercase(cls, value: str) -> str:
        return value.upper()

    @field_validator("method")
    @classmethod
    def method_must_be_valid(cls, value: str) -> str:
        if value not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{value}'")
        return value

    @model_validator(mode="after")
    def set_derived_fields(self) -> CorrelationCandidate:
        """Calculate strength, direction, and tier from the correlation value."""
        # Strength is just the absolute value
        self.strength = abs(self.correlation)

        # Direction from the sign
        self.direction = "positive" if self.correlation >= 0 else "negative"

        # Tier based on strength thresholds from config
        if self.strength >= settings.CORRELATION_TIER_STRONG:
            self.tier = TIER_STRONG
        elif self.strength >= settings.CORRELATION_TIER_ACTIONABLE:
            self.tier = TIER_ACTIONABLE
        else:
            self.tier = TIER_STORE

        return self

    @model_validator(mode="after")
    def entities_must_be_different(self) -> CorrelationCandidate:
        """Can't correlate something with itself."""
        if self.entity_a_id == self.entity_b_id:
            raise ValueError("entity_a_id and entity_b_id must be different")
        return self

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to a flat dict ready for Neo4j relationship properties.

        This is what gets stored on the CORRELATES_WITH edge between
        two entity nodes in the graph.
        """
        return {
            "strength": self.strength,
            "direction": self.direction,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "method": self.method,
            "tier": self.tier,
            "lag_days": self.lag_days,
            "observation_count": self.observation_count,
            "window_days": self.window_days,
            "window_end": self.window_end.to_iso8601_string(),
            "confidence": settings.CORRELATION_INITIAL_CONFIDENCE,
            "source": "statistical",
            "status": "candidate",
            "times_tested": 0,
            "times_confirmed": 0,
            "valid_from": pendulum.now("UTC").to_iso8601_string(),
            "valid_until": None,
        }

    model_config = {
        "arbitrary_types_allowed": True,
    }


class CorrelationRunSummary(BaseModel):
    """Stats from a full correlation engine run.

    Gives a quick picture of what happened — how many pairs were tested,
    how many survived filtering, how long it took.
    """

    total_pairs_tested: int = Field(ge=0, description="Total entity pairs checked")
    pairs_above_threshold: int = Field(ge=0, description="Pairs that passed min strength")
    pairs_surviving_fdr: int = Field(ge=0, description="Pairs that survived FDR correction")
    relationships_stored: int = Field(ge=0, description="Candidates written to Neo4j")
    window_days: int = Field(gt=0, description="Window size for this run")
    window_end: pendulum.DateTime = Field(description="End date of analysis window")
    duration_seconds: float = Field(ge=0.0, description="How long the run took")

    model_config = {
        "arbitrary_types_allowed": True,
    }
