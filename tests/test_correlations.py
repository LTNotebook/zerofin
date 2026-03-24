"""
Tests for the correlation engine Pydantic models.

Makes sure CorrelationCandidate accepts good data, rejects bad data,
and calculates derived fields (strength, direction, tier) correctly.

Run with:
    pytest tests/test_correlations.py

Or run all tests:
    pytest
"""

from __future__ import annotations

import pendulum
import pytest
from pydantic import ValidationError

from zerofin.models.correlations import CorrelationCandidate, CorrelationRunSummary


# Helper — builds a valid candidate so we don't repeat ourselves every test.
# Tests override specific fields to test that one thing.
def _make_candidate(**overrides) -> CorrelationCandidate:
    """Create a valid CorrelationCandidate with sensible defaults."""
    defaults = {
        "entity_a_type": "asset",
        "entity_a_id": "NVDA",
        "entity_b_type": "asset",
        "entity_b_id": "AMD",
        "correlation": 0.85,
        "p_value": 0.001,
        "method": "pearson",
        "lag_days": 0,
        "observation_count": 252,
        "window_days": 252,
        "window_end": pendulum.now("UTC"),
    }
    defaults.update(overrides)
    return CorrelationCandidate(**defaults)


# =====================================================================
# Valid data — should pass through
# =====================================================================


class TestCorrelationCandidateValid:
    """Tests that good data gets accepted."""

    def test_valid_candidate_accepted(self) -> None:
        """A well-formed correlation should pass validation."""
        candidate = _make_candidate()
        assert candidate.entity_a_id == "NVDA"
        assert candidate.entity_b_id == "AMD"
        assert candidate.correlation == 0.85

    def test_negative_correlation_accepted(self) -> None:
        """Negative correlations are valid — they mean things move opposite."""
        candidate = _make_candidate(correlation=-0.72)
        assert candidate.correlation == -0.72

    def test_indicator_entity_types_accepted(self) -> None:
        """Correlations between economic indicators should work."""
        candidate = _make_candidate(
            entity_a_type="indicator",
            entity_a_id="DGS10",
            entity_b_type="indicator",
            entity_b_id="DFF",
        )
        assert candidate.entity_a_type == "indicator"

    def test_mixed_entity_types_accepted(self) -> None:
        """Asset-to-indicator correlations should work."""
        candidate = _make_candidate(
            entity_a_type="asset",
            entity_a_id="TLT",
            entity_b_type="indicator",
            entity_b_id="DGS10",
        )
        assert candidate.entity_a_type == "asset"
        assert candidate.entity_b_type == "indicator"

    def test_spearman_method_accepted(self) -> None:
        """Spearman correlations should be accepted alongside Pearson."""
        candidate = _make_candidate(method="spearman")
        assert candidate.method == "spearman"

    def test_lagged_correlation_accepted(self) -> None:
        """Correlations with time lag should work."""
        candidate = _make_candidate(lag_days=5)
        assert candidate.lag_days == 5


# =====================================================================
# Derived fields — calculated automatically
# =====================================================================


class TestCorrelationCandidateDerived:
    """Tests that strength, direction, and tier are set correctly."""

    def test_strength_is_absolute_value(self) -> None:
        """Strength should always be positive, even for negative correlations."""
        positive = _make_candidate(correlation=0.75)
        negative = _make_candidate(correlation=-0.75)
        assert positive.strength == 0.75
        assert negative.strength == 0.75

    def test_direction_positive(self) -> None:
        """Positive correlation should have direction 'positive'."""
        candidate = _make_candidate(correlation=0.60)
        assert candidate.direction == "positive"

    def test_direction_negative(self) -> None:
        """Negative correlation should have direction 'negative'."""
        candidate = _make_candidate(correlation=-0.60)
        assert candidate.direction == "negative"

    def test_tier_strong(self) -> None:
        """Correlation |r| >= 0.7 should be tier 'strong'."""
        candidate = _make_candidate(correlation=0.85)
        assert candidate.tier == "strong"

    def test_tier_actionable(self) -> None:
        """Correlation 0.5 <= |r| < 0.7 should be tier 'actionable'."""
        candidate = _make_candidate(correlation=0.55)
        assert candidate.tier == "actionable"

    def test_tier_store(self) -> None:
        """Correlation 0.3 <= |r| < 0.5 should be tier 'store'."""
        candidate = _make_candidate(correlation=0.35)
        assert candidate.tier == "store"

    def test_tier_negative_uses_absolute(self) -> None:
        """A strong negative correlation should still be tier 'strong'."""
        candidate = _make_candidate(correlation=-0.80)
        assert candidate.tier == "strong"


# =====================================================================
# Neo4j properties — ready for storage
# =====================================================================


class TestCorrelationCandidateNeo4j:
    """Tests the to_neo4j_properties() output."""

    def test_neo4j_properties_has_required_fields(self) -> None:
        """Neo4j properties should include everything needed for the edge."""
        candidate = _make_candidate()
        props = candidate.to_neo4j_properties()

        assert "strength" in props
        assert "direction" in props
        assert "correlation" in props
        assert "p_value" in props
        assert "method" in props
        assert "lag_days" in props
        assert "confidence" in props
        assert "source" in props
        assert "status" in props
        assert "times_tested" in props
        assert "times_confirmed" in props
        assert "valid_from" in props

    def test_neo4j_properties_defaults(self) -> None:
        """New candidates should start with neutral confidence and candidate status."""
        candidate = _make_candidate()
        props = candidate.to_neo4j_properties()

        assert props["confidence"] == 0.5
        assert props["source"] == "statistical"
        assert props["status"] == "candidate"
        assert props["times_tested"] == 0
        assert props["times_confirmed"] == 0

    def test_neo4j_properties_values_match(self) -> None:
        """Property values should match the candidate's fields."""
        candidate = _make_candidate(correlation=0.75, lag_days=3, method="pearson")
        props = candidate.to_neo4j_properties()

        assert props["correlation"] == 0.75
        assert props["strength"] == 0.75
        assert props["direction"] == "positive"
        assert props["lag_days"] == 3
        assert props["method"] == "pearson"


# =====================================================================
# Rejected data — bad input should fail
# =====================================================================


class TestCorrelationCandidateRejected:
    """Tests that bad data gets rejected."""

    def test_correlation_above_1_rejected(self) -> None:
        """Correlation can't be greater than 1.0 — that's mathematically impossible."""
        with pytest.raises(ValidationError):
            _make_candidate(correlation=1.5)

    def test_correlation_below_negative_1_rejected(self) -> None:
        """Correlation can't be less than -1.0."""
        with pytest.raises(ValidationError):
            _make_candidate(correlation=-1.5)

    def test_self_correlation_rejected(self) -> None:
        """Can't correlate an entity with itself — that's always 1.0 and meaningless."""
        with pytest.raises(ValidationError):
            _make_candidate(entity_a_id="NVDA", entity_b_id="NVDA")

    def test_invalid_entity_type_rejected(self) -> None:
        """Entity types must be 'asset' or 'indicator'."""
        with pytest.raises(ValidationError):
            _make_candidate(entity_a_type="banana")

    def test_invalid_method_rejected(self) -> None:
        """Method must be 'pearson' or 'spearman'."""
        with pytest.raises(ValidationError):
            _make_candidate(method="magic")

    def test_negative_lag_rejected(self) -> None:
        """Lag days can't be negative."""
        with pytest.raises(ValidationError):
            _make_candidate(lag_days=-1)

    def test_zero_observations_rejected(self) -> None:
        """Can't have zero observations — there's no data to correlate."""
        with pytest.raises(ValidationError):
            _make_candidate(observation_count=0)

    def test_p_value_above_1_rejected(self) -> None:
        """P-value can't exceed 1.0."""
        with pytest.raises(ValidationError):
            _make_candidate(p_value=1.5)

    def test_empty_entity_id_rejected(self) -> None:
        """Entity IDs can't be empty."""
        with pytest.raises(ValidationError):
            _make_candidate(entity_a_id="")

    def test_entity_id_uppercase_enforced(self) -> None:
        """Lowercase entity IDs should be auto-converted to uppercase."""
        candidate = _make_candidate(entity_a_id="nvda", entity_b_id="amd")
        assert candidate.entity_a_id == "NVDA"
        assert candidate.entity_b_id == "AMD"


# =====================================================================
# CorrelationRunSummary — run stats
# =====================================================================


class TestCorrelationRunSummary:
    """Tests for the run summary model."""

    def test_valid_summary_accepted(self) -> None:
        """A well-formed summary should pass."""
        summary = CorrelationRunSummary(
            total_pairs_tested=19900,
            pairs_above_threshold=2500,
            pairs_surviving_fdr=350,
            relationships_stored=350,
            window_days=252,
            window_end=pendulum.now("UTC"),
            duration_seconds=12.5,
        )
        assert summary.total_pairs_tested == 19900
        assert summary.relationships_stored == 350

    def test_zero_results_accepted(self) -> None:
        """A run that found nothing is still valid — just means no correlations."""
        summary = CorrelationRunSummary(
            total_pairs_tested=19900,
            pairs_above_threshold=0,
            pairs_surviving_fdr=0,
            relationships_stored=0,
            window_days=252,
            window_end=pendulum.now("UTC"),
            duration_seconds=8.3,
        )
        assert summary.pairs_surviving_fdr == 0
