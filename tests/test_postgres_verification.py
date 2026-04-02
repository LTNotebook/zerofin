"""Tests for the Postgres verification audit trail tables and methods.

No live database calls — tests SQL generation and data mapping logic.

Run with: pytest tests/test_postgres_verification.py
"""

from __future__ import annotations

from zerofin.storage.postgres import (
    CREATE_VERIFICATION_RESULTS_TABLE,
    CREATE_VERIFICATION_RUNS_TABLE,
)

# =====================================================================
# Schema definition tests
# =====================================================================


class TestVerificationRunsSchema:
    """Tests for the verification_runs table definition."""

    def test_table_uses_if_not_exists(self) -> None:
        """Table creation is idempotent."""
        assert "IF NOT EXISTS" in CREATE_VERIFICATION_RUNS_TABLE

    def test_table_has_required_columns(self) -> None:
        """All expected columns are in the CREATE statement."""
        required = [
            "started_at", "finished_at", "duration_seconds",
            "total_pairs", "promoted", "rejected", "uncertain",
            "pass1_model", "pass2_model",
        ]
        for col in required:
            assert col in CREATE_VERIFICATION_RUNS_TABLE, f"Missing column: {col}"

    def test_table_has_serial_primary_key(self) -> None:
        """Runs table has an auto-incrementing primary key."""
        assert "SERIAL PRIMARY KEY" in CREATE_VERIFICATION_RUNS_TABLE


class TestVerificationResultsSchema:
    """Tests for the verification_results table definition."""

    def test_table_uses_if_not_exists(self) -> None:
        """Table creation is idempotent."""
        assert "IF NOT EXISTS" in CREATE_VERIFICATION_RESULTS_TABLE

    def test_table_has_required_columns(self) -> None:
        """All expected columns match what the verification pipeline produces."""
        required = [
            "run_id", "entity_a_id", "entity_b_id", "method",
            "window_days", "correlation", "direction",
            "verdict", "confidence", "mechanism",
            "alternative_explanations", "reasoning",
            "relationship_category",
        ]
        for col in required:
            assert col in CREATE_VERIFICATION_RESULTS_TABLE, f"Missing column: {col}"

    def test_table_has_foreign_key_to_runs(self) -> None:
        """Results table references the runs table."""
        assert "REFERENCES verification_runs(id)" in CREATE_VERIFICATION_RESULTS_TABLE

    def test_table_has_serial_primary_key(self) -> None:
        """Results table has an auto-incrementing primary key."""
        assert "SERIAL PRIMARY KEY" in CREATE_VERIFICATION_RESULTS_TABLE
