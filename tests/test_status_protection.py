"""Tests for correlation engine status protection.

Verified relationships (those with llm_verdict set) must not have their
status overwritten when the correlation engine re-runs. These tests verify
that the Cypher queries contain the necessary guards.

Run with: pytest tests/test_status_protection.py
"""

from __future__ import annotations

import ast
import inspect

from zerofin.analysis import correlations, partial

# =====================================================================
# Helper to extract Cypher query strings from function source
# =====================================================================


def _extract_cypher_queries(module: object) -> list[str]:
    """Pull all triple-quoted strings that contain CORRELATES_WITH from a module."""
    source = inspect.getsource(module)
    queries = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "CORRELATES_WITH" in node.value:
                queries.append(node.value)
    return queries


# =====================================================================
# Status protection in MERGE queries
# =====================================================================


class TestCorrelationEngineStatusProtection:
    """Ensure MERGE queries preserve status on LLM-verified edges."""

    def test_pearson_merge_queries_protect_status(self) -> None:
        """Pearson MERGE queries restore old status when llm_verdict exists."""
        queries = _extract_cypher_queries(correlations)
        merge_queries = [q for q in queries if "MERGE" in q and "SET r +=" in q]
        assert len(merge_queries) >= 2, "Expected at least 2 MERGE queries in correlations.py"

        for query in merge_queries:
            assert "old_verdict" in query, (
                "MERGE query missing old_verdict capture — "
                "verified edges will be overwritten"
            )
            assert "old_status" in query, (
                "MERGE query missing old_status capture — "
                "verified edges will be overwritten"
            )

    def test_partial_merge_query_protects_status(self) -> None:
        """Partial correlation MERGE query restores old status when llm_verdict exists."""
        queries = _extract_cypher_queries(partial)
        merge_queries = [q for q in queries if "MERGE" in q and "SET r +=" in q]
        assert len(merge_queries) >= 1, "Expected at least 1 MERGE query in partial.py"

        for query in merge_queries:
            assert "old_verdict" in query, (
                "Partial MERGE query missing old_verdict capture — "
                "verified edges will be overwritten"
            )
            assert "old_status" in query, (
                "Partial MERGE query missing old_status capture — "
                "verified edges will be overwritten"
            )


# =====================================================================
# Status protection in DELETE queries
# =====================================================================


class TestCorrelationEngineDeleteProtection:
    """Ensure DELETE queries skip LLM-verified edges."""

    def test_pearson_clear_candidates_skips_verified(self) -> None:
        """Pearson _clear_old_candidates only deletes edges without llm_verdict."""
        queries = _extract_cypher_queries(correlations)
        delete_queries = [q for q in queries if "DELETE r" in q]
        assert len(delete_queries) >= 1, "Expected at least 1 DELETE query in correlations.py"

        for query in delete_queries:
            assert "llm_verdict IS NULL" in query, (
                "DELETE query missing llm_verdict IS NULL guard — "
                "will delete verified edges"
            )

    def test_partial_clear_candidates_skips_verified(self) -> None:
        """Partial _clear_old_partial_candidates only deletes edges without llm_verdict."""
        queries = _extract_cypher_queries(partial)
        delete_queries = [q for q in queries if "DELETE r" in q]
        assert len(delete_queries) >= 1, "Expected at least 1 DELETE query in partial.py"

        for query in delete_queries:
            assert "llm_verdict IS NULL" in query, (
                "Partial DELETE query missing llm_verdict IS NULL guard — "
                "will delete verified edges"
            )


# =====================================================================
# Config tests
# =====================================================================


class TestOutputConfig:
    """Tests for the LOG_OUTPUT_DIR setting."""

    def test_log_output_dir_exists_in_settings(self) -> None:
        """LOG_OUTPUT_DIR is defined in settings."""
        from zerofin.config import settings
        assert hasattr(settings, "LOG_OUTPUT_DIR")
        assert settings.LOG_OUTPUT_DIR, "LOG_OUTPUT_DIR should not be empty"

    def test_log_output_dir_does_not_double_nest(self) -> None:
        """LOG_OUTPUT_DIR should not contain 'zerofin/zerofin' (platformdirs bug)."""
        from zerofin.config import settings
        path = settings.LOG_OUTPUT_DIR.replace("\\", "/")
        assert "zerofin/zerofin" not in path, (
            f"LOG_OUTPUT_DIR has double-nested path: {settings.LOG_OUTPUT_DIR}"
        )
