"""
Tests for the partial correlation engine.

These test the EBIC-tuned Graphical Lasso precision matrix computation
and pair extraction. Uses synthetic data where we know the right answer.

Run with:
    pytest tests/test_partial.py -v
"""

from __future__ import annotations

import numpy as np
import polars as pl

from zerofin.analysis.partial import (
    MIN_PRECISION_OBS,
    MIN_PRECISION_VARS,
    _compute_partial_correlation_matrix,
    _extract_significant_pairs,
)

# =====================================================================
# Precision Matrix Computation
# =====================================================================


class TestComputePartialCorrelationMatrix:
    """Tests for the EBIC-tuned Graphical Lasso precision matrix computation."""

    def test_strongly_correlated_pair(self) -> None:
        """Two strongly correlated variables should have a high partial corr."""
        rng = np.random.default_rng(42)
        n = 100
        a = rng.normal(0, 1, n)
        b = a * 0.8 + rng.normal(0, 0.5, n)
        c = rng.normal(0, 1, n)  # independent

        df = pl.DataFrame({
            "date": list(range(n)),
            "asset:A": a,
            "asset:B": b,
            "asset:C": c,
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B", "asset:C"]
        )

        assert matrix is not None
        assert n_obs == n

        # A and B should have strong partial correlation
        a_idx = cols.index("asset:A")
        b_idx = cols.index("asset:B")
        assert abs(matrix[a_idx, b_idx]) > 0.5

    def test_indirect_correlation_removed(self) -> None:
        """Two variables correlated only through a third should have low partial corr."""
        rng = np.random.default_rng(42)
        n = 200
        # C is the common cause — A and B both depend on C
        c = rng.normal(0, 1, n)
        a = c * 0.7 + rng.normal(0, 0.5, n)
        b = c * 0.7 + rng.normal(0, 0.5, n)

        df = pl.DataFrame({
            "date": list(range(n)),
            "asset:A": a,
            "asset:B": b,
            "asset:C": c,
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B", "asset:C"]
        )

        assert matrix is not None

        # A and B should have LOW partial correlation (C explains their link)
        a_idx = cols.index("asset:A")
        b_idx = cols.index("asset:B")
        assert abs(matrix[a_idx, b_idx]) < 0.3

    def test_diagonal_is_one(self) -> None:
        """Diagonal of partial correlation matrix should be 1.0."""
        rng = np.random.default_rng(42)
        n = 50
        df = pl.DataFrame({
            "date": list(range(n)),
            "asset:A": rng.normal(0, 1, n),
            "asset:B": rng.normal(0, 1, n),
            "asset:C": rng.normal(0, 1, n),
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B", "asset:C"]
        )

        assert matrix is not None
        for i in range(len(cols)):
            assert abs(matrix[i, i] - 1.0) < 1e-10

    def test_too_few_observations_returns_none(self) -> None:
        """Should return None when there aren't enough observations."""
        df = pl.DataFrame({
            "date": list(range(5)),
            "asset:A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "asset:B": [2.0, 3.0, 4.0, 5.0, 6.0],
            "asset:C": [3.0, 4.0, 5.0, 6.0, 7.0],
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B", "asset:C"]
        )

        assert matrix is None
        assert n_obs == 0

    def test_too_few_variables_returns_none(self) -> None:
        """Should return None when there are fewer than MIN_PRECISION_VARS."""
        rng = np.random.default_rng(42)
        n = 50
        df = pl.DataFrame({
            "date": list(range(n)),
            "asset:A": rng.normal(0, 1, n),
            "asset:B": rng.normal(0, 1, n),
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B"]
        )

        assert matrix is None

    def test_n_obs_returned_correctly(self) -> None:
        """Returned n_obs should match actual observations after NaN removal."""
        rng = np.random.default_rng(42)
        n = 50
        a = rng.normal(0, 1, n).tolist()
        b = rng.normal(0, 1, n).tolist()
        c = rng.normal(0, 1, n).tolist()
        # Add some NaNs
        a[0] = float("nan")
        a[10] = float("nan")
        b[20] = float("nan")

        df = pl.DataFrame({
            "date": list(range(n)),
            "asset:A": a,
            "asset:B": b,
            "asset:C": c,
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B", "asset:C"]
        )

        assert matrix is not None
        # 3 rows had NaN, so n_obs should be n - 3
        assert n_obs == n - 3

    def test_matrix_is_symmetric(self) -> None:
        """Partial correlation matrix should be symmetric."""
        rng = np.random.default_rng(42)
        n = 100
        df = pl.DataFrame({
            "date": list(range(n)),
            "asset:A": rng.normal(0, 1, n),
            "asset:B": rng.normal(0, 1, n),
            "asset:C": rng.normal(0, 1, n),
            "asset:D": rng.normal(0, 1, n),
        })

        matrix, cols, n_obs = _compute_partial_correlation_matrix(
            df, ["asset:A", "asset:B", "asset:C", "asset:D"]
        )

        assert matrix is not None
        np.testing.assert_array_almost_equal(matrix, matrix.T)


# =====================================================================
# Pair Extraction and P-Values
# =====================================================================


class TestExtractSignificantPairs:
    """Tests for extracting pairs from the partial correlation matrix."""

    def _make_matrix(self, values: dict, n: int) -> tuple:
        """Helper: build a partial correlation matrix from specified values."""
        cols = sorted(set(
            k for pair in values for k in pair
        ))
        size = len(cols)
        matrix = np.eye(size)
        for (a, b), r in values.items():
            i = cols.index(a)
            j = cols.index(b)
            matrix[i, j] = r
            matrix[j, i] = r
        return matrix, cols

    def test_above_threshold_extracted(self) -> None:
        """Pairs above threshold should be extracted."""
        matrix, cols = self._make_matrix({
            ("asset:A", "asset:B"): 0.5,
            ("asset:A", "asset:C"): 0.1,
            ("asset:B", "asset:C"): 0.05,
        }, 3)

        results = _extract_significant_pairs(matrix, cols, 0.18, 200)
        entity_pairs = {(r["entity_a"], r["entity_b"]) for r in results}

        assert ("asset:A", "asset:B") in entity_pairs
        assert ("asset:A", "asset:C") not in entity_pairs

    def test_p_value_is_zero_for_glasso(self) -> None:
        """Glasso handles significance via EBIC, so p-value is set to 0."""
        matrix, cols = self._make_matrix({
            ("asset:A", "asset:B"): 0.5,
            ("asset:A", "asset:C"): 0.0,
            ("asset:B", "asset:C"): 0.0,
        }, 3)

        results = _extract_significant_pairs(matrix, cols, 0.18, 200)

        assert len(results) >= 1
        for r in results:
            assert "pearson_p" in r
            assert r["pearson_p"] == 0.0

    def test_observation_count_passed_through(self) -> None:
        """Results should carry the actual n_obs, not window_days."""
        matrix, cols = self._make_matrix({
            ("asset:A", "asset:B"): 0.5,
            ("asset:A", "asset:C"): 0.0,
            ("asset:B", "asset:C"): 0.0,
        }, 3)

        results = _extract_significant_pairs(matrix, cols, 0.18, 175)

        assert len(results) >= 1
        assert results[0]["observation_count"] == 175

    def test_redundant_pairs_skipped(self) -> None:
        """Pairs in the same redundancy group should be skipped."""
        # IEMG and EEM are in the same redundancy group (em_equity)
        matrix, cols = self._make_matrix({
            ("asset:IEMG", "asset:EEM"): 0.9,
            ("asset:IEMG", "asset:AAPL"): 0.0,
            ("asset:EEM", "asset:AAPL"): 0.0,
        }, 3)

        results = _extract_significant_pairs(matrix, cols, 0.18, 200)
        entity_pairs = {(r["entity_a"], r["entity_b"]) for r in results}

        # IEMG-EEM should be skipped (same redundancy group)
        assert ("asset:IEMG", "asset:EEM") not in entity_pairs

    def test_empty_matrix_returns_empty(self) -> None:
        """A matrix of zeros (except diagonal) should return no pairs."""
        matrix = np.eye(4)
        cols = ["asset:A", "asset:B", "asset:C", "asset:D"]

        results = _extract_significant_pairs(matrix, cols, 0.18, 200)
        assert len(results) == 0

    def test_lag_days_always_zero(self) -> None:
        """Partial correlations are always at lag 0."""
        matrix, cols = self._make_matrix({
            ("asset:A", "asset:B"): 0.5,
            ("asset:A", "asset:C"): 0.0,
            ("asset:B", "asset:C"): 0.0,
        }, 3)

        results = _extract_significant_pairs(matrix, cols, 0.18, 200)

        for r in results:
            assert r["lag_days"] == 0


# =====================================================================
# Constants
# =====================================================================


class TestConstants:
    """Verify named constants are reasonable."""

    def test_min_precision_obs_is_positive(self) -> None:
        assert MIN_PRECISION_OBS > 0

    def test_min_precision_vars_at_least_three(self) -> None:
        assert MIN_PRECISION_VARS >= 3
