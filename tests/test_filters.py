"""
Tests for the correlation engine filter functions.

These test the filters that remove false positives:
FDR correction, stability filter, and monthly stability checks.
Each test uses fake data where we know the right answer.

Run with:
    pytest tests/test_filters.py -v
"""

from __future__ import annotations

import numpy as np

from zerofin.analysis.filters import (
    _apply_fdr_correction,
    _bootstrap_ci_excludes_zero,
    _magnitude_stable,
    _sign_consistent,
    _tercile_consistent,
    classify_entity,
    is_pair_plausible,
)

# =====================================================================
# FDR Correction
# =====================================================================


class TestFDRCorrection:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_obvious_signal_survives(self) -> None:
        """A very low p-value should always survive FDR."""
        results = [
            {"pearson_r": 0.9, "pearson_p": 0.0001, "entity_a": "A", "entity_b": "B"},
        ]
        surviving = _apply_fdr_correction(results)
        assert len(surviving) == 1
        assert surviving[0]["entity_a"] == "A"

    def test_obvious_noise_rejected(self) -> None:
        """A high p-value (random noise) should be rejected."""
        results = [
            {"pearson_r": 0.1, "pearson_p": 0.8, "entity_a": "A", "entity_b": "B"},
        ]
        surviving = _apply_fdr_correction(results)
        assert len(surviving) == 0

    def test_fdr_reduces_count(self) -> None:
        """FDR should reject some results that look significant individually.

        With many tests, some will look significant by chance.
        FDR adjusts for this and should reduce the count.
        """
        # 100 results: 5 real signals, 95 noise
        results = []
        for i in range(5):
            results.append({
                "pearson_r": 0.8,
                "pearson_p": 0.001,
                "entity_a": f"real_{i}",
                "entity_b": "X",
            })
        for i in range(95):
            results.append({
                "pearson_r": 0.3,
                "pearson_p": 0.04 + i * 0.001,
                "entity_a": f"noise_{i}",
                "entity_b": "X",
            })

        surviving = _apply_fdr_correction(results)
        # The 5 real signals should survive
        assert len(surviving) >= 5
        # But not all 100 should survive
        assert len(surviving) < 100

    def test_empty_input(self) -> None:
        """Empty input should return empty output."""
        assert _apply_fdr_correction([]) == []

    def test_does_not_mutate_input(self) -> None:
        """FDR should not modify the original input dicts."""
        results = [
            {"pearson_r": 0.9, "pearson_p": 0.001, "entity_a": "A", "entity_b": "B"},
        ]
        original_keys = set(results[0].keys())
        _apply_fdr_correction(results)
        # Original dict should not have adjusted_p added
        assert set(results[0].keys()) == original_keys

    def test_adjusted_p_added_to_survivors(self) -> None:
        """Surviving results should have an adjusted_p field."""
        results = [
            {"pearson_r": 0.9, "pearson_p": 0.0001, "entity_a": "A", "entity_b": "B"},
        ]
        surviving = _apply_fdr_correction(results)
        assert "adjusted_p" in surviving[0]


# =====================================================================
# Bootstrap CI
# =====================================================================


class TestBootstrapCI:
    """Tests for bootstrap confidence interval check."""

    def test_strong_correlation_excludes_zero(self) -> None:
        """Two clearly correlated series should have a CI that excludes zero."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.3  # Strongly correlated
        assert _bootstrap_ci_excludes_zero(x, y)

    def test_random_data_includes_zero(self) -> None:
        """Two unrelated series should have a CI that includes zero."""
        np.random.seed(42)
        x = np.random.randn(30)
        y = np.random.randn(30)  # No correlation
        assert not _bootstrap_ci_excludes_zero(x, y)


# =====================================================================
# Sign Consistency
# =====================================================================


class TestSignConsistency:
    """Tests for split-half sign consistency."""

    def test_consistent_positive(self) -> None:
        """Positive in both halves should pass."""
        np.random.seed(42)
        n = 40
        x = np.random.randn(n)
        y = x + np.random.randn(n) * 0.3  # Positive throughout
        assert _sign_consistent(x, y)

    def test_sign_flip_fails(self) -> None:
        """Positive in first half, negative in second should fail."""
        np.random.seed(42)
        n = 40
        x = np.arange(n, dtype=float)
        # First half: y goes up with x. Second half: y goes down with x.
        y = np.concatenate([
            np.arange(20, dtype=float),
            np.arange(20, 0, -1, dtype=float),
        ])
        assert not _sign_consistent(x, y)


# =====================================================================
# Tercile Consistency
# =====================================================================


class TestTercileConsistency:
    """Tests for three-way sign consistency."""

    def test_consistent_across_thirds(self) -> None:
        """Same sign in all three thirds should pass."""
        np.random.seed(42)
        n = 60
        x = np.random.randn(n)
        y = x + np.random.randn(n) * 0.3
        assert _tercile_consistent(x, y)

    def test_one_third_flipped_fails(self) -> None:
        """If one third has opposite sign, should fail."""
        n = 60
        x = np.arange(n, dtype=float)
        # First two thirds positive, last third negative
        y = np.concatenate([
            np.arange(20, dtype=float),
            np.arange(20, dtype=float),
            np.arange(20, 0, -1, dtype=float),
        ])
        assert not _tercile_consistent(x, y)


# =====================================================================
# Magnitude Stability
# =====================================================================


class TestMagnitudeStability:
    """Tests for strength consistency across halves."""

    def test_similar_strength_passes(self) -> None:
        """Both halves with similar correlation should pass."""
        np.random.seed(42)
        n = 40
        x = np.random.randn(n)
        y = x + np.random.randn(n) * 0.3
        assert _magnitude_stable(x, y)

    def test_one_weak_half_fails(self) -> None:
        """If one half is much weaker than overall, should fail."""
        # First half: strong correlation
        x1 = np.arange(20, dtype=float)
        y1 = x1 * 2
        # Second half: no correlation
        np.random.seed(42)
        x2 = np.random.randn(20)
        y2 = np.random.randn(20)

        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        assert not _magnitude_stable(x, y)


# =====================================================================
# Entity Classification
# =====================================================================


class TestClassifyEntity:
    """Tests for entity type classification."""

    def test_stock_classified_as_equity(self) -> None:
        """Individual stocks should be classified as equity."""
        assert classify_entity("asset:NVDA") == "equity"
        assert classify_entity("asset:AAPL") == "equity"

    def test_fred_housing_classified(self) -> None:
        """Housing FRED indicators get their own category."""
        assert classify_entity("indicator:HOUST") == "fred_housing"
        assert classify_entity("indicator:PERMIT") == "fred_housing"
        assert classify_entity("indicator:MORTGAGE30US") == "fred_housing"

    def test_fred_labor_classified(self) -> None:
        """Labor FRED indicators get their own category."""
        assert classify_entity("indicator:UNRATE") == "fred_labor"
        assert classify_entity("indicator:PAYEMS") == "fred_labor"

    def test_commodity_classified(self) -> None:
        """Commodities get sub-typed."""
        assert classify_entity("asset:ZC=F") == "commodity_ag"
        assert classify_entity("asset:GC=F") == "commodity_metal"
        assert classify_entity("asset:CL=F") == "commodity_energy"

    def test_crypto_classified(self) -> None:
        """Crypto gets its own category."""
        assert classify_entity("asset:BTC-USD") == "crypto"

    def test_intl_etf_classified(self) -> None:
        """International ETFs get their own category."""
        assert classify_entity("asset:EWZ") == "intl_etf"
        assert classify_entity("asset:IEMG") == "intl_etf"

    def test_bond_etf_classified(self) -> None:
        """Bond ETFs get their own category."""
        assert classify_entity("asset:TLT") == "bond_etf"
        assert classify_entity("asset:HYG") == "bond_etf"

    def test_vix_classified(self) -> None:
        """VIX gets its own category."""
        assert classify_entity("asset:^VIX") == "volatility"


# =====================================================================
# Gate 2: Plausibility Filter
# =====================================================================


class TestPlausibilityFilter:
    """Tests for entity type pairing plausibility check."""

    def test_equity_equity_allowed(self) -> None:
        """Two stocks should be allowed to correlate."""
        assert is_pair_plausible("asset:NVDA", "asset:AMD")

    def test_housing_bond_allowed(self) -> None:
        """Housing indicators and bond ETFs make sense together."""
        assert is_pair_plausible("indicator:MORTGAGE30US", "asset:TLT")

    def test_housing_intl_blocked(self) -> None:
        """Housing indicators shouldn't correlate with international ETFs."""
        assert not is_pair_plausible("indicator:HOUST", "asset:EWZ")
        assert not is_pair_plausible("indicator:PERMIT", "asset:IEMG")

    def test_housing_ag_commodity_blocked(self) -> None:
        """Housing and soybeans have no connection."""
        assert not is_pair_plausible("indicator:PERMIT", "asset:ZS=F")

    def test_labor_commodity_blocked(self) -> None:
        """Labor indicators and commodities have no connection."""
        assert not is_pair_plausible("indicator:UNRATE", "asset:ZC=F")
        assert not is_pair_plausible("indicator:PAYEMS", "asset:GC=F")

    def test_labor_intl_blocked(self) -> None:
        """US labor data and international ETFs have no connection."""
        assert not is_pair_plausible("indicator:JTSJOL", "asset:EWZ")

    def test_labor_crypto_blocked(self) -> None:
        """Labor data and crypto have no connection."""
        assert not is_pair_plausible("indicator:CCSA", "asset:BTC-USD")

    def test_credit_equity_allowed(self) -> None:
        """Credit conditions and equities make sense together."""
        assert is_pair_plausible("indicator:NFCI", "asset:NVDA")

    def test_consumer_equity_allowed(self) -> None:
        """Consumer indicators and stocks make sense."""
        assert is_pair_plausible("indicator:UMCSENT", "asset:WMT")

    def test_rates_bond_allowed(self) -> None:
        """Interest rates and bond ETFs are directly related."""
        assert is_pair_plausible("indicator:DGS10", "asset:TLT")

    def test_commodity_commodity_allowed(self) -> None:
        """Commodities can correlate with each other."""
        assert is_pair_plausible("asset:GC=F", "asset:SI=F")
