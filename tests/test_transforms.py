"""
Tests for the correlation engine transform functions.

These test the math that processes raw data before correlation:
log returns, z-scoring, winsorizing, beta removal, and monthly
downsampling. Each test uses simple fake data where we know
the right answer ahead of time.

Run with:
    pytest tests/test_transforms.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pendulum
import polars as pl

from zerofin.analysis.monthly import (
    _build_monthly_indicator_changes,
    _build_monthly_returns,
)
from zerofin.analysis.transforms import (
    _compute_transforms,
    _log_returns,
    _remove_market_sector_beta,
    _winsorize,
    _z_score_all,
)

# =====================================================================
# Log Returns
# =====================================================================


class TestLogReturns:
    """Tests for log return calculation."""

    def test_doubling_price(self) -> None:
        """A stock going from 100 to 200 should give ln(2) ≈ 0.693."""
        series = pl.Series("price", [100.0, 200.0])
        result = _log_returns(series)
        # First value is NaN (no previous price), second is ln(2)
        assert result[0] is None or math.isnan(result[0])
        assert abs(result[1] - math.log(2)) < 0.0001

    def test_no_change(self) -> None:
        """A stock that doesn't move should give 0."""
        series = pl.Series("price", [100.0, 100.0, 100.0])
        result = _log_returns(series)
        assert abs(result[1]) < 0.0001
        assert abs(result[2]) < 0.0001

    def test_symmetry(self) -> None:
        """Going up 10% then down 10% should roughly cancel out.

        This is why we use log returns — with simple returns,
        +10% and -10% don't cancel (you end up with 99, not 100).
        With log returns, they're symmetrical.
        """
        series = pl.Series("price", [100.0, 110.0, 100.0])
        result = _log_returns(series)
        # ln(110/100) + ln(100/110) should equal 0
        total = result[1] + result[2]
        assert abs(total) < 0.0001


# =====================================================================
# Z-Score All
# =====================================================================


class TestZScoreAll:
    """Tests for the z-scoring step."""

    def test_mean_near_zero(self) -> None:
        """After z-scoring, the average should be approximately 0."""
        df = pl.DataFrame({
            "date": pl.Series("date", [1, 2, 3, 4, 5]),
            "asset:TEST": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = _z_score_all(df)
        mean = result["asset:TEST"].mean()
        assert abs(mean) < 0.0001

    def test_std_near_one(self) -> None:
        """After z-scoring, the standard deviation should be approximately 1."""
        df = pl.DataFrame({
            "date": pl.Series("date", [1, 2, 3, 4, 5]),
            "asset:TEST": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = _z_score_all(df)
        std = result["asset:TEST"].std()
        assert abs(std - 1.0) < 0.15  # Allow some tolerance

    def test_constant_series_becomes_null(self) -> None:
        """A series with no variation can't be z-scored — should be null."""
        df = pl.DataFrame({
            "date": pl.Series("date", [1, 2, 3, 4, 5]),
            "asset:FLAT": [42.0, 42.0, 42.0, 42.0, 42.0],
        })
        result = _z_score_all(df)
        assert result["asset:FLAT"].null_count() == 5

    def test_date_column_untouched(self) -> None:
        """The date column should pass through without being z-scored."""
        df = pl.DataFrame({
            "date": pl.Series("date", [1, 2, 3]),
            "asset:A": [10.0, 20.0, 30.0],
        })
        result = _z_score_all(df)
        assert result["date"].to_list() == [1, 2, 3]


# =====================================================================
# Winsorize
# =====================================================================


class TestWinsorize:
    """Tests for extreme value capping."""

    def test_extreme_values_capped(self) -> None:
        """Values far outside the normal range should be clipped."""
        # Create data with one extreme outlier
        values = [0.0] * 98 + [100.0] + [-100.0]
        df = pl.DataFrame({
            "date": list(range(100)),
            "asset:TEST": values,
        })
        result = _winsorize(df)
        # The extreme values should be smaller after capping
        assert result["asset:TEST"].max() < 100.0
        assert result["asset:TEST"].min() > -100.0

    def test_normal_values_unchanged(self) -> None:
        """Values within the normal range shouldn't be affected."""
        values = [float(i) for i in range(100)]
        df = pl.DataFrame({
            "date": list(range(100)),
            "asset:TEST": values,
        })
        result = _winsorize(df)
        # Middle values should be unchanged
        assert result["asset:TEST"][50] == 50.0


# =====================================================================
# Compute Transforms
# =====================================================================


class TestComputeTransforms:
    """Tests for the per-asset-type transformation logic."""

    def test_asset_gets_log_returns(self) -> None:
        """Stock prices should be transformed to log returns."""
        df = pl.DataFrame({
            "date": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True
            ),
            "asset:TEST": [100.0, 110.0, 105.0, 115.0],
        })
        result = _compute_transforms(df)
        # Should have 3 rows (first row lost to diff)
        assert len(result) == 3
        # First value should be ln(110/100)
        expected = math.log(110.0 / 100.0)
        assert abs(result["asset:TEST"][0] - expected) < 0.0001

    def test_indicator_gets_first_diff(self) -> None:
        """FRED indicators should be transformed to first differences."""
        df = pl.DataFrame({
            "date": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True
            ),
            "indicator:DGS10": [4.25, 4.30, 4.28, 4.35],
        })
        result = _compute_transforms(df)
        # First diff: 4.30 - 4.25 = 0.05
        assert abs(result["indicator:DGS10"][0] - 0.05) < 0.0001


# =====================================================================
# Beta Removal
# =====================================================================


class TestBetaRemoval:
    """Tests for market + sector beta removal."""

    def test_unlisted_stock_passes_through(self) -> None:
        """A stock NOT in STOCK_SECTOR_MAP should pass through untouched.

        Beta removal only applies to stocks with a sector mapping.
        Unknown tickers are left as-is.
        """
        np.random.seed(42)
        market = np.random.randn(100) * 0.01

        df = pl.DataFrame({
            "date": list(range(100)),
            "asset:^GSPC": market.tolist(),
            # FAKE isn't in STOCK_SECTOR_MAP so beta removal won't touch it
            "asset:FAKE": (market * 1.5).tolist(),
        })
        result = _remove_market_sector_beta(df)

        assert "asset:^GSPC" in result.columns
        assert "asset:FAKE" in result.columns
        # FAKE should be identical before and after (not beta-removed)
        assert result["asset:FAKE"].to_list() == df["asset:FAKE"].to_list()

    def test_market_column_preserved(self) -> None:
        """The market (^GSPC) itself shouldn't be beta-removed."""
        df = pl.DataFrame({
            "date": list(range(50)),
            "asset:^GSPC": [float(i) * 0.01 for i in range(50)],
            "indicator:DGS10": [float(i) * 0.001 for i in range(50)],
        })
        result = _remove_market_sector_beta(df)
        # Market and indicator should pass through unchanged
        assert result["asset:^GSPC"].to_list() == df["asset:^GSPC"].to_list()
        assert result["indicator:DGS10"].to_list() == df["indicator:DGS10"].to_list()

    def test_indicators_not_beta_removed(self) -> None:
        """Economic indicators shouldn't have market beta removed."""
        np.random.seed(42)
        df = pl.DataFrame({
            "date": list(range(50)),
            "asset:^GSPC": np.random.randn(50).tolist(),
            "indicator:CPI": np.random.randn(50).tolist(),
        })
        result = _remove_market_sector_beta(df)
        # Indicator should be identical before and after
        original = df["indicator:CPI"].to_list()
        after = result["indicator:CPI"].to_list()
        assert original == after


# =====================================================================
# Monthly Returns
# =====================================================================


class TestMonthlyReturns:
    """Tests for monthly downsampling."""

    def test_monthly_return_calculation(self) -> None:
        """Monthly return should be (end price / start price) - 1."""
        rows = []
        # January: price goes 100 -> 110
        for day in range(1, 32):
            try:
                ts = pendulum.datetime(2024, 1, day)
                rows.append({
                    "entity_type": "asset",
                    "entity_id": "TEST",
                    "value": 100.0 + day * (10.0 / 31),
                    "timestamp": ts,
                })
            except ValueError:
                pass

        # February: price goes 110 -> 121
        for day in range(1, 29):
            try:
                ts = pendulum.datetime(2024, 2, day)
                rows.append({
                    "entity_type": "asset",
                    "entity_id": "TEST",
                    "value": 110.0 + day * (11.0 / 28),
                    "timestamp": ts,
                })
            except ValueError:
                pass

        # March: price goes 121 -> 133
        for day in range(1, 32):
            try:
                ts = pendulum.datetime(2024, 3, day)
                rows.append({
                    "entity_type": "asset",
                    "entity_id": "TEST",
                    "value": 121.0 + day * (12.0 / 31),
                    "timestamp": ts,
                })
            except ValueError:
                pass

        result = _build_monthly_returns(rows)
        assert len(result) >= 1  # At least one monthly return
        assert "asset:TEST" in result.columns


class TestMonthlyIndicatorChanges:
    """Tests for monthly indicator first differences."""

    def test_first_difference(self) -> None:
        """Monthly change should be this month minus last month."""
        rows = [
            {
                "entity_type": "indicator",
                "entity_id": "CPI",
                "value": 310.0,
                "timestamp": pendulum.datetime(2024, 1, 15),
            },
            {
                "entity_type": "indicator",
                "entity_id": "CPI",
                "value": 312.0,
                "timestamp": pendulum.datetime(2024, 2, 15),
            },
            {
                "entity_type": "indicator",
                "entity_id": "CPI",
                "value": 315.0,
                "timestamp": pendulum.datetime(2024, 3, 15),
            },
        ]

        result = _build_monthly_indicator_changes(rows)
        assert "indicator:CPI" in result.columns
        # Should have 2 rows (first lost to diff)
        assert len(result) == 2
        # First diff: 312 - 310 = 2
        assert abs(result["indicator:CPI"][0] - 2.0) < 0.01
