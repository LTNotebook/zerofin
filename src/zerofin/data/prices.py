"""
Price data collector — pulls stock/ETF/commodity prices from Yahoo Finance.

This module is the first data collection plugin for Zerofin. It uses the
yfinance library to download price data and stores it in PostgreSQL via
the PostgresStorage class.

How it works:
1. Give it a list of ticker symbols (or use the defaults from tickers.py)
2. It asks Yahoo Finance for price history using yf.download() — one API
   call for ALL tickers at once (batch download, much faster)
3. Each row/ticker combo gets validated through the DataPointCreate Pydantic model
4. Valid data points get stored in PostgreSQL
5. You get back a summary of what succeeded and what failed

Note on pandas: yfinance returns pandas DataFrames — we can't avoid that
since it's baked into yfinance's API. We extract values from the DataFrame
and convert them to Decimal for financial precision. We do NOT use pandas
for our own analysis — that's what Polars is for later.
"""

from __future__ import annotations

import logging
import math
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import pendulum
import yfinance as yf
from pydantic import ValidationError

from zerofin.data.collector import BaseCollector
from zerofin.data.tickers import ALL_TICKERS
from zerofin.models.entities import DataPointCreate
from zerofin.storage.postgres import PostgresStorage

# Set up logger — every module gets its own logger named after the module path.
# In log output you'll see "zerofin.data.prices" so you know exactly where
# the message came from.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default tickers — pulled from the central tickers module.
# Other code can still do PriceCollector(tickers=["AAPL"]) to override.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = ALL_TICKERS

# How far back to pull when collecting historical data.
# 1 year gives enough data for meaningful trend analysis.
DEFAULT_HISTORY_PERIOD = "1y"

# How far back to look when collecting the "latest" price.
# We ask for 5 days so weekends and holidays don't return empty results —
# we just take the last row, which will always be the most recent trading day.
DEFAULT_LATEST_PERIOD = "5d"

# Decimal precision for financial prices — 4 decimal places covers
# penny stocks and forex pairs without excess noise.
PRICE_DECIMAL_PLACES = Decimal("0.0001")

# The data source name we record in the database.
# Must match one of VALID_SOURCES in models/entities.py.
DATA_SOURCE = "yfinance"


def _parse_batch_row(
    ticker: str,
    date_index: object,
    close_value: object,
    volume_value: object,
) -> list[DataPointCreate]:
    """Convert one ticker's data for one date into validated DataPointCreate models.

    This is the batch-download version of the old _parse_yfinance_row().
    Instead of receiving a pandas row, it receives the individual values
    already extracted from the MultiIndex DataFrame.

    Each call produces up to 2 data points: close_price and volume.

    Args:
        ticker:       The ticker symbol (e.g., "NVDA").
        date_index:   The pandas Timestamp index value for this row.
        close_value:  The closing price (float or NaN).
        volume_value: The trading volume (float or NaN).

    Returns:
        A list of validated DataPointCreate models (0, 1, or 2 items).
        Returns fewer than 2 if some values are missing or invalid.
    """
    data_points: list[DataPointCreate] = []

    # Convert the pandas Timestamp to a pendulum DateTime.
    # yfinance dates are timezone-aware (usually UTC or exchange TZ).
    # We convert to UTC for consistent storage.
    try:
        timestamp = pendulum.parse(str(date_index), tz="UTC")
    except Exception:
        logger.warning(
            "Could not parse date '%s' for %s — skipping row",
            date_index,
            ticker,
        )
        return []

    # --- Close price ---
    # Convert the float from pandas to Decimal for financial precision.
    # Floats can't represent 0.10 exactly, but Decimal can.
    try:
        if isinstance(close_value, float) and math.isnan(close_value):
            logger.debug("Close price is NaN for %s on %s — skipping", ticker, date_index)
        else:
            # Round to the standard price precision defined at the top of this file
            close_decimal = Decimal(str(close_value)).quantize(PRICE_DECIMAL_PLACES)

            close_point = DataPointCreate(
                entity_type="asset",
                entity_id=ticker,
                metric="close_price",
                value=close_decimal,
                unit="USD",
                timestamp=timestamp,
                source=DATA_SOURCE,
            )
            data_points.append(close_point)

    except (InvalidOperation, ValidationError) as error:
        logger.warning(
            "Failed to parse close price for %s on %s: %s",
            ticker,
            date_index,
            error,
        )

    # --- Volume ---
    # Volume is the number of shares traded. It's always a whole number,
    # but we still store it as Decimal for consistency with the data model.
    try:
        if isinstance(volume_value, float) and math.isnan(volume_value):
            logger.debug("Volume is NaN for %s on %s — skipping", ticker, date_index)
        else:
            # Volume is an integer, no decimal places needed
            volume_decimal = Decimal(str(int(volume_value)))

            volume_point = DataPointCreate(
                entity_type="asset",
                entity_id=ticker,
                metric="volume",
                value=volume_decimal,
                unit="shares",
                timestamp=timestamp,
                source=DATA_SOURCE,
            )
            data_points.append(volume_point)

    except (InvalidOperation, ValidationError) as error:
        logger.warning(
            "Failed to parse volume for %s on %s: %s",
            ticker,
            date_index,
            error,
        )

    return data_points


def _is_multiindex(df: object) -> bool:
    """Check whether a DataFrame has MultiIndex columns.

    yf.download() returns MultiIndex columns when given multiple tickers
    (columns are tuples like ("Close", "NVDA")), but flat columns when
    given a single ticker (columns are just "Close", "Volume", etc.).

    We need to know which format we're dealing with so we can extract
    data correctly.
    """
    import pandas as pd

    return isinstance(df.columns, pd.MultiIndex)


class PriceCollector(BaseCollector):
    """Collects price data from Yahoo Finance and stores it in PostgreSQL.

    Uses yf.download() for batch downloading — one API call pulls data for
    ALL tickers at once, which is much faster than one-at-a-time pulls.

    Inherits from BaseCollector, which provides the standard collector
    interface (collect_latest, collect_history) and the _build_summary()
    helper for consistent return values across all collectors.

    Usage:
        # Collect today's prices for the default tickers
        collector = PriceCollector()
        result = collector.collect_latest()
        logger.info(result)  # {"collector": "prices", "stored": 10, "failed": 0, ...}

        # Collect 1 year of history for custom tickers
        collector = PriceCollector(tickers=["AAPL", "MSFT", "GOOG"])
        result = collector.collect_history(period="1y")

    Args:
        tickers: List of ticker symbols to collect. Defaults to DEFAULT_TICKERS.
    """

    # Identifies this collector in logs and summary dicts
    collector_name: str = "prices"

    def __init__(self, tickers: list[str] | None = None) -> None:
        # Use the provided tickers, or fall back to the defaults.
        # We uppercase everything to stay consistent with the DataPointCreate
        # validator, which forces entity_id to uppercase anyway.
        self.tickers: list[str] = [t.upper() for t in (tickers or DEFAULT_TICKERS)]
        logger.info(
            "PriceCollector initialized with %d tickers: %s",
            len(self.tickers),
            self.tickers,
        )

    def collect_latest(self) -> dict[str, Any]:
        """Pull the most recent day's prices for all tickers and store them.

        Uses period="5d" and takes only the last row to handle weekends
        and holidays — we always get the most recent trading day.

        Returns:
            A summary dict with counts:
            - collector:      Name of this collector ("prices")
            - stored:         Total data points successfully saved
            - failed:         Total data points that failed validation or storage
            - tickers_ok:     Number of tickers that succeeded
            - tickers_failed: Number of tickers that had errors
        """
        logger.info("Collecting latest prices for %d tickers", len(self.tickers))
        return self._collect(period=DEFAULT_LATEST_PERIOD, latest_only=True)

    def collect_history(self, **kwargs: Any) -> dict[str, Any]:
        """Pull historical prices for all tickers and store them.

        This is meant to be run once to backfill data, not daily. After the
        initial backfill, use collect_latest() for daily updates.

        Keyword Args:
            period: How far back to go. Uses yfinance period strings:
                    "1mo", "3mo", "6mo", "1y", "2y", "5y", "max".
                    Default is "1y" (one year).

        Returns:
            A summary dict with the same shape as collect_latest().
        """
        period: str = kwargs.get("period", DEFAULT_HISTORY_PERIOD)

        logger.info(
            "Collecting %s of historical prices for %d tickers",
            period,
            len(self.tickers),
        )
        return self._collect(period=period, latest_only=False)

    def _collect(
        self,
        period: str,
        latest_only: bool,
    ) -> dict[str, Any]:
        """Internal workhorse — batch-downloads data, validates it, and stores it.

        This uses yf.download() to pull ALL tickers in a single API call,
        then iterates through the results to build DataPointCreate models.

        yf.download() with multiple tickers returns a DataFrame with
        MultiIndex columns: (metric, ticker). For a single ticker, columns
        are just metric names (no MultiIndex). We handle both cases.

        Args:
            period:      yfinance period string (e.g., "5d", "1y").
            latest_only: If True, only store the most recent row per ticker.

        Returns:
            Summary dict with collector/stored/failed/tickers_ok/tickers_failed.
        """
        total_stored = 0
        total_failed = 0
        tickers_ok = 0
        tickers_failed = 0

        # ------------------------------------------------------------------
        # Step 1: Batch-download all tickers in ONE call
        # ------------------------------------------------------------------
        try:
            logger.info(
                "Batch downloading %s data for %d tickers: %s",
                period,
                len(self.tickers),
                self.tickers,
            )

            # yf.download() is the batch API — much faster than calling
            # yf.Ticker().history() one at a time.
            # Retry once on failure per CLAUDE.md conventions.
            data = yf.download(self.tickers, period=period)

            if data.empty:
                logger.warning("yf.download returned empty — retrying once")
                time.sleep(2)
                data = yf.download(self.tickers, period=period)

            if data.empty:
                logger.warning("yf.download returned no data after retry")
                return self._build_summary(
                    stored=0,
                    failed=0,
                    tickers_ok=0,
                    tickers_failed=len(self.tickers),
                )

        except Exception as download_error:
            logger.warning("Batch download failed: %s — retrying once", download_error)
            try:
                time.sleep(2)
                data = yf.download(self.tickers, period=period)
                if data.empty:
                    raise ValueError("Retry returned empty data")
            except Exception as retry_error:
                logger.error("Batch download failed after retry: %s", retry_error)
                return self._build_summary(
                    stored=0,
                    failed=0,
                    tickers_ok=0,
                    tickers_failed=len(self.tickers),
                )

        # If we only want the latest day, slice to just the last row.
        if latest_only:
            data = data.iloc[-1:]

        # Detect whether we got MultiIndex columns (multiple tickers) or
        # flat columns (single ticker). This changes how we access the data.
        multi = _is_multiindex(data)

        logger.info(
            "Got %d rows, MultiIndex=%s, columns=%s",
            len(data),
            multi,
            list(data.columns),
        )

        # ------------------------------------------------------------------
        # Step 2: Iterate through tickers, validate, and store
        # ------------------------------------------------------------------
        with PostgresStorage() as db:
            for ticker in self.tickers:
                try:
                    # Extract this ticker's close and volume columns.
                    # MultiIndex: data["Close"]["NVDA"]
                    # Flat (single ticker): data["Close"]
                    if multi:
                        # Check if this ticker exists in the downloaded data.
                        # Some tickers might not have data (delisted, typo, etc.)
                        if ticker not in data["Close"].columns:
                            logger.warning(
                                "No data for %s in batch download — ticker may be invalid",
                                ticker,
                            )
                            tickers_failed += 1
                            continue

                        close_series = data["Close"][ticker]
                        volume_series = data["Volume"][ticker]
                    else:
                        # Single ticker — columns are flat, no ticker level
                        close_series = data["Close"]
                        volume_series = data["Volume"]

                    # Count how many non-NaN rows this ticker has
                    valid_row_count = close_series.dropna().shape[0]
                    if valid_row_count == 0:
                        logger.warning(
                            "All data is NaN for %s — skipping",
                            ticker,
                        )
                        tickers_failed += 1
                        continue

                    logger.info(
                        "Processing %d rows for %s (from %s to %s)",
                        len(close_series),
                        ticker,
                        close_series.index[0],
                        close_series.index[-1],
                    )

                    # --------------------------------------------------
                    # Validate each row through the Pydantic model
                    # --------------------------------------------------
                    valid_points: list[DataPointCreate] = []
                    row_failed = 0

                    for date_index in close_series.index:
                        close_val = close_series[date_index]
                        volume_val = volume_series[date_index]

                        # Skip rows where both values are NaN — these are
                        # non-trading days (weekends, holidays). Not a failure.
                        close_is_nan = close_val != close_val
                        volume_is_nan = volume_val != volume_val
                        if close_is_nan and volume_is_nan:
                            continue

                        points = _parse_batch_row(ticker, date_index, close_val, volume_val)
                        valid_points.extend(points)

                        # Count actual validation failures — rows that HAD data
                        # but failed Pydantic validation. NaN skips don't count.
                        expected = (0 if close_is_nan else 1) + (0 if volume_is_nan else 1)
                        row_failed += expected - len(points)

                    if row_failed > 0:
                        logger.warning(
                            "%d data points failed validation for %s",
                            row_failed,
                            ticker,
                        )

                    # --------------------------------------------------
                    # Store all valid data points in PostgreSQL (batch)
                    # --------------------------------------------------
                    store_failed = 0
                    batch = [
                        {
                            "entity_type": point.entity_type,
                            "entity_id": point.entity_id,
                            "metric": point.metric,
                            "value": point.value,
                            "unit": point.unit,
                            "timestamp": point.timestamp,
                            "source": point.source,
                            "is_revised": point.is_revised,
                            "revision_of": point.revision_of,
                        }
                        for point in valid_points
                    ]

                    try:
                        stored_count = db.insert_data_points_batch(batch)
                    except Exception as store_error:
                        logger.error(
                            "Batch store failed for %s: %s",
                            ticker,
                            store_error,
                        )
                        stored_count = 0
                        store_failed = len(batch)

                    total_stored += stored_count
                    total_failed += row_failed + store_failed
                    tickers_ok += 1

                    logger.info(
                        "Finished %s: stored %d data points, %d failed",
                        ticker,
                        stored_count,
                        row_failed + store_failed,
                    )

                except Exception as ticker_error:
                    # Catch-all for unexpected errors processing a single ticker.
                    # Log it and move on — one failure shouldn't stop the whole batch.
                    logger.error(
                        "Failed to process data for %s: %s",
                        ticker,
                        ticker_error,
                    )
                    tickers_failed += 1

        # Use the base class _build_summary() for consistent return format
        return self._build_summary(
            stored=total_stored,
            failed=total_failed,
            tickers_ok=tickers_ok,
            tickers_failed=tickers_failed,
        )


def collect_prices() -> dict[str, Any]:
    """Convenience function — create a PriceCollector and run collect_latest().

    This is the entry point for the daily pipeline script. It uses the
    default tickers and pulls just the most recent day's prices.

    Usage:
        from zerofin.data.prices import collect_prices
        result = collect_prices()

    Returns:
        Summary dict from PriceCollector.collect_latest().
    """
    collector = PriceCollector()
    return collector.collect_latest()
