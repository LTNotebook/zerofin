"""
Economic data collector — pulls indicator data from the FRED API.

FRED (Federal Reserve Economic Data) is a free API from the St. Louis Fed that
provides thousands of economic time series: inflation, unemployment, GDP, interest
rates, yield curves, and more.

This module:
1. Connects to FRED using the fredapi library
2. Pulls the latest values (or historical data) for key economic indicators
3. Validates every data point through our Pydantic model (DataPointCreate)
4. Stores the validated data in PostgreSQL

The fredapi library returns pandas Series objects. We convert values to Decimal
for financial precision and dates to pendulum DateTime for consistency with the
rest of the codebase.
"""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any

import pendulum
from fredapi import Fred
from pydantic import ValidationError

from zerofin.config import settings
from zerofin.data.collector import BaseCollector
from zerofin.data.tickers import FRED_INDICATORS
from zerofin.models.entities import DataPointCreate
from zerofin.storage.postgres import PostgresStorage

# Set up logger — every module gets its own logger named after the module path.
logger = logging.getLogger(__name__)


# The full indicator list lives in tickers.py (FRED_INDICATORS) — single source
# of truth for all FRED series IDs and their metadata. You can pass a custom
# subset to EconomicCollector if you only want specific series.


class EconomicCollector(BaseCollector):
    """Collects economic indicator data from FRED and stores it in PostgreSQL.

    Inherits from BaseCollector to follow the standard collector interface.
    Uses self._build_summary() for consistent return formats across all collectors.

    Usage:
        # Collect the latest reading for all FRED_INDICATORS:
        collector = EconomicCollector()
        summary = collector.collect_latest()

        # Collect 2 years of historical data:
        collector = EconomicCollector()
        summary = collector.collect_history(years=2)

        # Track only a subset — build a dict with the same FredMeta shape:
        from zerofin.data.tickers import FRED_INDICATORS
        subset = {k: v for k, v in FRED_INDICATORS.items() if v["category"] == "inflation"}
        collector = EconomicCollector(indicators=subset)
        summary = collector.collect_latest()
    """

    # Identifies this collector in logs and pipeline summaries
    collector_name: str = "economic"

    def __init__(
        self,
        indicators: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Set up the collector with a FRED client and indicator list.

        Args:
            indicators: Optional dict mapping FRED series IDs to their metadata.
                        Defaults to FRED_INDICATORS from tickers.py if not provided.
        """
        # Use the provided indicators or fall back to the full set from tickers.py
        self.indicators = indicators if indicators is not None else FRED_INDICATORS

        # Create the FRED API client. The API key comes from our .env file
        # via settings. We don't create the client if the key is empty —
        # that check happens in the collect methods so we can log a warning.
        self._api_key = settings.FRED_API_KEY
        self._fred: Fred | None = None

    def _get_fred_client(self) -> Fred | None:
        """Create and return the FRED client, or None if no API key is set.

        We lazily create the client so the class can be instantiated even
        without an API key (useful for testing). The client is cached after
        the first call.
        """
        # If no API key, we can't do anything
        if not self._api_key:
            logger.warning(
                "FRED_API_KEY is empty — cannot collect economic data. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return None

        # Create the client once and reuse it
        if self._fred is None:
            self._fred = Fred(api_key=self._api_key)
            logger.debug("FRED API client created")

        return self._fred

    def collect_latest(self) -> dict[str, Any]:
        """Pull the most recent observation for each indicator and store it.

        This is the main method for daily collection. For each indicator:
        1. Fetch the most recent value from FRED
        2. Validate it through DataPointCreate
        3. Store it in PostgreSQL

        If one indicator fails, we log the error and keep going with the rest.
        The pipeline shouldn't stop because one series is temporarily unavailable.

        Returns:
            Summary dict from _build_summary() with stored/failed counts,
            plus extra keys: skipped_null, details.
        """
        # Check for API key before doing anything
        fred = self._get_fred_client()
        if fred is None:
            return self._build_summary(stored=0, failed=0, skipped_null=0, details=[])

        # Track results for the summary
        collected_count = 0
        failed_count = 0
        skipped_null_count = 0
        details: list[dict[str, Any]] = []

        # Open one database connection for all inserts — more efficient than
        # opening/closing for each indicator
        with PostgresStorage() as db:
            for series_id, info in self.indicators.items():
                try:
                    # fredapi.Fred.get_series() returns a pandas Series with
                    # dates as the index and values as floats. We only want
                    # the last (most recent) observation.
                    logger.info("Fetching latest %s (%s) from FRED", info["name"], series_id)
                    series = fred.get_series(series_id)

                    # The series might be empty if FRED has no data
                    if series.empty:
                        logger.warning("FRED returned empty series for %s", series_id)
                        failed_count += 1
                        details.append({"series_id": series_id, "status": "empty_series"})
                        continue

                    # Get the last observation (most recent date)
                    last_date = series.index[-1]
                    last_value = series.iloc[-1]

                    # Some FRED series have NaN for the latest observation
                    # (e.g., data not yet released). Skip those.
                    if _is_null_value(last_value):
                        logger.info(
                            "Latest value for %s (%s) is null — data may not be released yet",
                            info["name"],
                            series_id,
                        )
                        skipped_null_count += 1
                        details.append({"series_id": series_id, "status": "null_value"})
                        continue

                    # Convert the pandas Timestamp to a pendulum DateTime with UTC timezone.
                    # FRED dates are just dates (no time), so we set time to midnight UTC.
                    observation_date = _pandas_timestamp_to_pendulum(last_date)

                    # Convert the float value to Decimal for financial precision.
                    # Floats can't represent some decimals exactly (e.g., 0.1),
                    # but Decimal can. We convert via string to avoid float artifacts.
                    decimal_value = _float_to_decimal(last_value)

                    # Validate through our Pydantic model — this is the bouncer at the door.
                    # If any field is wrong, ValidationError is raised and we catch it below.
                    data_point = DataPointCreate(
                        entity_type="indicator",
                        entity_id=series_id,
                        metric=info["metric"],
                        value=decimal_value,
                        unit=info["unit"],
                        timestamp=observation_date,
                        source="fred",
                    )

                    # Store in PostgreSQL
                    inserted_id = db.insert_data_point(
                        entity_type=data_point.entity_type,
                        entity_id=data_point.entity_id,
                        metric=data_point.metric,
                        value=data_point.value,
                        unit=data_point.unit,
                        timestamp=data_point.timestamp,
                        source=data_point.source,
                    )

                    logger.info(
                        "Stored %s = %s (date: %s, id: %d)",
                        info["name"],
                        decimal_value,
                        observation_date.to_date_string(),
                        inserted_id,
                    )
                    collected_count += 1
                    details.append(
                        {
                            "series_id": series_id,
                            "status": "ok",
                            "value": str(decimal_value),
                            "date": observation_date.to_date_string(),
                            "db_id": inserted_id,
                        }
                    )

                except ValidationError as exc:
                    # Pydantic rejected the data — something is wrong with the values
                    logger.error(
                        "Validation failed for %s (%s): %s",
                        info["name"],
                        series_id,
                        exc,
                    )
                    failed_count += 1
                    details.append(
                        {
                            "series_id": series_id,
                            "status": "validation_error",
                            "error": str(exc),
                        }
                    )

                except Exception as exc:
                    # Catch-all for FRED API errors, network issues, etc.
                    # Log and continue — don't let one bad indicator kill the whole run.
                    logger.error(
                        "Failed to collect %s (%s): %s",
                        info["name"],
                        series_id,
                        exc,
                    )
                    failed_count += 1
                    details.append(
                        {
                            "series_id": series_id,
                            "status": "error",
                            "error": str(exc),
                        }
                    )

        # Build and return the standard summary via the base class helper.
        # "skipped_null" and "details" are passed as extras.
        return self._build_summary(
            stored=collected_count,
            failed=failed_count,
            skipped_null=skipped_null_count,
            details=details,
        )

    def collect_history(self, **kwargs: Any) -> dict[str, Any]:
        """Pull historical data for all indicators and store it.

        Fetches data going back `years` years from today. Useful for
        backfilling data when setting up the system for the first time,
        or when adding a new indicator.

        Each observation becomes a separate data point in PostgreSQL.
        Some series have daily data (Fed funds rate), others monthly (CPI),
        and others quarterly (GDP). The code handles all frequencies the same
        way — we just store every observation FRED gives us.

        Keyword Args:
            years: How many years of history to fetch. Default is 1.

        Returns:
            Summary dict from _build_summary() with stored/failed counts,
            plus extra keys: skipped_null, details.
        """
        # Extract the 'years' keyword arg, defaulting to 1
        years: int = kwargs.get("years", 1)

        # Validate that years is a positive integer.
        # A value of 0 or negative makes no sense as a look-back window,
        # and a non-integer would break pendulum's subtract(years=...) call.
        if not isinstance(years, int) or years < 1:
            logger.warning(
                "Invalid value for 'years': %r — must be a positive integer. Defaulting to 1.",
                years,
            )
            years = 1

        # Check for API key before doing anything
        fred = self._get_fred_client()
        if fred is None:
            return self._build_summary(stored=0, failed=0, skipped_null=0, details=[])

        # Calculate the start date — go back `years` years from today
        end_date = pendulum.now("UTC")
        start_date = end_date.subtract(years=years)

        logger.info(
            "Collecting economic history from %s to %s (%d indicators)",
            start_date.to_date_string(),
            end_date.to_date_string(),
            len(self.indicators),
        )

        # Track results
        collected_count = 0
        failed_count = 0
        skipped_null_count = 0
        details: list[dict[str, Any]] = []

        with PostgresStorage() as db:
            for series_id, info in self.indicators.items():
                try:
                    logger.info(
                        "Fetching history for %s (%s)",
                        info["name"],
                        series_id,
                    )

                    # fredapi accepts Python date objects for the date range.
                    # pendulum DateTime has a .date() method that returns a date.
                    series = fred.get_series(
                        series_id,
                        observation_start=start_date.date(),
                        observation_end=end_date.date(),
                    )

                    if series.empty:
                        logger.warning("FRED returned empty series for %s", series_id)
                        failed_count += 1
                        details.append({"series_id": series_id, "status": "empty_series"})
                        continue

                    # Process every observation in the series
                    series_collected = 0
                    series_null = 0

                    for timestamp, raw_value in series.items():
                        # Skip null/NaN values — FRED sometimes has gaps
                        if _is_null_value(raw_value):
                            series_null += 1
                            continue

                        observation_date = _pandas_timestamp_to_pendulum(timestamp)
                        decimal_value = _float_to_decimal(raw_value)

                        # Validate each data point through Pydantic
                        try:
                            data_point = DataPointCreate(
                                entity_type="indicator",
                                entity_id=series_id,
                                metric=info["metric"],
                                value=decimal_value,
                                unit=info["unit"],
                                timestamp=observation_date,
                                source="fred",
                            )

                            db.insert_data_point(
                                entity_type=data_point.entity_type,
                                entity_id=data_point.entity_id,
                                metric=data_point.metric,
                                value=data_point.value,
                                unit=data_point.unit,
                                timestamp=data_point.timestamp,
                                source=data_point.source,
                            )
                            series_collected += 1

                        except ValidationError as exc:
                            logger.warning(
                                "Validation failed for %s at %s: %s",
                                series_id,
                                observation_date,
                                exc,
                            )
                            failed_count += 1
                        except Exception as exc:
                            logger.warning(
                                "DB insert failed for %s at %s: %s",
                                series_id,
                                observation_date,
                                exc,
                            )
                            failed_count += 1

                    collected_count += series_collected
                    skipped_null_count += series_null

                    logger.info(
                        "Stored %d observations for %s (%d null skipped)",
                        series_collected,
                        info["name"],
                        series_null,
                    )
                    details.append(
                        {
                            "series_id": series_id,
                            "status": "ok",
                            "observations": series_collected,
                            "nulls_skipped": series_null,
                        }
                    )

                except Exception as exc:
                    # Catch-all for API errors, network issues, etc.
                    logger.error(
                        "Failed to collect history for %s (%s): %s",
                        info["name"],
                        series_id,
                        exc,
                    )
                    failed_count += 1
                    details.append(
                        {
                            "series_id": series_id,
                            "status": "error",
                            "error": str(exc),
                        }
                    )

        # Build and return the standard summary via the base class helper
        return self._build_summary(
            stored=collected_count,
            failed=failed_count,
            skipped_null=skipped_null_count,
            details=details,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def collect_economic_data() -> dict[str, Any]:
    """One-liner to collect the latest economic data.

    Creates an EconomicCollector with default indicators and runs
    collect_latest(). Use this from scripts or the daily pipeline:

        from zerofin.data.economic import collect_economic_data
        summary = collect_economic_data()
    """
    collector = EconomicCollector()
    return collector.collect_latest()


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _is_null_value(value: Any) -> bool:
    """Check if a value from FRED is null/NaN.

    FRED returns NaN (not-a-number) for missing data points. Pandas NaN
    has the weird property that it doesn't equal itself (NaN != NaN is True),
    so we use that trick plus a None check.
    """
    if value is None:
        return True
    # NaN != NaN is True in IEEE 754 floating point — this is the standard
    # way to check for NaN without importing math or numpy.
    try:
        return value != value
    except (TypeError, ValueError):
        return False


def _pandas_timestamp_to_pendulum(pandas_timestamp: Any) -> pendulum.DateTime:
    """Convert a pandas Timestamp to a pendulum DateTime in UTC.

    FRED dates are calendar dates (no time component), so we set the
    time to midnight UTC. This keeps all timestamps consistent across
    the system — everything is UTC, always.

    Args:
        pandas_timestamp: A pandas Timestamp from a Series index.

    Returns:
        A pendulum DateTime at midnight UTC on that date.
    """
    # pandas Timestamps have .year, .month, .day attributes just like
    # Python datetime objects.
    return pendulum.datetime(
        pandas_timestamp.year,
        pandas_timestamp.month,
        pandas_timestamp.day,
        tz="UTC",
    )


def _float_to_decimal(value: float) -> Decimal:
    """Convert a float to Decimal via string to avoid floating-point artifacts.

    Why via string? Because:
        Decimal(0.1)   -> Decimal('0.1000000000000000055511151231257827021181583404541015625')
        Decimal("0.1") -> Decimal('0.1')

    The string conversion gives us the clean number we actually want.

    Args:
        value: A float value from a pandas Series.

    Returns:
        The same number as a Decimal with clean representation.

    Raises:
        InvalidOperation: If the value can't be converted to Decimal.
    """
    try:
        return Decimal(str(value))
    except InvalidOperation:
        logger.error("Could not convert value to Decimal: %s", value)
        raise
