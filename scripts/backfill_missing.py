"""
One-off script to backfill tickers that failed during the main backfill.

yfinance sometimes fails on large batch downloads but works fine
when pulling tickers individually. This script handles the stragglers.

Run with:
    python scripts/backfill_missing.py
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal, InvalidOperation

import pendulum
import yfinance
from pydantic import ValidationError

from zerofin.models.entities import DataPointCreate
from zerofin.storage.postgres import PostgresStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Tickers that failed during the 3-year backfill
MISSING_TICKERS = [
    "GOOGL", "GC=F", "BZ=F", "ORCL", "MRVL", "RTX",
    "SLB", "XLP", "VXUS", "KSA", "^W5000",
]

# Standard price precision (4 decimal places)
PRICE_DECIMAL_PLACES = Decimal("0.0001")


def main() -> None:
    """Download and store missing tickers one at a time."""
    logger.info("Backfilling %d missing tickers...", len(MISSING_TICKERS))

    with PostgresStorage() as db:
        total_stored = 0
        total_failed = 0

        for ticker in MISSING_TICKERS:
            try:
                data = yfinance.download(ticker, period="3y", progress=False)

                # Flatten MultiIndex columns if present (yfinance quirk)
                if hasattr(data.columns, "levels"):
                    data.columns = data.columns.get_level_values(0)

                if data.empty:
                    logger.warning("%s: no data returned — skipping", ticker)
                    total_failed += 1
                    continue

                # Validate all rows first, then batch insert per ticker
                batch: list[dict] = []
                for date, row in data.iterrows():
                    close = row.get("Close")

                    if close is None or (isinstance(close, float) and math.isnan(close)):
                        continue

                    try:
                        timestamp = pendulum.instance(date.to_pydatetime())
                        close_decimal = Decimal(str(close)).quantize(PRICE_DECIMAL_PLACES)

                        data_point = DataPointCreate(
                            entity_type="asset",
                            entity_id=ticker,
                            metric="close_price",
                            value=close_decimal,
                            timestamp=timestamp,
                            source="yfinance",
                            unit="USD",
                        )

                        batch.append({
                            "entity_type": data_point.entity_type,
                            "entity_id": data_point.entity_id,
                            "metric": data_point.metric,
                            "value": data_point.value,
                            "unit": data_point.unit,
                            "timestamp": data_point.timestamp,
                            "source": data_point.source,
                            "is_revised": False,
                            "revision_of": None,
                        })

                    except (InvalidOperation, ValidationError) as error:
                        logger.warning(
                            "Validation failed for %s on %s: %s",
                            ticker, date, error,
                        )

                stored = db.insert_data_points_batch(batch)
                logger.info("%s: stored %d data points", ticker, stored)
                total_stored += stored

            except Exception as error:
                logger.error("%s: failed — %s", ticker, error)
                total_failed += 1

    logger.info("=" * 60)
    logger.info("MISSING TICKER BACKFILL COMPLETE")
    logger.info("  Total stored: %d", total_stored)
    logger.info("  Failed tickers: %d", total_failed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
