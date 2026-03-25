"""
One-off script to backfill tickers that failed during the main backfill.

yfinance sometimes fails on large batch downloads but works fine
when pulling tickers individually. This script handles the stragglers.

Run with:
    python scripts/backfill_missing.py
"""

from __future__ import annotations

import logging
from decimal import Decimal

import pendulum
import yfinance as yf

from zerofin.storage.postgres import PostgresStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Tickers that failed during the 3-year backfill
MISSING_TICKERS = [
    "GOOGL", "GC=F", "BZ=F", "ORCL", "MRVL", "RTX",
    "SLB", "XLP", "VXUS", "KSA", "^W5000", "^IRX",
]


def main() -> None:
    """Download and store missing tickers one at a time."""
    logger.info("Backfilling %d missing tickers...", len(MISSING_TICKERS))

    with PostgresStorage() as db:
        total_stored = 0
        total_failed = 0

        for ticker in MISSING_TICKERS:
            try:
                data = yf.download(ticker, period="3y", progress=False)

                # Flatten MultiIndex columns if present (yfinance quirk)
                if hasattr(data.columns, "levels"):
                    data.columns = data.columns.get_level_values(0)

                if data.empty:
                    logger.warning("%s: no data returned — skipping", ticker)
                    total_failed += 1
                    continue

                stored = 0
                for date, row in data.iterrows():
                    close = row.get("Close")

                    if close is not None and not (float(close) != float(close)):
                        ts = pendulum.instance(date.to_pydatetime())
                        db.insert_data_point(
                            entity_type="asset",
                            entity_id=ticker,
                            metric="close_price",
                            value=Decimal(str(round(float(close), 4))),
                            timestamp=ts,
                            source="yfinance",
                            unit="USD",
                        )
                        stored += 1

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
