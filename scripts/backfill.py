"""
Backfill script — loads historical data into PostgreSQL.

Run this once when setting up the project to load past data.
Safe to re-run, but will create duplicate rows if data already exists
for the same dates. Clear the market_data table first if re-running.

Run with:
    python scripts/backfill.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from zerofin.data.economic import EconomicCollector
from zerofin.data.prices import PriceCollector
from zerofin.storage.postgres import PostgresStorage

# Log everything to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Also log warnings to a file so we can check failures after
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
file_handler = logging.FileHandler(LOG_DIR / "backfill_warnings.log", mode="w")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
logging.getLogger().addHandler(file_handler)


def backfill_prices(period: str = "1y") -> dict[str, Any]:
    """Load historical price data for all tracked tickers."""
    logger.info("=" * 60)
    logger.info("BACKFILL: Loading %s of price history", period)
    logger.info("=" * 60)

    collector = PriceCollector()
    return collector.collect_history(period=period)


def backfill_economic(years: int = 1) -> dict[str, Any]:
    """Load historical economic data for all FRED indicators."""
    logger.info("=" * 60)
    logger.info("BACKFILL: Loading %d year(s) of FRED history", years)
    logger.info("=" * 60)

    collector = EconomicCollector()
    return collector.collect_history(years=years)


def main() -> None:
    """Run the full backfill."""
    logger.info("Starting Zerofin historical data backfill...")

    # Make sure tables exist
    with PostgresStorage() as db:
        db.setup_tables()

    # Load price history
    price_result = backfill_prices(period="10y")
    logger.info(
        "Prices: %d stored, %d failed",
        price_result.get("stored", 0),
        price_result.get("failed", 0),
    )

    # Load economic history
    econ_result = backfill_economic(years=10)
    logger.info(
        "Economic: %d stored, %d failed",
        econ_result.get("stored", 0),
        econ_result.get("failed", 0),
    )

    # Summary
    total_stored = price_result.get("stored", 0) + econ_result.get("stored", 0)
    total_failed = price_result.get("failed", 0) + econ_result.get("failed", 0)

    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("  Total stored: %d", total_stored)
    logger.info("  Total failed: %d", total_failed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
