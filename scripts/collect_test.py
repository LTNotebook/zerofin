"""
Test script — runs both data collectors to verify they work.

Pulls real stock prices from yfinance and real economic data from FRED,
validates everything through Pydantic, and stores it in PostgreSQL.

After running this, check Adminer (localhost:8080) to see the data
sitting in your market_data table.

Run with:
    python scripts/collect_test.py
"""

from __future__ import annotations

import logging

from zerofin.data.economic import EconomicCollector
from zerofin.data.prices import PriceCollector
from zerofin.storage.postgres import PostgresStorage

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_prices() -> None:
    """Pull latest stock prices and store them in PostgreSQL."""
    logger.info("=" * 60)
    logger.info("TEST 1: Collecting stock prices from yfinance")
    logger.info("=" * 60)

    try:
        collector = PriceCollector()
        result = collector.collect_latest()

        logger.info("  Results:")
        logger.info("    Stored: %d data points", result.get("stored", 0))
        logger.info("    Failed: %d data points", result.get("failed", 0))
        logger.info("    Tickers OK: %s", result.get("tickers_ok", []))
        logger.info("    Tickers failed: %s", result.get("tickers_failed", []))
        logger.info("  Price collection PASSED")

    except Exception as error:
        logger.error("  Price collection FAILED: %s", error)


def test_economic() -> None:
    """Pull latest economic indicators from FRED and store them."""
    logger.info("=" * 60)
    logger.info("TEST 2: Collecting economic data from FRED")
    logger.info("=" * 60)

    try:
        collector = EconomicCollector()
        result = collector.collect_latest()

        logger.info("  Results:")
        logger.info("    Collected: %d data points", result.get("collected", 0))
        logger.info("    Failed: %d indicators", result.get("failed", 0))
        logger.info("  Economic collection PASSED")

    except Exception as error:
        logger.error("  Economic collection FAILED: %s", error)


def show_database_summary() -> None:
    """Show how many data points are now in PostgreSQL."""
    logger.info("=" * 60)
    logger.info("DATABASE SUMMARY")
    logger.info("=" * 60)

    try:
        with PostgresStorage() as db:
            # Count total data points
            with db._conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM market_data")
                row = cursor.fetchone()
                total = row["total"] if row else 0

            # Count by entity type
            with db._conn.cursor() as cursor:
                cursor.execute(
                    "SELECT entity_type, COUNT(*) AS count FROM market_data GROUP BY entity_type"
                )
                rows = cursor.fetchall()

            logger.info("  Total data points in database: %d", total)
            for row in rows:
                logger.info("    %s: %d", row["entity_type"], row["count"])

    except Exception as error:
        logger.error("  Could not read database: %s", error)


def main() -> None:
    """Run all collection tests."""
    logger.info("Starting Zerofin data collection test...")
    logger.info("")

    # Make sure tables exist
    with PostgresStorage() as db:
        db.setup_tables()

    test_prices()
    logger.info("")

    test_economic()
    logger.info("")

    show_database_summary()
    logger.info("")

    logger.info("=" * 60)
    logger.info("COLLECTION TEST COMPLETE")
    logger.info("=" * 60)
    logger.info("Check Adminer at http://localhost:8080 to see the data!")


if __name__ == "__main__":
    main()
