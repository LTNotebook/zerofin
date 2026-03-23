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
from zerofin.data.news import NewsCollector
from zerofin.data.prices import PriceCollector
from zerofin.storage.graph import GraphStorage
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


def test_news() -> None:
    """Pull latest news from RSS feeds and store in Neo4j."""
    logger.info("=" * 60)
    logger.info("TEST 3: Collecting news from RSS feeds")
    logger.info("=" * 60)

    try:
        # Only test with first 3 feeds to keep it quick
        from zerofin.data.news import RSS_FEEDS

        test_feeds = RSS_FEEDS[:3]
        with GraphStorage() as graph:
            collector = NewsCollector(graph=graph, feeds=test_feeds)
            result = collector.collect_latest()

        logger.info("  Results:")
        logger.info("    Stored: %d articles", result.get("stored", 0))
        logger.info("    Failed: %d feeds", result.get("failed", 0))
        logger.info("    Duplicates skipped: %d", result.get("duplicates_skipped", 0))
        logger.info("  News collection PASSED")

    except Exception as error:
        logger.error("  News collection FAILED: %s", error)


def show_database_summary() -> None:
    """Show how many data points are now in PostgreSQL and Neo4j."""
    logger.info("=" * 60)
    logger.info("DATABASE SUMMARY")
    logger.info("=" * 60)

    try:
        with PostgresStorage() as db:
            total_rows = db.execute_query(
                "SELECT COUNT(*) AS total FROM market_data"
            )
            total = total_rows[0]["total"] if total_rows else 0

            type_rows = db.execute_query(
                "SELECT entity_type, COUNT(*) AS count "
                "FROM market_data GROUP BY entity_type"
            )

            logger.info("  PostgreSQL market_data: %d rows", total)
            for row in type_rows:
                logger.info("    %s: %d", row["entity_type"], row["count"])

    except Exception as error:
        logger.error("  Could not read PostgreSQL: %s", error)

    try:
        with GraphStorage() as graph:
            article_count = graph.run_query(
                "MATCH (a:Article) RETURN count(a) AS total"
            )
            total_articles = article_count[0]["total"] if article_count else 0
            logger.info("  Neo4j articles: %d", total_articles)

    except Exception as error:
        logger.error("  Could not read Neo4j: %s", error)


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

    test_news()
    logger.info("")

    show_database_summary()
    logger.info("")

    logger.info("=" * 60)
    logger.info("COLLECTION TEST COMPLETE")
    logger.info("=" * 60)
    logger.info("Check Adminer at http://localhost:8080 to see the data!")


if __name__ == "__main__":
    main()
