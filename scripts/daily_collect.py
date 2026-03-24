"""
Daily pipeline — run this once per day to keep your data current.

Pulls latest prices from yfinance and latest economic data from FRED,
validates everything, and stores it in PostgreSQL.

Later this will also collect news from RSS feeds and trigger
relationship extraction via DeepSeek.

Run manually:
    python scripts/daily_collect.py

Or set up a scheduled task / cron job to run it automatically.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pendulum

from zerofin.data.economic import EconomicCollector
from zerofin.data.news import NewsCollector
from zerofin.data.prices import PriceCollector
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline() -> dict[str, Any]:
    """Run all data collectors and return a combined summary."""
    start_time = pendulum.now("UTC")
    logger.info("Daily pipeline started at %s", start_time.to_iso8601_string())

    # Make sure tables exist before collecting
    with PostgresStorage() as db:
        db.setup_tables()

    results: dict[str, Any] = {
        "started_at": start_time.to_iso8601_string(),
        "collectors": {},
    }

    # --- Prices ---
    try:
        price_collector = PriceCollector()
        price_result = price_collector.collect_latest()
        results["collectors"]["prices"] = price_result
        logger.info(
            "Prices: %d stored, %d failed",
            price_result.get("stored", 0),
            price_result.get("failed", 0),
        )
    except Exception as error:
        logger.error("Price collection crashed: %s", error)
        results["collectors"]["prices"] = {"error": str(error)}

    # --- Economic indicators ---
    try:
        econ_collector = EconomicCollector()
        econ_result = econ_collector.collect_latest()
        results["collectors"]["economic"] = econ_result
        logger.info(
            "Economic: %d stored, %d failed",
            econ_result.get("stored", 0),
            econ_result.get("failed", 0),
        )
    except Exception as error:
        logger.error("Economic collection crashed: %s", error)
        results["collectors"]["economic"] = {"error": str(error)}

    # --- News ---
    # NewsCollector needs an open Neo4j connection to store articles,
    # so we wrap it in a GraphStorage context manager.
    try:
        with GraphStorage() as graph:
            news_collector = NewsCollector(graph=graph)
            news_result = news_collector.collect_latest()
        results["collectors"]["news"] = news_result
        logger.info(
            "News: %d stored, %d duplicates, %d failed",
            news_result.get("stored", 0),
            news_result.get("duplicates", 0),
            news_result.get("failed", 0),
        )
    except Exception as error:
        logger.error("News collection crashed: %s", error)
        results["collectors"]["news"] = {"error": str(error)}

    # --- Summary ---
    end_time = pendulum.now("UTC")
    duration = (end_time - start_time).in_seconds()
    results["finished_at"] = end_time.to_iso8601_string()
    results["duration_seconds"] = duration

    total_stored = sum(
        r.get("stored", 0)
        for r in results["collectors"].values()
        if isinstance(r, dict) and "stored" in r
    )
    total_failed = sum(
        r.get("failed", 0)
        for r in results["collectors"].values()
        if isinstance(r, dict) and "failed" in r
    )

    logger.info("=" * 60)
    logger.info("DAILY PIPELINE COMPLETE")
    logger.info("  Duration: %d seconds", duration)
    logger.info("  Total stored: %d", total_stored)
    logger.info("  Total failed: %d", total_failed)
    logger.info("=" * 60)

    return results


def main() -> None:
    """Entry point — run the pipeline and exit with appropriate code."""
    results = run_pipeline()

    # Exit with error code if any collector crashed
    has_errors = any("error" in r for r in results["collectors"].values() if isinstance(r, dict))
    if has_errors:
        logger.warning("Pipeline finished with errors — check logs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
