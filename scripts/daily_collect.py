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

from zerofin.ai.mentions import (
    build_entity_list,
    build_mention_chain,
    create_mentioned_in_edges,
    format_entity_list,
    validate_mention_ids,
)
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

    # --- Mentions ---
    # Run mention indexer on new articles if news collection stored any.
    news_stored = results["collectors"].get("news", {}).get("stored", 0)
    if news_stored > 0:
        try:
            with GraphStorage() as graph:
                # Skip headline-only articles
                skip_result = graph.run_query(
                    "MATCH (a:Article) "
                    "WHERE a.status = 'raw' AND (a.summary IS NULL OR a.summary = '') "
                    "SET a.status = 'skipped_no_summary' "
                    "RETURN count(a) AS skipped"
                )
                skipped = skip_result[0]["skipped"] if skip_result else 0
                if skipped > 0:
                    logger.info("Skipped %d headline-only articles", skipped)

                # Fetch articles to process
                raw_articles = graph.run_query(
                    "MATCH (a:Article) "
                    "WHERE a.status = 'raw' "
                    "  AND a.summary IS NOT NULL AND a.summary <> '' "
                    "RETURN a.id AS url, a.title AS title, a.summary AS summary, "
                    "       a.source AS source, a.tier AS tier "
                    "ORDER BY a.collected_at DESC"
                )

                if raw_articles:
                    entities = build_entity_list(graph)
                    entity_list_text = format_entity_list(entities)
                    valid_ids = {e["id"] for e in entities}
                    chain = build_mention_chain()

                    # Process in chunks of 20
                    chunk_size = 20
                    total_edges = 0
                    processed = 0

                    for i in range(0, len(raw_articles), chunk_size):
                        chunk = raw_articles[i : i + chunk_size]
                        inputs = [
                            {
                                "entity_list": entity_list_text,
                                "article_text": f"{a['title']}\n\n{a['summary']}",
                            }
                            for a in chunk
                        ]

                        batch_results = chain.batch(
                            inputs, config={"max_concurrency": chunk_size}
                        )

                        for article, result in zip(chunk, batch_results):
                            validated = validate_mention_ids(result, valid_ids)
                            edges = create_mentioned_in_edges(
                                graph, article["url"], validated.mentioned_ids,
                            )
                            total_edges += edges

                        # Update status
                        urls = [a["url"] for a in chunk]
                        graph.run_query(
                            "UNWIND $urls AS url "
                            "MATCH (a:Article {id: url}) "
                            "SET a.status = 'mentions_done', "
                            "    a.mentions_processed_at = datetime()",
                            {"urls": urls},
                        )
                        processed += len(chunk)

                    logger.info(
                        "Mentions: %d articles processed, %d edges created",
                        processed, total_edges,
                    )
                    results["collectors"]["mentions"] = {
                        "processed": processed,
                        "edges_created": total_edges,
                        "skipped": skipped,
                    }
                else:
                    logger.info("Mentions: no articles to process")
        except Exception as error:
            logger.error("Mention processing crashed: %s", error)
            results["collectors"]["mentions"] = {"error": str(error)}
    else:
        logger.info("Mentions: skipped (no new articles)")

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
