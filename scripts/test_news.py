"""
Test the news collector — pulls from all RSS feeds and shows what we get.

Run with:
    python scripts/test_news.py
"""

from __future__ import annotations

import logging

from zerofin.data.news import RSS_FEEDS, NewsCollector
from zerofin.storage.graph import GraphStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Pull from all active RSS feeds and report what we got."""
    feeds = RSS_FEEDS
    logger.info("Testing %d RSS feeds...", len(feeds))

    with GraphStorage() as graph:
        collector = NewsCollector(graph=graph, feeds=feeds)
        result = collector.collect_latest()

    logger.info("")
    logger.info("=" * 60)
    logger.info("NEWS COLLECTION RESULTS")
    logger.info("=" * 60)
    logger.info("  Total stored: %d", result.get("stored", 0))
    logger.info("  Duplicates skipped: %d", result.get("duplicates", 0))
    logger.info("  Failed: %d", result.get("failed", 0))
    logger.info("")
    logger.info("Per feed breakdown:")

    for fr in result.get("feed_results", []):
        logger.info(
            "  %-40s  stored: %3d  dupes: %3d  failed: %d",
            fr["feed"],
            fr["stored"],
            fr["duplicates"],
            fr["failed"],
        )

    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
