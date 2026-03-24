"""
Run the correlation engine — discovers long-term statistical relationships.

Pulls historical data from PostgreSQL, calculates pairwise correlations
across all tracked entities, and stores the significant ones in Neo4j
as CORRELATES_WITH candidate relationships.

This runs independently from the daily pipeline. Run it weekly or
whenever you want fresh correlation analysis.

Run with:
    python scripts/run_correlations.py
"""

from __future__ import annotations

import logging
import sys

from zerofin.analysis.correlations import run_correlation_pipeline
from zerofin.config import settings
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the correlation engine for each configured window size."""
    logger.info("=" * 60)
    logger.info("ZEROFIN CORRELATION ENGINE")
    logger.info("=" * 60)
    logger.info("Windows: %s days", settings.CORRELATION_WINDOWS)
    logger.info("Min observations ratio: %.0f%%", settings.CORRELATION_MIN_OBSERVATIONS_RATIO * 100)
    logger.info("FDR alpha: %.2f", settings.CORRELATION_FDR_ALPHA)
    logger.info("Stability filter: %s", settings.CORRELATION_STABILITY_FILTER)
    logger.info("Spearman check: %s", settings.CORRELATION_SPEARMAN_CHECK)
    logger.info("")

    all_summaries = []
    has_errors = False

    with PostgresStorage() as db, GraphStorage() as graph:
        for window in settings.CORRELATION_WINDOWS:
            logger.info("-" * 60)
            logger.info("Running %d-day window", window)
            logger.info("-" * 60)

            try:
                summary = run_correlation_pipeline(db, graph, window_days=window)
                all_summaries.append(summary)

                logger.info("  Pairs tested:      %d", summary.total_pairs_tested)
                logger.info("  Above threshold:   %d", summary.pairs_above_threshold)
                logger.info("  Survived FDR:      %d", summary.pairs_surviving_fdr)
                logger.info("  Stored in Neo4j:   %d", summary.relationships_stored)
                logger.info("  Duration:          %.1fs", summary.duration_seconds)
                logger.info("")

            except Exception as error:
                logger.error("Correlation engine failed for %d-day window: %s", window, error)
                has_errors = True

    # Final summary
    logger.info("=" * 60)
    logger.info("CORRELATION ENGINE COMPLETE")
    logger.info("=" * 60)

    total_stored = sum(s.relationships_stored for s in all_summaries)
    total_duration = sum(s.duration_seconds for s in all_summaries)
    logger.info("  Total relationships stored: %d", total_stored)
    logger.info("  Total duration: %.1fs", total_duration)

    if has_errors:
        logger.warning("  Some windows had errors — check logs above")
        sys.exit(1)

    logger.info("  All windows completed successfully")


if __name__ == "__main__":
    main()
