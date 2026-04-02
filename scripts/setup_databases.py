"""Set up all database tables, indexes, and constraints.

Run this once when setting up a new environment, or after adding
new tables to the schema. Safe to run multiple times — everything
uses IF NOT EXISTS.

Usage:
    python scripts/setup_databases.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zerofin.storage.graph import GraphStorage  # noqa: E402
from zerofin.storage.postgres import PostgresStorage  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Setting up databases...")

    # PostgreSQL — market_data, verification_runs, verification_results
    with PostgresStorage() as db:
        db.setup_tables()
    logger.info("PostgreSQL setup complete")

    # Neo4j — uniqueness constraints and indexes for all entity types
    with GraphStorage() as graph:
        graph.setup_indexes()
    logger.info("Neo4j setup complete")

    logger.info("All databases ready")


if __name__ == "__main__":
    main()
