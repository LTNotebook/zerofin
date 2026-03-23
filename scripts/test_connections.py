"""
Test script — verifies that both databases are working and our code can talk to them.

This is NOT a pytest test (those go in tests/). This is a runnable script you
can execute anytime to make sure everything is connected and functioning.

Run it with:
    python scripts/test_connections.py
"""

from __future__ import annotations

import logging
from decimal import Decimal

import pendulum

from zerofin.config import settings
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

# Set up logging so we can see what's happening.
# level=INFO means we see the important stuff but not every tiny detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_config() -> None:
    """Check that config.py loaded our .env file correctly."""
    logger.info("=" * 60)
    logger.info("STEP 1: Testing config.py")
    logger.info("=" * 60)

    # These should match what's in your .env file
    logger.info("  Postgres host: %s", settings.POSTGRES_HOST)
    logger.info("  Postgres port: %s", settings.POSTGRES_PORT)
    logger.info("  Postgres database: %s", settings.POSTGRES_DB)
    logger.info("  Neo4j URI: %s", settings.NEO4J_URI)

    # Check that passwords aren't empty — common mistake if .env isn't filled in
    if not settings.POSTGRES_PASSWORD:
        logger.warning("  POSTGRES_PASSWORD is empty — did you fill in .env?")
    else:
        logger.info("  Postgres password: loaded (not shown for security)")

    if not settings.NEO4J_PASSWORD:
        logger.warning("  NEO4J_PASSWORD is empty — did you fill in .env?")
    else:
        logger.info("  Neo4j password: loaded (not shown for security)")

    logger.info("  Config test PASSED")


def test_postgres() -> None:
    """Connect to Postgres, create tables, insert a test row, read it back."""
    logger.info("=" * 60)
    logger.info("STEP 2: Testing PostgreSQL")
    logger.info("=" * 60)

    # 'with' automatically connects and disconnects when done
    with PostgresStorage() as db:
        # Create the market_data table (safe to run multiple times)
        db.setup_tables()
        logger.info("  Tables created successfully")

        # Insert a fake data point to prove writes work
        test_time = pendulum.now("UTC")
        inserted_id = db.insert_data_point(
            entity_type="asset",
            entity_id="TEST_TICKER",
            metric="close_price",
            value=Decimal("123.45"),
            timestamp=test_time,
            source="test_script",
            unit="USD",
        )
        logger.info("  Inserted test data point with id=%d", inserted_id)

        # Read it back to prove reads work
        rows = db.get_latest_market_data(
            entity_type="asset",
            entity_id="TEST_TICKER",
            metric="close_price",
            limit=1,
        )

        # Check we got something back
        if len(rows) == 0:
            logger.error("  FAILED — no data returned after insert!")
            return

        # Check the value matches what we put in
        row = rows[0]
        saved_value = row["value"]
        logger.info("  Read back value: %s", saved_value)

        if saved_value == Decimal("123.45"):
            logger.info("  Value matches — Postgres read/write PASSED")
        else:
            logger.error("  FAILED — expected 123.45, got %s", saved_value)


def test_neo4j() -> None:
    """Connect to Neo4j, create indexes, add a test entity, read it back."""
    logger.info("=" * 60)
    logger.info("STEP 3: Testing Neo4j")
    logger.info("=" * 60)

    # 'with' automatically connects and disconnects when done
    with GraphStorage() as graph:
        # Create indexes for all entity types (safe to run multiple times)
        graph.setup_indexes()
        logger.info("  Indexes created successfully")

        # Create a test entity to prove writes work
        test_entity = {
            "id": "TEST_ENTITY",
            "name": "Test Company",
            "type": "stock",
        }
        result = graph.create_entity("Company", test_entity)
        logger.info("  Created test entity: %s", result)

        # Read it back to prove reads work
        found = graph.get_entity("Company", "TEST_ENTITY")

        if found is None:
            logger.error("  FAILED — entity not found after creation!")
            return

        logger.info("  Read back entity: %s", found["name"])

        if found["name"] == "Test Company":
            logger.info("  Entity matches — Neo4j read/write PASSED")
        else:
            logger.error("  FAILED — expected 'Test Company', got '%s'", found["name"])


def test_yfinance() -> None:
    """Pull a real stock price from yfinance to prove the internet works."""
    logger.info("=" * 60)
    logger.info("STEP 4: Testing yfinance (real stock price)")
    logger.info("=" * 60)

    # Import here so the other tests still work even if yfinance has issues
    import yfinance

    # Pull NVIDIA's data — just the last day
    ticker = yfinance.Ticker("NVDA")
    history = ticker.history(period="1d")

    if history.empty:
        logger.error("  FAILED — no data returned from yfinance")
        return

    # Get the closing price from the last row
    close_price = history["Close"].iloc[-1]
    logger.info("  NVDA latest close price: $%.2f", close_price)
    logger.info("  yfinance test PASSED")


def main() -> None:
    """Run all tests in order."""
    logger.info("Starting Zerofin connection tests...")
    logger.info("")

    test_config()
    logger.info("")

    test_postgres()
    logger.info("")

    test_neo4j()
    logger.info("")

    test_yfinance()
    logger.info("")

    logger.info("=" * 60)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
