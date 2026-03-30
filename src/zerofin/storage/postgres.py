"""
PostgreSQL storage module — handles all Postgres interactions for Zerofin.

This is the "time-series side" of the system. It stores price data, economic
indicators, and system settings. The other database (Neo4j) handles the
knowledge graph and article storage.

Uses psycopg v3 (the modern async-capable PostgreSQL driver for Python).
All queries use parameterized placeholders (%s) to prevent SQL injection.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from types import TracebackType
from typing import Any

import pendulum
import psycopg
from psycopg.rows import dict_row

from zerofin.config import settings

# Set up logger — every module gets its own logger named after the module path.
# This way log messages automatically show which file they came from.
logger = logging.getLogger(__name__)

# Metric excluded from bulk correlation queries — volume data adds noise
# to price-based correlation analysis and should be skipped.
EXCLUDED_CORRELATION_METRIC = "volume"


# ---------------------------------------------------------------------------
# SQL statements — kept as constants so they're easy to find and update.
# Using triple-quoted strings for readability.
# ---------------------------------------------------------------------------

# The main table for all numeric data points (prices, indicators, rates, etc.).
# "entity_type" + "entity_id" together identify WHAT this data is about.
# "metric" says WHICH measurement it is (close_price, volume, value, etc.).
CREATE_DATA_POINTS_TABLE = """
CREATE TABLE IF NOT EXISTS market_data (
    id              SERIAL PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           DECIMAL NOT NULL,
    unit            TEXT,
    timestamp       TIMESTAMP NOT NULL,
    collected_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL,
    is_revised      BOOLEAN DEFAULT FALSE,
    revision_of     INTEGER REFERENCES market_data(id)
);
"""

# Composite index for the most common query pattern: "give me all data points
# for entity X between date A and date B". Without this index, Postgres would
# have to scan every row in the table — with it, lookups are near-instant.
CREATE_DATA_POINTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_market_data_entity_time
ON market_data (entity_type, entity_id, timestamp);
"""

# Unique constraint to prevent duplicate data points for the same entity/metric/time.
# Enables ON CONFLICT upsert so re-running backfills updates rather than duplicates.
CREATE_DATA_POINTS_UNIQUE = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_unique
ON market_data (entity_type, entity_id, metric, timestamp);
"""

# Partial index for correlation queries — covers the common access pattern
# (all non-volume data in a date range) without indexing volume rows.
CREATE_DATA_POINTS_PARTIAL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_market_data_correlation
ON market_data (timestamp, entity_type, entity_id)
WHERE metric != 'volume';
"""

# Insert a single data point and return the generated id so we can reference it.
INSERT_DATA_POINT = """
INSERT INTO market_data (
    entity_type, entity_id, metric, value, unit,
    timestamp, source, is_revised, revision_of
)
VALUES (
    %(entity_type)s, %(entity_id)s, %(metric)s, %(value)s, %(unit)s,
    %(timestamp)s, %(source)s, %(is_revised)s, %(revision_of)s
)
RETURNING id;
"""

# Get the N most recent data points for a given entity + metric.
# ORDER BY timestamp DESC gives us newest-first, LIMIT caps the count.
SELECT_LATEST_DATA_POINTS = """
SELECT *
FROM market_data
WHERE entity_type = %(entity_type)s
  AND entity_id   = %(entity_id)s
  AND metric      = %(metric)s
ORDER BY timestamp DESC
LIMIT %(limit)s;
"""

# Get all data points for an entity + metric within a date range (inclusive).
# Ordered oldest-first so the data is in chronological order for analysis.
SELECT_DATA_POINTS_RANGE = """
SELECT *
FROM market_data
WHERE entity_type = %(entity_type)s
  AND entity_id   = %(entity_id)s
  AND metric      = %(metric)s
  AND timestamp >= %(start)s
  AND timestamp <= %(end)s
ORDER BY timestamp ASC;
"""


# ---------------------------------------------------------------------------
# Verification audit trail — tracks every LLM verification pipeline run
# and the individual results for each relationship evaluated.
# ---------------------------------------------------------------------------

# One row per pipeline run — the summary view.
CREATE_VERIFICATION_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS verification_runs (
    id                  SERIAL PRIMARY KEY,
    started_at          TIMESTAMP NOT NULL,
    finished_at         TIMESTAMP NOT NULL,
    duration_seconds    DECIMAL NOT NULL,
    total_pairs         INTEGER NOT NULL,
    promoted            INTEGER NOT NULL,
    rejected            INTEGER NOT NULL,
    uncertain           INTEGER NOT NULL,
    pass1_model         TEXT NOT NULL,
    pass2_model         TEXT
);
"""

# One row per relationship per run — the full LLM output.
CREATE_VERIFICATION_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS verification_results (
    id                      SERIAL PRIMARY KEY,
    run_id                  INTEGER NOT NULL REFERENCES verification_runs(id),
    entity_a_id             TEXT NOT NULL,
    entity_b_id             TEXT NOT NULL,
    method                  TEXT NOT NULL,
    window_days             INTEGER NOT NULL,
    correlation             DECIMAL NOT NULL,
    direction               TEXT NOT NULL,
    verdict                 TEXT NOT NULL,
    confidence              DECIMAL NOT NULL,
    mechanism               TEXT NOT NULL,
    alternative_explanations TEXT NOT NULL,
    reasoning               TEXT NOT NULL,
    relationship_category   TEXT NOT NULL
);
"""

# Index for the most common query: "show me all results for a specific entity"
CREATE_VERIFICATION_RESULTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_verification_results_entity
ON verification_results (entity_a_id, entity_b_id);
"""

# Index for looking up all results from a specific run
CREATE_VERIFICATION_RESULTS_RUN_INDEX = """
CREATE INDEX IF NOT EXISTS idx_verification_results_run
ON verification_results (run_id);
"""


class PostgresStorage:
    """Manages the PostgreSQL connection and provides methods to read/write data.

    Usage as a context manager (recommended — automatically connects and cleans up):

        with PostgresStorage() as db:
            db.insert_data_point(...)

    Or manually:

        db = PostgresStorage()
        db.connect()
        ...
        db.close()
    """

    def __init__(self) -> None:
        # The connection object — None until connect() is called.
        # psycopg.Connection is the v3 synchronous connection type.
        self._connection: psycopg.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a connection to PostgreSQL using credentials from settings.

        Credentials are passed as individual keyword arguments rather than a
        URL string so the password is never concatenated into a readable value.
        """
        if self._connection is not None and not self._connection.closed:
            logger.debug("Already connected to PostgreSQL — skipping reconnect")
            return

        logger.info(
            "Connecting to PostgreSQL at %s:%s",
            settings.POSTGRES_HOST,
            settings.POSTGRES_PORT,
        )

        # psycopg v3 uses psycopg.connect() (not psycopg2.connect()).
        # row_factory=dict_row makes every row come back as a dict instead of
        # a tuple, so we can access columns by name: row["entity_id"].
        # autocommit=False (the default) means we need to call conn.commit()
        # after writes — this is safer because we can roll back on errors.
        # connect_timeout prevents hanging if Postgres is unreachable.
        try:
            self._connection = psycopg.connect(
                **settings.postgres_connection_params(),
                row_factory=dict_row,
                connect_timeout=30,
            )
        except psycopg.OperationalError:
            logger.warning("PostgreSQL connection failed — retrying in 2s")
            time.sleep(2)
            self._connection = psycopg.connect(
                **settings.postgres_connection_params(),
                row_factory=dict_row,
                connect_timeout=30,
            )

        logger.info("Connected to PostgreSQL database '%s'", settings.POSTGRES_DB)

    def close(self) -> None:
        """Close the PostgreSQL connection and release resources."""
        if self._connection is not None and not self._connection.closed:
            self._connection.close()
            logger.info("PostgreSQL connection closed")
        self._connection = None

    # ------------------------------------------------------------------
    # Context manager — lets us use `with PostgresStorage() as db:`
    # ------------------------------------------------------------------

    def __enter__(self) -> PostgresStorage:
        """Called when entering a `with` block — connects automatically."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Called when leaving a `with` block — closes the connection.

        If an exception happened inside the block, the connection is still
        closed cleanly (psycopg rolls back uncommitted transactions on close).
        """
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> psycopg.Connection:
        """Get the active connection or raise if not connected.

        This is a convenience property so every method doesn't have to
        repeat the same "is the connection open?" check.
        """
        if self._connection is None or self._connection.closed:
            raise RuntimeError(
                "Not connected to PostgreSQL. Call connect() first or use "
                "PostgresStorage as a context manager."
            )
        return self._connection

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup_tables(self) -> None:
        """Create all required tables and indexes if they don't already exist.

        Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS, so it
        won't destroy existing data. Run this once at project setup, or at the
        start of every pipeline run as a safety net.
        """
        logger.info("Setting up PostgreSQL tables and indexes")

        # cursor() opens a database cursor for executing queries.
        # The `with` block auto-closes the cursor when done.
        with self._conn.cursor() as cursor:
            # Market data
            cursor.execute(CREATE_DATA_POINTS_TABLE)
            cursor.execute(CREATE_DATA_POINTS_INDEX)
            cursor.execute(CREATE_DATA_POINTS_UNIQUE)
            cursor.execute(CREATE_DATA_POINTS_PARTIAL_INDEX)
            # Verification audit trail
            cursor.execute(CREATE_VERIFICATION_RUNS_TABLE)
            cursor.execute(CREATE_VERIFICATION_RESULTS_TABLE)
            cursor.execute(CREATE_VERIFICATION_RESULTS_INDEX)
            cursor.execute(CREATE_VERIFICATION_RESULTS_RUN_INDEX)

        # Commit the transaction — without this, the table creation would be
        # rolled back when the connection closes.
        self._conn.commit()

        logger.info("PostgreSQL tables and indexes are ready")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert_data_point(
        self,
        *,
        entity_type: str,
        entity_id: str,
        metric: str,
        value: Decimal | float,
        timestamp: pendulum.DateTime,
        source: str,
        unit: str | None = None,
        is_revised: bool = False,
        revision_of: int | None = None,
    ) -> int:
        """Store a single price or indicator data point in the database.

        Uses keyword-only arguments (the * in the signature) to prevent
        accidentally mixing up the order. You MUST name every parameter:

            db.insert_data_point(
                entity_type="asset",
                entity_id="NVDA",
                metric="close_price",
                value=Decimal("125.50"),
                timestamp=pendulum.parse("2025-01-15T16:00:00"),
                source="yfinance",
                unit="USD",
            )

        Returns:
            The auto-generated id of the inserted row.
        """
        logger.debug(
            "Inserting data point: %s/%s/%s = %s at %s",
            entity_type,
            entity_id,
            metric,
            value,
            timestamp,
        )

        # Build the parameter dict that maps to the %(name)s placeholders
        # in the INSERT_DATA_POINT query. psycopg handles escaping and type
        # conversion automatically — no SQL injection risk.
        params = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "metric": metric,
            "value": value,
            "unit": unit,
            "timestamp": timestamp,
            "source": source,
            "is_revised": is_revised,
            "revision_of": revision_of,
        }

        with self._conn.cursor() as cursor:
            cursor.execute(INSERT_DATA_POINT, params)

            # RETURNING id gives us back the generated primary key.
            # fetchone() gets a single row; since we use dict_row, it's a dict.
            row = cursor.fetchone()

        # Commit so the data is actually persisted to disk.
        self._conn.commit()

        if row is None:
            raise RuntimeError("INSERT did not return an id — this should never happen")

        inserted_id: int = row["id"]
        logger.debug("Inserted data point with id=%d", inserted_id)
        return inserted_id

    def insert_data_points_batch(
        self,
        data_points: list[dict],
    ) -> int:
        """Store many data points in a single database commit.

        Much faster than calling insert_data_point() in a loop —
        sends all rows in one executemany() call with one commit
        instead of hundreds of thousands of individual commits.

        Uses ON CONFLICT to handle duplicates gracefully — if a data
        point already exists for the same entity/metric/timestamp,
        the value and source are updated instead of raising an error.

        Each dict in data_points should have:
            entity_type, entity_id, metric, value, unit,
            timestamp, source, is_revised, revision_of

        Returns:
            Number of rows inserted or updated.
        """
        if not data_points:
            return 0

        query = """
            INSERT INTO market_data (
                entity_type, entity_id, metric, value, unit,
                timestamp, source, is_revised, revision_of
            )
            VALUES (
                %(entity_type)s, %(entity_id)s, %(metric)s,
                %(value)s, %(unit)s, %(timestamp)s,
                %(source)s, %(is_revised)s, %(revision_of)s
            )
            ON CONFLICT (entity_type, entity_id, metric, timestamp)
            DO UPDATE SET
                value = EXCLUDED.value,
                source = EXCLUDED.source,
                collected_at = NOW()
        """

        try:
            with self._conn.cursor() as cursor:
                cursor.executemany(query, data_points)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.error(
                "Batch insert failed for %d data points — rolled back",
                len(data_points),
            )
            raise

        count = len(data_points)
        logger.info("Batch inserted %d data points", count)
        return count

    # ------------------------------------------------------------------
    # Verification audit trail
    # ------------------------------------------------------------------

    def insert_verification_run(
        self,
        *,
        started_at: str,
        finished_at: str,
        duration_seconds: float,
        total_pairs: int,
        promoted: int,
        rejected: int,
        uncertain: int,
        pass1_model: str,
        pass2_model: str | None = None,
    ) -> int:
        """Record a verification pipeline run and return its id.

        Args:
            started_at: ISO 8601 timestamp when the run started.
            finished_at: ISO 8601 timestamp when the run finished.
            duration_seconds: How long the run took.
            total_pairs: Total relationships evaluated.
            promoted: Count promoted to candidate.
            rejected: Count rejected.
            uncertain: Count still uncertain.
            pass1_model: Model used for first pass (e.g. "deepseek/deepseek-chat").
            pass2_model: Model used for second pass, if any.

        Returns:
            The auto-generated id of the run row.
        """
        query = """
            INSERT INTO verification_runs (
                started_at, finished_at, duration_seconds,
                total_pairs, promoted, rejected, uncertain,
                pass1_model, pass2_model
            )
            VALUES (
                %(started_at)s, %(finished_at)s, %(duration_seconds)s,
                %(total_pairs)s, %(promoted)s, %(rejected)s, %(uncertain)s,
                %(pass1_model)s, %(pass2_model)s
            )
            RETURNING id;
        """

        params = {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "total_pairs": total_pairs,
            "promoted": promoted,
            "rejected": rejected,
            "uncertain": uncertain,
            "pass1_model": pass1_model,
            "pass2_model": pass2_model,
        }

        with self._conn.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

        self._conn.commit()

        run_id: int = row["id"]
        logger.info("Recorded verification run id=%d (%d pairs)", run_id, total_pairs)
        return run_id

    def insert_verification_results(
        self,
        run_id: int,
        results: list[dict],
    ) -> int:
        """Store the detailed LLM output for each relationship in a run.

        Args:
            run_id: The id from insert_verification_run().
            results: List of result dicts (same format as the CSV rows).

        Returns:
            Number of rows inserted.
        """
        if not results:
            return 0

        query = """
            INSERT INTO verification_results (
                run_id, entity_a_id, entity_b_id, method, window_days,
                correlation, direction, verdict, confidence,
                mechanism, alternative_explanations, reasoning,
                relationship_category
            )
            VALUES (
                %(run_id)s, %(entity_a_id)s, %(entity_b_id)s,
                %(method)s, %(window_days)s, %(correlation)s,
                %(direction)s, %(verdict)s, %(confidence)s,
                %(mechanism)s, %(alternative_explanations)s,
                %(reasoning)s, %(relationship_category)s
            )
        """

        rows = [
            {
                "run_id": run_id,
                "entity_a_id": r["entity_a_id"],
                "entity_b_id": r["entity_b_id"],
                "method": r.get("method", "partial"),
                "window_days": r["window_days"],
                "correlation": r["correlation"],
                "direction": r["direction"],
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "mechanism": r["mechanism"],
                "alternative_explanations": r["alternative_explanations"],
                "reasoning": r["reasoning"],
                "relationship_category": r["relationship_category"],
            }
            for r in results
        ]

        try:
            with self._conn.cursor() as cursor:
                cursor.executemany(query, rows)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.error(
                "Failed to insert %d verification results — rolled back",
                len(rows),
            )
            raise

        logger.info("Inserted %d verification results for run %d", len(rows), run_id)
        return len(rows)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_latest_market_data(
        self,
        *,
        entity_type: str,
        entity_id: str,
        metric: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get the most recent data points for a specific entity and metric.

        Returns rows newest-first. Each row is a dict with keys matching
        the column names (id, entity_type, entity_id, metric, value, etc.).

        Args:
            entity_type: Category of entity (e.g., "asset", "indicator").
            entity_id:   Identifier within that category (e.g., "NVDA", "CPI").
            metric:      Which measurement (e.g., "close_price", "value").
            limit:       Maximum number of rows to return (default 30).

        Returns:
            List of dicts, newest first. Empty list if no data found.
        """
        logger.debug(
            "Fetching latest %d data points for %s/%s/%s",
            limit,
            entity_type,
            entity_id,
            metric,
        )

        params = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "metric": metric,
            "limit": limit,
        }

        with self._conn.cursor() as cursor:
            cursor.execute(SELECT_LATEST_DATA_POINTS, params)
            rows = cursor.fetchall()

        logger.debug("Found %d data points", len(rows))
        return rows

    def get_market_data_range(
        self,
        *,
        entity_type: str,
        entity_id: str,
        metric: str,
        start: pendulum.DateTime,
        end: pendulum.DateTime,
    ) -> list[dict[str, Any]]:
        """Get all data points for an entity/metric within a date range.

        Returns rows in chronological order (oldest first), which is the
        natural order for time-series analysis and charting.

        Args:
            entity_type: Category of entity (e.g., "asset", "indicator").
            entity_id:   Identifier within that category (e.g., "NVDA", "CPI").
            metric:      Which measurement (e.g., "close_price", "value").
            start:       Beginning of the date range (inclusive).
            end:         End of the date range (inclusive).

        Returns:
            List of dicts, oldest first. Empty list if no data found.
        """
        logger.debug(
            "Fetching data points for %s/%s/%s from %s to %s",
            entity_type,
            entity_id,
            metric,
            start,
            end,
        )

        params = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "metric": metric,
            "start": start,
            "end": end,
        }

        with self._conn.cursor() as cursor:
            cursor.execute(SELECT_DATA_POINTS_RANGE, params)
            rows = cursor.fetchall()

        logger.debug("Found %d data points in range", len(rows))
        return rows

    def get_all_market_data_range(
        self,
        *,
        start: pendulum.DateTime,
        end: pendulum.DateTime,
    ) -> list[dict[str, Any]]:
        """Get ALL data points across ALL entities within a date range.

        Unlike get_market_data_range() which fetches one entity at a time,
        this pulls everything in one query. The correlation engine uses this
        to load all price and indicator data in a single database call
        instead of making 200 separate queries.

        Returns rows in chronological order, grouped by entity. Each row
        is a dict with: entity_type, entity_id, metric, value, timestamp.

        Args:
            start: Beginning of the date range (inclusive).
            end:   End of the date range (inclusive).

        Returns:
            List of dicts, ordered by entity then timestamp.
        """
        logger.debug("Fetching ALL market data from %s to %s", start, end)

        query = """
            SELECT entity_type, entity_id, metric, value, timestamp
            FROM market_data
            WHERE timestamp >= %(start)s
              AND timestamp <= %(end)s
              AND metric != %(excluded_metric)s
            ORDER BY entity_type, entity_id, timestamp ASC;
        """

        with self._conn.cursor() as cursor:
            cursor.execute(query, {
                "start": start,
                "end": end,
                "excluded_metric": EXCLUDED_CORRELATION_METRIC,
            })
            rows = cursor.fetchall()

        logger.info("Loaded %d data points across all entities", len(rows))
        return rows

    # ------------------------------------------------------------------
    # Generic query execution
    # ------------------------------------------------------------------

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run an arbitrary SQL query and return results as a list of dicts.

        Use this for one-off queries in scripts or exploratory work where
        none of the specialised methods above fit. For anything called in
        a tight loop, prefer writing a dedicated method with a named constant.

        SELECT statements return every matching row as a dict whose keys
        match the column names (e.g. [{"id": 1, "entity_id": "NVDA", ...}]).

        INSERT / UPDATE / DELETE statements are committed automatically and
        return an empty list — if you need the affected rows back, add a
        RETURNING clause to your query and it will come through as a SELECT.

        Args:
            query:  Any valid SQL string. Use %(name)s placeholders for
                    values — never build the query by string concatenation.
            params: Dict of values to bind to the %(name)s placeholders.
                    Pass None (or omit) when the query has no parameters.

        Returns:
            List of dicts for SELECT-style queries; empty list otherwise.

        Example:
            rows = db.execute_query(
                "SELECT * FROM market_data WHERE entity_id = %(eid)s",
                {"eid": "NVDA"},
            )
        """
        # Reject empty or whitespace-only queries immediately — they would
        # cause a cryptic IndexError on the .split()[0] call below, and no
        # valid caller should ever pass an empty string.
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Normalise the query so we can detect SELECT without worrying about
        # leading whitespace or mixed case (e.g. "  select * FROM ...").
        query_type = query.strip().split()[0].upper()

        logger.debug("Executing %s query via execute_query()", query_type)

        with self._conn.cursor() as cursor:
            # psycopg v3 accepts None params just fine — it treats it the same
            # as passing no params at all, so no special-casing needed here.
            cursor.execute(query, params)

            # For SELECT and RETURNING clauses, fetch all rows.
            # For everything else, commit the transaction and return nothing.
            has_returning = "RETURNING" in query.upper()
            if query_type == "SELECT" or has_returning:
                rows: list[dict] = cursor.fetchall()
                logger.debug("execute_query() returned %d row(s)", len(rows))
                if has_returning:
                    self._conn.commit()
                return rows

        # Non-SELECT path — commit so the write is persisted.
        self._conn.commit()
        logger.debug("execute_query() committed %s statement", query_type)
        return []
