"""
Neo4j graph database storage — all knowledge graph interactions live here.

This module provides GraphStorage, the single gateway for reading and writing
to Neo4j. Every other module that needs the graph imports this class rather
than touching the Neo4j driver directly.

Key design decisions:
- Uses the official neo4j Python driver (v5+ API with driver.execute_query).
- Batch inserts use UNWIND so we send one Cypher statement instead of N.
- Every entity label gets a uniqueness constraint on `id` and an index on
  `name` so lookups are always fast.
- Works as a context manager (with statement) so the connection is always
  cleaned up, even if something crashes.

Usage:
    from zerofin.storage.graph import GraphStorage

    with GraphStorage() as graph:
        graph.setup_indexes()
        graph.create_entity("Asset", {"id": "AAPL", "name": "Apple Inc."})
"""

from __future__ import annotations

import logging
import re
import time
from types import TracebackType
from typing import Any

import neo4j

from zerofin.config import settings
from zerofin.constants import ENTITY_LABELS

# One logger for the whole module — messages show up as "zerofin.storage.graph"
logger = logging.getLogger(__name__)


class GraphStorage:
    """Manages all Neo4j interactions for the Zerofin knowledge graph.

    Can be used standalone or as a context manager:

        # Standalone
        graph = GraphStorage()
        graph.connect()
        ...
        graph.close()

        # Context manager (preferred — auto-closes on exit)
        with GraphStorage() as graph:
            ...
    """

    def __init__(self) -> None:
        """Set up instance variables. Does NOT connect yet — call connect()."""
        # The driver is created in connect(). We store None here so the
        # type checker knows it might not exist yet.
        self._driver: neo4j.Driver | None = None

    # ── Connection lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Open a connection to Neo4j and verify it's reachable.

        Reads URI, user, and password from the global settings object
        (which loads them from .env). Raises if Neo4j is unreachable.
        """
        logger.info("Connecting to Neo4j at %s", settings.NEO4J_URI)

        # Create the driver with a connection timeout so we don't hang
        # indefinitely if Neo4j is unreachable.
        self._driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=30,
        )

        # Verify connectivity with one retry — transient failures happen.
        try:
            self._driver.verify_connectivity()
        except Exception:
            logger.warning("Neo4j connection failed — retrying in 2s")
            time.sleep(2)
            self._driver.verify_connectivity()

        logger.info("Connected to Neo4j successfully")

    def close(self) -> None:
        """Close the Neo4j driver and release its resources."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    # ── Context manager support ───────────────────────────────────────
    # These two dunder methods let you use `with GraphStorage() as g:`

    def __enter__(self) -> GraphStorage:
        """Connect when entering a `with` block."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Always close the connection when leaving a `with` block."""
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────

    @property
    def driver(self) -> neo4j.Driver:
        """Return the active driver, or raise if we never connected.

        This is a safety net — every public method goes through this
        property so we get a clear error instead of a cryptic AttributeError.
        """
        if self._driver is None:
            raise RuntimeError(
                "Not connected to Neo4j. Call connect() first "
                "or use GraphStorage as a context manager."
            )
        return self._driver

    # ── Index & constraint setup ──────────────────────────────────────

    def setup_indexes(self) -> None:
        """Create uniqueness constraints and indexes for every entity label.

        Safe to call multiple times — Neo4j skips creation if the
        constraint/index already exists (IF NOT EXISTS).

        For each of the 12 entity types this creates:
        - A uniqueness constraint on `id`  (also acts as an index)
        - A range index on `name` for fast text lookups
        """
        logger.info("Setting up Neo4j indexes and constraints")

        for label in ENTITY_LABELS:
            # Uniqueness constraint — ensures no two nodes of the same
            # label share an id, AND gives us a fast lookup index for free.
            constraint_query = (
                f"CREATE CONSTRAINT constraint_{label.lower()}_id "
                f"IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )
            self.driver.execute_query(constraint_query)
            logger.debug("Constraint ensured for %s.id", label)

            # Name index — we often search entities by name, so this
            # makes those queries fast. Separate from the constraint
            # because names aren't necessarily unique (two entities
            # could share a display name).
            index_query = (
                f"CREATE INDEX index_{label.lower()}_name IF NOT EXISTS FOR (n:{label}) ON (n.name)"
            )
            self.driver.execute_query(index_query)
            logger.debug("Index ensured for %s.name", label)

        logger.info(
            "Indexes and constraints ready for %d entity types",
            len(ENTITY_LABELS),
        )

    # ── Entity operations ─────────────────────────────────────────────

    def create_entity(self, label: str, properties: dict[str, Any]) -> dict[str, Any]:
        """Create a single entity node in the graph.

        Args:
            label: The node label — must be one of ENTITY_LABELS
                   (e.g. "Asset", "Company", "Indicator").
            properties: A dict of properties for the node. Must include
                        an "id" key at minimum.

        Returns:
            The properties of the created (or merged) node.

        Raises:
            ValueError: If the label isn't recognized or id is missing.
        """
        # ── Validate inputs ──
        self._validate_label(label)

        if "id" not in properties:
            raise ValueError("Entity properties must include an 'id' key")

        # MERGE = create if it doesn't exist, match if it does.
        # SET n += $props updates any changed properties on an existing node.
        # This makes the operation idempotent — safe to call twice.
        query = f"MERGE (n:{label} {{id: $id}}) SET n += $props RETURN n"

        # execute_query returns a named tuple: (records, summary, keys)
        records, _, _ = self.driver.execute_query(
            query,
            id=properties["id"],
            props=properties,
        )

        # Pull the node's properties out of the first (only) result row
        node_data = dict(records[0]["n"])
        logger.info("Created/updated %s entity: %s", label, properties["id"])
        return node_data

    def create_entities_batch(self, label: str, entities: list[dict[str, Any]]) -> int:
        """Create multiple entities in one shot using UNWIND.

        This is MUCH faster than calling create_entity() in a loop because
        it sends one Cypher statement to Neo4j instead of N separate ones.

        Args:
            label: The node label for ALL entities in this batch.
            entities: A list of property dicts. Each must include "id".

        Returns:
            The number of entities created or updated.

        Raises:
            ValueError: If the label isn't recognized or any entity lacks an id.
        """
        # ── Validate inputs ──
        self._validate_label(label)

        if not entities:
            logger.warning("create_entities_batch called with empty list")
            return 0

        # Check every entity has an id before we send anything to Neo4j.
        # We validate up front so we don't get a partial insert.
        for entity in entities:
            if "id" not in entity:
                raise ValueError(f"Every entity must have an 'id'. Got: {entity}")

        # UNWIND takes a list and "unwraps" it into individual rows.
        # For each row we MERGE (upsert) a node and set its properties.
        # This is the recommended Neo4j pattern for bulk inserts.
        query = (
            f"UNWIND $batch AS item "
            f"MERGE (n:{label} {{id: item.id}}) "
            f"SET n += item "
            f"RETURN count(n) AS total"
        )

        records, _, _ = self.driver.execute_query(
            query,
            batch=entities,
        )

        total = records[0]["total"]
        logger.info("Batch created/updated %d %s entities", total, label)
        return total

    def get_entity(self, label: str, entity_id: str) -> dict[str, Any] | None:
        """Look up a single entity by its label and id.

        Args:
            label: The node label (e.g. "Asset").
            entity_id: The unique id to search for.

        Returns:
            A dict of the node's properties, or None if not found.
        """
        self._validate_label(label)

        query = f"MATCH (n:{label} {{id: $id}}) RETURN n"
        records, _, _ = self.driver.execute_query(query, id=entity_id)

        if not records:
            logger.debug("%s with id '%s' not found", label, entity_id)
            return None

        return dict(records[0]["n"])

    # ── Relationship operations ───────────────────────────────────────

    def create_relationship(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a directed relationship between two existing entities.

        Uses MERGE so calling this twice with the same endpoints and type
        won't create a duplicate — it updates the properties instead.

        Args:
            from_label: Label of the source node (e.g. "Asset").
            from_id: id of the source node.
            to_label: Label of the target node (e.g. "Sector").
            to_id: id of the target node.
            relationship_type: The relationship name in UPPER_SNAKE_CASE
                               (e.g. "BELONGS_TO", "CORRELATES_WITH").
            properties: Optional dict of relationship metadata. Typical
                        keys: confidence, times_tested, times_confirmed,
                        valid_from, valid_until, source, status.

        Returns:
            The properties stored on the relationship.

        Raises:
            ValueError: If either label is invalid.
        """
        self._validate_label(from_label)
        self._validate_label(to_label)
        self._validate_relationship_type(relationship_type)

        # Default to empty properties if none provided
        rel_props = properties or {}

        # We match both endpoints first, then MERGE the relationship.
        # If either endpoint doesn't exist, the MATCH returns nothing
        # and the MERGE never runs — no orphan edges.
        query = (
            f"MATCH (a:{from_label} {{id: $from_id}}) "
            f"MATCH (b:{to_label} {{id: $to_id}}) "
            f"MERGE (a)-[r:{relationship_type}]->(b) "
            f"SET r += $props "
            f"RETURN r"
        )

        records, _, _ = self.driver.execute_query(
            query,
            from_id=from_id,
            to_id=to_id,
            props=rel_props,
        )

        if not records:
            logger.warning(
                "Relationship not created — one or both entities missing: %s(%s) -[%s]-> %s(%s)",
                from_label,
                from_id,
                relationship_type,
                to_label,
                to_id,
            )
            return {}

        rel_data = dict(records[0]["r"])
        logger.info(
            "Created/updated relationship: %s(%s) -[%s]-> %s(%s)",
            from_label,
            from_id,
            relationship_type,
            to_label,
            to_id,
        )
        return rel_data

    def get_entity_relationships(self, label: str, entity_id: str) -> list[dict[str, Any]]:
        """Get ALL relationships (incoming and outgoing) for an entity.

        Returns a list of dicts, each describing one relationship and
        the node on the other end.

        Args:
            label: The node label of the entity to look up.
            entity_id: The id of the entity.

        Returns:
            A list of dicts with keys:
            - "direction": "outgoing" or "incoming"
            - "relationship_type": the relationship label string
            - "relationship_properties": dict of properties on the edge
            - "other_node": dict of properties on the connected node
            - "other_label": the label(s) of the connected node
        """
        self._validate_label(label)

        # We use two separate patterns to capture direction:
        #   (n)-[r]->(other)  — outgoing
        #   (n)<-[r]-(other)  — incoming
        # UNION combines them into one result set.
        query = (
            f"MATCH (n:{label} {{id: $id}})-[r]->(other) "
            f"RETURN 'outgoing' AS direction, type(r) AS rel_type, "
            f"       properties(r) AS rel_props, "
            f"       properties(other) AS other_props, "
            f"       labels(other) AS other_labels "
            f"UNION "
            f"MATCH (n:{label} {{id: $id}})<-[r]-(other) "
            f"RETURN 'incoming' AS direction, type(r) AS rel_type, "
            f"       properties(r) AS rel_props, "
            f"       properties(other) AS other_props, "
            f"       labels(other) AS other_labels"
        )

        records, _, _ = self.driver.execute_query(query, id=entity_id)

        # Convert each record into a plain dict for easy consumption
        results: list[dict[str, Any]] = []
        for record in records:
            results.append(
                {
                    "direction": record["direction"],
                    "relationship_type": record["rel_type"],
                    "relationship_properties": dict(record["rel_props"]),
                    "other_node": dict(record["other_props"]),
                    "other_labels": list(record["other_labels"]),
                }
            )

        logger.info(
            "Found %d relationships for %s(%s)",
            len(results),
            label,
            entity_id,
        )
        return results

    def create_relationships_batch(
        self,
        relationships: list[dict[str, Any]],
    ) -> int:
        """Batch-create relationships using UNWIND, grouped by type.

        Since Neo4j can't parameterize relationship types, this groups
        the input by rel_type and runs one UNWIND query per type. Still
        much faster than individual create_relationship() calls.

        Args:
            relationships: List of dicts, each with keys:
                - from_id: id of the source node
                - to_id: id of the target node
                - rel_type: relationship name in UPPER_SNAKE_CASE
                - props: dict of relationship properties (optional)

        Returns:
            Total number of relationships created or updated.
        """
        if not relationships:
            return 0

        # Validate all relationship types up front
        for rel in relationships:
            self._validate_relationship_type(rel["rel_type"])

        # Group by relationship type — one UNWIND query per type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for rel in relationships:
            rel_type = rel["rel_type"]
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append({
                "from_id": rel["from_id"],
                "to_id": rel["to_id"],
                "props": rel.get("props", {}),
            })

        total = 0
        for rel_type, batch in by_type.items():
            query = (
                f"UNWIND $batch AS item "
                f"MATCH (a {{id: item.from_id}}) "
                f"MATCH (b {{id: item.to_id}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"SET r += item.props "
                f"RETURN count(r) AS total"
            )
            records, _, _ = self.driver.execute_query(query, batch=batch)
            total += records[0]["total"]

        logger.info("Batch created/updated %d relationships", total)
        return total

    # ── Generic query ─────────────────────────────────────────────────

    def run_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run an arbitrary Cypher query and return results as dicts.

        This is the escape hatch for queries that don't fit the standard
        CRUD methods above. Use it sparingly — prefer the named methods
        when possible so queries stay consistent.

        Args:
            query: A Cypher query string. Use $param_name for parameters.
            parameters: Optional dict of query parameters.

        Returns:
            A list of dicts, one per result row. Keys are the RETURN
            column names from your Cypher query.
        """
        params = parameters or {}
        logger.debug("Running custom query: %s", query)

        records, _, _ = self.driver.execute_query(query, **params)

        # Convert each Record to a plain dict so callers don't need
        # to know about the neo4j Record type
        return [dict(record) for record in records]

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _safe_label(label: str) -> str:
        """Validate and return the label — impossible to use unvalidated.

        This prevents typos from silently creating random node labels
        in the graph, and guards against Cypher injection since labels
        are interpolated into f-strings. Returns the label so callers
        can use it directly: f"MERGE (n:{self._safe_label(label)} ...)".
        """
        if label not in ENTITY_LABELS:
            raise ValueError(
                f"Unknown entity label '{label}'. "
                f"Must be one of: {ENTITY_LABELS}"
            )
        return label

    @staticmethod
    def _validate_label(label: str) -> None:
        """Raise ValueError if the label isn't in our allowed list.

        Kept for backwards compatibility — prefer _safe_label() for new code.
        """
        if label not in ENTITY_LABELS:
            raise ValueError(
                f"Unknown entity label '{label}'. "
                f"Must be one of: {ENTITY_LABELS}"
            )

    @staticmethod
    def _safe_relationship_type(rel_type: str) -> str:
        """Validate and return the relationship type.

        Must start with a letter and contain only uppercase letters and
        underscores (minimum 2 characters). Returns the type so callers
        can use it directly in f-strings.
        """
        if not re.match(r"^[A-Z][A-Z_]{1,}$", rel_type):
            raise ValueError(
                f"Invalid relationship type '{rel_type}'. "
                f"Must be UPPER_SNAKE_CASE, starting with a letter, "
                f"at least 2 characters."
            )
        return rel_type

    @staticmethod
    def _validate_relationship_type(rel_type: str) -> None:
        """Raise ValueError if the relationship type has invalid characters.

        Kept for backwards compatibility — prefer _safe_relationship_type().
        """
        if not re.match(r"^[A-Z][A-Z_]{1,}$", rel_type):
            raise ValueError(
                f"Invalid relationship type '{rel_type}'. "
                f"Must be UPPER_SNAKE_CASE, starting with a letter, "
                f"at least 2 characters."
            )
