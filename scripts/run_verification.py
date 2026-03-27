"""Run LLM verification on all pending_verification relationships in Neo4j.

Pulls pending relationships, sends them through DeepSeek for plausibility
judgment, writes results back to Neo4j, and exports a CSV for auditing.

Usage:
    python scripts/run_verification.py

Output:
    logs/verification_results.csv — full results for audit
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
import time
from pathlib import Path

# UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["LLM_PROVIDER"] = "openrouter"
os.environ["LLM_MODEL"] = "deepseek/deepseek-chat"

from zerofin.ai.verification import build_verification_chain  # noqa: E402
from zerofin.storage.graph import GraphStorage  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Cypher to pull pending relationships with entity descriptions
FETCH_PENDING_QUERY = """
MATCH (a)-[r:CORRELATES_WITH {status: 'pending_verification'}]->(b)
RETURN
    a.id AS entity_a_id,
    a.name AS entity_a_name,
    a.description AS entity_a_desc,
    CASE WHEN labels(a)[0] = 'Indicator' THEN 'indicator' ELSE 'asset' END AS entity_a_type,
    b.id AS entity_b_id,
    b.name AS entity_b_name,
    b.description AS entity_b_desc,
    CASE WHEN labels(b)[0] = 'Indicator' THEN 'indicator' ELSE 'asset' END AS entity_b_type,
    r.correlation AS correlation,
    r.direction AS direction,
    r.window_days AS window_days,
    r.observation_count AS observation_count,
    r.method AS method
ORDER BY abs(r.correlation) DESC
"""

# Cypher to write verification results back to the relationship
UPDATE_VERIFICATION_QUERY = """
UNWIND $results AS row
MATCH (a)-[r:CORRELATES_WITH]->(b)
WHERE a.id = row.entity_a_id AND b.id = row.entity_b_id
  AND r.method = row.method AND r.window_days = row.window_days
SET r.llm_verdict = row.verdict,
    r.llm_confidence = row.confidence,
    r.llm_mechanism = row.mechanism,
    r.llm_reasoning = row.reasoning,
    r.llm_category = row.relationship_category,
    r.llm_alternatives = row.alternative_explanations,
    r.status = CASE
        WHEN row.verdict IN ['confirmed_spurious', 'likely_spurious']
        THEN 'rejected'
        WHEN row.verdict IN ['confirmed_plausible', 'likely_plausible']
        THEN 'candidate'
        ELSE 'pending_verification'
    END
"""

CSV_PATH = Path(__file__).resolve().parent.parent / "logs" / "verification_results.csv"


def fetch_pending(graph: GraphStorage) -> list[dict]:
    """Pull all pending_verification relationships from Neo4j."""
    rows = graph.run_query(FETCH_PENDING_QUERY)
    logger.info("Found %d pending_verification relationships", len(rows))
    return rows


CHUNK_SIZE = 20


def _build_input(row: dict) -> dict:
    """Convert a Neo4j row to the format the verification chain expects."""
    return {
        "entity_a_id": row["entity_a_id"],
        "entity_a_desc": row.get("entity_a_desc") or row.get("entity_a_name") or "",
        "entity_a_type": row["entity_a_type"],
        "entity_b_id": row["entity_b_id"],
        "entity_b_desc": row.get("entity_b_desc") or row.get("entity_b_name") or "",
        "entity_b_type": row["entity_b_type"],
        "correlation": row["correlation"],
        "direction": row["direction"],
        "window_days": row["window_days"],
        "observation_count": row["observation_count"],
    }


def _combine_result(row: dict, result) -> dict:
    """Merge Neo4j row with verification result for CSV/Neo4j write."""
    return {
        "entity_a_id": row["entity_a_id"],
        "entity_a_name": row.get("entity_a_name", ""),
        "entity_b_id": row["entity_b_id"],
        "entity_b_name": row.get("entity_b_name", ""),
        "correlation": row["correlation"],
        "direction": row["direction"],
        "window_days": row["window_days"],
        "observation_count": row["observation_count"],
        "method": row.get("method", "partial"),
        "verdict": result.verdict,
        "confidence": result.confidence,
        "mechanism": result.mechanism,
        "alternative_explanations": result.alternative_explanations,
        "reasoning": result.reasoning,
        "relationship_category": result.relationship_category,
    }


def verify_batch(rows: list[dict]) -> list[dict]:
    """Run verification chain on all rows in chunks.

    Processes CHUNK_SIZE pairs at a time, saving progress after each chunk.
    If a chunk fails, previously completed chunks are preserved in the CSV.
    """
    chain = build_verification_chain()
    combined = []
    total = len(rows)
    total_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE

    logger.info(
        "Processing %d pairs in %d chunks of %d (max_concurrency=20)",
        total, total_chunks, CHUNK_SIZE,
    )
    start = time.time()

    for i in range(0, total, CHUNK_SIZE):
        chunk_rows = rows[i : i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        chunk_inputs = [_build_input(row) for row in chunk_rows]

        logger.info(
            "Chunk %d/%d: processing %d pairs (%d-%d of %d)...",
            chunk_num, total_chunks, len(chunk_rows),
            i + 1, min(i + CHUNK_SIZE, total), total,
        )

        try:
            results = chain.batch(chunk_inputs, config={"max_concurrency": CHUNK_SIZE})
            for row, result in zip(chunk_rows, results):
                combined.append(_combine_result(row, result))

            # Save progress after each chunk
            write_csv(combined)
            logger.info(
                "Chunk %d/%d complete — %d total results saved",
                chunk_num, total_chunks, len(combined),
            )
        except Exception as e:
            logger.error(
                "Chunk %d/%d FAILED: %s — stopping. %d results saved from previous chunks.",
                chunk_num, total_chunks, e, len(combined),
            )
            break

    elapsed = time.time() - start
    logger.info(
        "Verification finished in %.1fs — %d/%d pairs completed",
        elapsed, len(combined), total,
    )
    return combined


def write_csv(results: list[dict]) -> None:
    """Write results to CSV for audit."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "entity_a_id", "entity_a_name", "entity_b_id", "entity_b_name",
        "correlation", "direction", "window_days", "observation_count",
        "verdict", "confidence", "mechanism", "alternative_explanations",
        "reasoning", "relationship_category",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info("CSV written to %s", CSV_PATH)


def write_to_neo4j(graph: GraphStorage, results: list[dict]) -> None:
    """Write verification results back to Neo4j relationships."""
    graph.run_query(UPDATE_VERIFICATION_QUERY, {"results": results})
    logger.info("Updated %d relationships in Neo4j", len(results))


def print_summary(results: list[dict]) -> None:
    """Print verdict distribution."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    print("\n" + "=" * 60)
    print(f"RESULTS ({len(results)} pairs)")
    print("-" * 60)
    print("Verdict distribution:")
    for verdict in ["confirmed_plausible", "likely_plausible", "uncertain",
                     "likely_spurious", "confirmed_spurious"]:
        count = counts.get(verdict, 0)
        pct = count / len(results) * 100 if results else 0
        bar = "#" * count
        print(f"  {verdict:25s} {count:3d} ({pct:4.1f}%) {bar}")

    promoted = sum(
        1 for r in results
        if r["verdict"] in ("confirmed_plausible", "likely_plausible")
    )
    rejected = sum(1 for r in results if r["verdict"] in ("confirmed_spurious", "likely_spurious"))
    pending = sum(1 for r in results if r["verdict"] == "uncertain")
    print(f"\nPromoted to candidate: {promoted}")
    print(f"Rejected: {rejected}")
    print(f"Still pending: {pending}")
    print("=" * 60)


def main() -> None:
    with GraphStorage() as graph:
        # 1. Fetch pending relationships
        rows = fetch_pending(graph)
        if not rows:
            logger.info("No pending_verification relationships found. Nothing to do.")
            return

        # 2. Run verification
        results = verify_batch(rows)

        # 3. Export CSV for audit
        write_csv(results)

        # 4. Print summary
        print_summary(results)

        # 5. Write back to Neo4j
        # Commented out until we've audited the results
        # write_to_neo4j(graph, results)
        logger.info("Neo4j write is DISABLED — uncomment after auditing CSV")


if __name__ == "__main__":
    main()
