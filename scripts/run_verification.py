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
import sys
import time
from pathlib import Path
from typing import Any

# UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zerofin.ai.verification import (  # noqa: E402
    VerificationResult,
    build_review_chain,
    build_verification_chain,
    needs_second_pass,
)
from zerofin.config import settings  # noqa: E402
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
    CASE WHEN 'Indicator' IN labels(a) THEN 'indicator' ELSE 'asset' END AS entity_a_type,
    b.id AS entity_b_id,
    b.name AS entity_b_name,
    b.description AS entity_b_desc,
    CASE WHEN 'Indicator' IN labels(b) THEN 'indicator' ELSE 'asset' END AS entity_b_type,
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


def _escape_braces(text: str) -> str:
    """Escape curly braces so LangChain's template engine doesn't interpret them."""
    return text.replace("{", "{{").replace("}", "}}")


def _build_input(row: dict) -> dict:
    """Convert a Neo4j row to the format the verification chain expects."""
    return {
        # IDs are alphanumeric + =, ., ^, - only — no brace escaping needed
        "entity_a_id": row["entity_a_id"],
        "entity_a_desc": _escape_braces(row.get("entity_a_desc") or row.get("entity_a_name") or ""),
        "entity_a_type": row["entity_a_type"],
        "entity_b_id": row["entity_b_id"],
        "entity_b_desc": _escape_braces(row.get("entity_b_desc") or row.get("entity_b_name") or ""),
        "entity_b_type": row["entity_b_type"],
        "correlation": row["correlation"],
        "direction": row["direction"],
        "window_days": row["window_days"],
        "observation_count": row["observation_count"],
    }


def _combine_result(row: dict, result: VerificationResult) -> dict:
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


def verify_batch(rows: list[dict], chain: Any = None) -> list[dict]:
    """Run verification chain on all rows in chunks.

    Processes VERIFICATION_CHUNK_SIZE pairs at a time, saving progress after
    each chunk. If a chunk fails, previously completed chunks are preserved
    in the CSV.

    Args:
        rows: Relationship rows to verify.
        chain: Pre-built verification chain. If None, one is built internally.
            Pass the warm-up chain to avoid building it twice.
    """
    chunk_size = settings.VERIFICATION_CHUNK_SIZE
    if chain is None:
        chain = build_verification_chain()
    combined = []
    total = len(rows)
    total_chunks = (total + chunk_size - 1) // chunk_size

    logger.info(
        "Processing %d pairs in %d chunks of %d",
        total, total_chunks, chunk_size,
    )
    start = time.time()

    for i in range(0, total, chunk_size):
        chunk_rows = rows[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        chunk_inputs = [_build_input(row) for row in chunk_rows]

        logger.info(
            "Chunk %d/%d: processing %d pairs (%d-%d of %d)...",
            chunk_num, total_chunks, len(chunk_rows),
            i + 1, min(i + chunk_size, total), total,
        )

        try:
            results = chain.batch(chunk_inputs, config={"max_concurrency": chunk_size})
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
        "correlation", "direction", "window_days", "observation_count", "method",
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


def log_summary(results: list[dict]) -> None:
    """Log verdict distribution."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    logger.info("=" * 60)
    logger.info("RESULTS (%d pairs)", len(results))
    logger.info("-" * 60)
    logger.info("Verdict distribution:")
    for verdict in ["confirmed_plausible", "likely_plausible", "uncertain",
                     "likely_spurious", "confirmed_spurious"]:
        count = counts.get(verdict, 0)
        pct = count / len(results) * 100 if results else 0
        bar = "#" * count
        logger.info("  %-25s %3d (%4.1f%%) %s", verdict, count, pct, bar)

    promoted = sum(
        1 for r in results
        if r["verdict"] in ("confirmed_plausible", "likely_plausible")
    )
    rejected = sum(1 for r in results if r["verdict"] in ("confirmed_spurious", "likely_spurious"))
    pending = sum(1 for r in results if r["verdict"] == "uncertain")
    logger.info("Promoted to candidate: %d", promoted)
    logger.info("Rejected: %d", rejected)
    logger.info("Still pending: %d", pending)
    logger.info("=" * 60)


def main() -> None:
    with GraphStorage() as graph:
        # 1. Fetch pending relationships
        rows = fetch_pending(graph)
        if not rows:
            logger.info("No pending_verification relationships found. Nothing to do.")
            return

        # 2. Cap batch size to prevent runaway API costs
        max_batch = settings.MAX_VERIFICATION_BATCH
        if len(rows) > max_batch:
            logger.warning(
                "Found %d pending pairs, capping to MAX_VERIFICATION_BATCH=%d",
                len(rows), max_batch,
            )
            rows = rows[:max_batch]

        # 3. Warm-up: test a single call before committing to the full batch
        chain = build_verification_chain()
        test_input = _build_input(rows[0])
        try:
            test_result = chain.invoke(test_input)
            logger.info(
                "Warm-up OK: %s vs %s -> %s (confidence=%.2f)",
                rows[0]["entity_a_id"], rows[0]["entity_b_id"],
                test_result.verdict, test_result.confidence,
            )
        except Exception as e:
            logger.error("Warm-up call FAILED: %s — aborting batch.", e)
            return

        # 4. Run first pass (DeepSeek) — warm-up already processed rows[0],
        # so carry that result forward and only batch the remainder.
        warmup_combined = [_combine_result(rows[0], test_result)]
        results = warmup_combined + verify_batch(rows[1:], chain=chain)

        # 5. Run second pass (Sonnet) on borderline cases
        borderline_indices = []
        for i, r in enumerate(results):
            result_obj = VerificationResult(
                mechanism=r["mechanism"],
                alternative_explanations=r["alternative_explanations"],
                verdict=r["verdict"],
                confidence=r["confidence"],
                reasoning=r["reasoning"],
                relationship_category=r["relationship_category"],
            )
            if needs_second_pass(result_obj):
                borderline_indices.append(i)

        if borderline_indices:
            logger.info(
                "Second pass: %d borderline cases routed to Sonnet",
                len(borderline_indices),
            )
            review_chain = build_review_chain()
            review_inputs = []
            for i in borderline_indices:
                r = results[i]
                review_inputs.append({
                    "entity_a_id": r["entity_a_id"],
                    "entity_a_desc": _escape_braces(rows[i].get("entity_a_desc") or rows[i].get("entity_a_name") or ""),
                    "entity_a_type": rows[i]["entity_a_type"],
                    "entity_b_id": r["entity_b_id"],
                    "entity_b_desc": _escape_braces(rows[i].get("entity_b_desc") or rows[i].get("entity_b_name") or ""),
                    "entity_b_type": rows[i]["entity_b_type"],
                    "correlation": r["correlation"],
                    "direction": r["direction"],
                    "window_days": r["window_days"],
                    "observation_count": r["observation_count"],
                    "pass1_verdict": r["verdict"],
                    "pass1_confidence": r["confidence"],
                    "pass1_mechanism": r["mechanism"],
                    "pass1_reasoning": r["reasoning"],
                })

            try:
                review_results = review_chain.batch(
                    review_inputs, config={"max_concurrency": 10}
                )
                for idx, review_result in zip(borderline_indices, review_results):
                    old_verdict = results[idx]["verdict"]
                    results[idx]["verdict"] = review_result.verdict
                    results[idx]["confidence"] = review_result.confidence
                    results[idx]["mechanism"] = review_result.mechanism
                    results[idx]["alternative_explanations"] = review_result.alternative_explanations
                    results[idx]["reasoning"] = review_result.reasoning
                    results[idx]["relationship_category"] = review_result.relationship_category
                    if old_verdict != review_result.verdict:
                        logger.info(
                            "  Sonnet override: %s <-> %s: %s -> %s",
                            results[idx]["entity_a_id"],
                            results[idx]["entity_b_id"],
                            old_verdict,
                            review_result.verdict,
                        )
            except Exception as e:
                logger.error("Second pass FAILED: %s — keeping first-pass results.", e)
        else:
            logger.info("Second pass: no borderline cases to review")

        # 6. Export CSV for audit
        write_csv(results)

        # 7. Log summary
        log_summary(results)

        # 8. Write back to Neo4j
        # Commented out until we've audited the results
        # write_to_neo4j(graph, results)
        logger.info("Neo4j write is DISABLED — uncomment after auditing CSV")


if __name__ == "__main__":
    main()
