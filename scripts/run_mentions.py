"""Run entity mention indexer on unprocessed articles in Neo4j.

Pulls articles with status='raw', identifies which tracked entities each
article references using DeepSeek via LangChain, creates MENTIONS edges
in Neo4j, and updates article status to 'mentions_done'.

Usage:
    python scripts/run_mentions.py

Output:
    AppData/Local/zerofin/mentions/mentions_YYYY-MM-DD.csv
"""

from __future__ import annotations

import csv
import io
import logging
import sys
import time
from pathlib import Path

import pendulum

# UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zerofin.ai.mentions import (  # noqa: E402
    MentionResult,
    build_entity_list,
    build_mention_chain,
    create_mentioned_in_edges,
    format_entity_list,
    validate_mention_ids,
)
from zerofin.config import settings  # noqa: E402
from zerofin.storage.graph import GraphStorage  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# How many articles to process in parallel per chunk
CHUNK_SIZE = 20

# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

FETCH_RAW_ARTICLES = """
    MATCH (a:Article)
    WHERE a.status = 'raw'
      AND a.summary IS NOT NULL AND a.summary <> ''
    RETURN a.id AS url, a.title AS title, a.summary AS summary,
           a.source AS source, a.tier AS tier
    ORDER BY a.collected_at DESC
"""

UPDATE_STATUS_QUERY = """
    UNWIND $urls AS url
    MATCH (a:Article {id: url})
    SET a.status = 'mentions_done',
        a.mentions_processed_at = datetime()
"""

SKIP_NO_SUMMARY_QUERY = """
    MATCH (a:Article)
    WHERE a.status = 'raw'
      AND (a.summary IS NULL OR a.summary = '')
    SET a.status = 'skipped_no_summary'
    RETURN count(a) AS skipped
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_raw_articles(graph: GraphStorage) -> list[dict]:
    """Pull all unprocessed articles from Neo4j."""
    rows = graph.run_query(FETCH_RAW_ARTICLES)
    logger.info("Found %d unprocessed articles", len(rows))
    return rows


def build_chain_input(article: dict, entity_list_text: str) -> dict:
    """Convert an article dict to the format the mention chain expects."""
    article_text = f"{article['title']}\n\n{article['summary']}"
    return {
        "entity_list": entity_list_text,
        "article_text": article_text,
    }


def build_result_row(article: dict, mention_ids: list[str]) -> dict:
    """Build a result dict for CSV export."""
    return {
        "url": article["url"],
        "title": article["title"],
        "source": article["source"],
        "tier": article.get("tier", ""),
        "mentions_found": len(mention_ids),
        "entity_ids": ", ".join(mention_ids),
    }


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

def process_chunk(
    graph: GraphStorage,
    chunk: list[dict],
    chain: object,
    entity_list_text: str,
    valid_ids: set[str],
    chunk_num: int,
    total_chunks: int,
) -> list[dict]:
    """Process one chunk of articles through the mention chain.

    Sends all articles in the chunk to DeepSeek in parallel via
    chain.batch(), validates returned IDs, creates MENTIONS edges,
    and updates article status.

    Returns a list of result dicts for CSV export.
    """
    chunk_inputs = [
        build_chain_input(article, entity_list_text) for article in chunk
    ]

    logger.info(
        "Chunk %d/%d: processing %d articles...",
        chunk_num, total_chunks, len(chunk),
    )

    results = chain.batch(chunk_inputs, config={"max_concurrency": CHUNK_SIZE})

    chunk_results = []
    total_edges = 0

    for article, result in zip(chunk, results):
        # Guard against unexpected None or non-MentionResult responses
        if not isinstance(result, MentionResult):
            logger.warning(
                "Unexpected result type for '%s': %s — treating as empty",
                article["title"][:60], type(result).__name__,
            )
            mention_ids: list[str] = []
        else:
            validated = validate_mention_ids(result, valid_ids)
            mention_ids = validated.mentioned_ids

        # Create MENTIONS edges
        edges_created = create_mentioned_in_edges(
            graph, article["url"], mention_ids,
        )
        total_edges += edges_created

        chunk_results.append(build_result_row(article, mention_ids))

    # Update status for all articles in this chunk
    urls = [a["url"] for a in chunk]
    graph.run_query(UPDATE_STATUS_QUERY, {"urls": urls})

    logger.info(
        "Chunk %d/%d complete — %d edges created across %d articles",
        chunk_num, total_chunks, total_edges, len(chunk),
    )

    return chunk_results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "url", "title", "source", "tier", "mentions_found", "entity_ids",
]


def get_csv_path() -> Path:
    """Build the timestamped CSV path in the mentions output directory."""
    today = pendulum.now().format("YYYY-MM-DD")
    output_dir = Path(settings.LOG_OUTPUT_DIR) / "mentions"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"mentions_{today}.csv"


def write_csv(results: list[dict]) -> None:
    """Write results to a timestamped CSV for audit."""
    csv_path = get_csv_path()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info("CSV written to %s (%d rows)", csv_path, len(results))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def log_summary(results: list[dict], elapsed: float) -> None:
    """Log summary statistics."""
    total_articles = len(results)
    if total_articles == 0:
        return

    total_mentions = sum(r["mentions_found"] for r in results)
    with_mentions = sum(1 for r in results if r["mentions_found"] > 0)
    without_mentions = total_articles - with_mentions

    logger.info("=" * 60)
    logger.info("MENTION INDEXER COMPLETE")
    logger.info("-" * 60)
    logger.info("  Articles processed:     %d", total_articles)
    logger.info(
        "  Articles with mentions: %d (%.0f%%)",
        with_mentions,
        with_mentions / total_articles * 100,
    )
    logger.info("  Articles with none:     %d", without_mentions)
    logger.info("  Total MENTIONS edges:   %d", total_mentions)
    logger.info(
        "  Avg mentions/article:   %.1f",
        total_mentions / total_articles,
    )
    logger.info(
        "  Duration:               %.1fs (%.2fs/article)",
        elapsed,
        elapsed / total_articles,
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — process all unprocessed articles."""
    run_start = time.time()

    with GraphStorage() as graph:
        # 1. Mark headline-only articles so they don't clog the queue
        skip_result = graph.run_query(SKIP_NO_SUMMARY_QUERY)
        skipped = skip_result[0]["skipped"] if skip_result else 0
        if skipped > 0:
            logger.info("Skipped %d headline-only articles (no summary)", skipped)

        # 2. Fetch unprocessed articles (with summaries)
        articles = fetch_raw_articles(graph)
        if not articles:
            logger.info("No unprocessed articles found. Nothing to do.")
            return

        # 3. Load entity list (once)
        entities = build_entity_list(graph)
        entity_list_text = format_entity_list(entities)
        valid_ids = {e["id"] for e in entities}
        logger.info("Loaded %d entities for matching", len(entities))

        # 4. Build chain (once)
        chain = build_mention_chain()

        # 5. Warm-up: test one article before committing to the batch
        test_input = build_chain_input(articles[0], entity_list_text)
        try:
            test_result = chain.invoke(test_input)
            validated = validate_mention_ids(test_result, valid_ids)
            logger.info(
                "Warm-up OK: '%s' -> %d mentions: %s",
                articles[0]["title"][:60],
                len(validated.mentioned_ids),
                ", ".join(validated.mentioned_ids),
            )
        except Exception as e:
            logger.error("Warm-up call FAILED: %s — aborting.", e)
            return

        # Process the warm-up article (create edges + update status)
        create_mentioned_in_edges(
            graph, articles[0]["url"], validated.mentioned_ids,
        )
        graph.run_query(UPDATE_STATUS_QUERY, {"urls": [articles[0]["url"]]})

        all_results = [build_result_row(articles[0], validated.mentioned_ids)]

        # 6. Process remaining articles in chunks
        remaining = articles[1:]
        if remaining:
            total_chunks = (len(remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for i in range(0, len(remaining), CHUNK_SIZE):
                chunk = remaining[i : i + CHUNK_SIZE]
                chunk_num = i // CHUNK_SIZE + 1

                try:
                    chunk_results = process_chunk(
                        graph, chunk, chain, entity_list_text,
                        valid_ids, chunk_num, total_chunks,
                    )
                    all_results.extend(chunk_results)

                    # Save progress after each chunk
                    write_csv(all_results)
                except Exception as e:
                    logger.error(
                        "Chunk %d/%d FAILED: %s — stopping. "
                        "%d articles saved from previous chunks.",
                        chunk_num, total_chunks, e, len(all_results),
                    )
                    break

        # 7. Final CSV write
        write_csv(all_results)

        # 8. Log summary
        elapsed = time.time() - run_start
        log_summary(all_results, elapsed)


if __name__ == "__main__":
    main()
