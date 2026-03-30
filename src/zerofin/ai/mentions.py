"""Entity mention indexer — identifies known entities in article text.

Uses LangChain + DeepSeek to identify which tracked entities an article
references. Creates MENTIONS edges in Neo4j linking articles to entities.

This is simpler than full relationship extraction — just "which of our
known entities does this article talk about?" No types, no directions,
no relationships.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from zerofin.ai.provider import get_llm
from zerofin.storage.graph import GraphStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity list builder
# ---------------------------------------------------------------------------

def build_entity_list(graph: GraphStorage) -> list[dict]:
    """Pull all tracked entities from Neo4j for mention matching.

    Returns a list of dicts with id, name, and label for each entity.
    """
    query = """
        MATCH (n)
        WHERE n.id IS NOT NULL AND n.name IS NOT NULL
          AND any(label IN labels(n) WHERE label IN [
              'Asset', 'Company', 'Index', 'Indicator', 'Commodity',
              'Currency', 'Country', 'CentralBank', 'GovernmentBody',
              'Sector', 'Person', 'Event'
          ])
        RETURN n.id AS id, n.name AS name, labels(n)[0] AS label
        ORDER BY n.name
    """
    entities = graph.run_query(query)
    logger.info("Loaded %d tracked entities from Neo4j", len(entities))
    return entities


def format_entity_list(entities: list[dict]) -> str:
    """Format entity list for the LLM prompt."""
    lines = []
    for e in entities:
        lines.append(f"- {e['name']} ({e['id']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class MentionResult(BaseModel):
    """List of entity IDs mentioned in an article."""

    mentioned_ids: list[str] = Field(
        description=(
            "List of entity IDs (tickers/codes) from the provided entity "
            "list that are mentioned or clearly referenced in the article. "
            "Use the exact ID from the list, not the name."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Given an article and a list of tracked entities, return the IDs of every \
entity from the list that the article explicitly names, directly \
references, or is clearly about.

Rules:
1. Only return IDs from the provided entity list. If an entity is not \
   in the list, do not return it.
2. Include an entity if the article names it, uses a common alias \
   (e.g. "the Fed" = Federal Reserve), or refers to its CEO/leader \
   by name (e.g. "Jensen Huang" = NVIDIA, "Jamie Dimon" = JPMorgan).
3. For commodities: "oil prices rose" = include crude oil entities. \
   "Iran war disrupts supply" = include crude oil (oil is the implied \
   commodity). "Energy crisis" = include crude oil and natural gas.
4. For central bank speeches: if a Federal Reserve official is the \
   speaker or subject, include Fed Funds rate indicators.
5. DO NOT include entities that are merely thematically related or \
   could theoretically be affected. A broad economic article about \
   inflation does NOT mean every stock, ETF, and commodity is mentioned. \
   Only include what the article explicitly discusses.
6. Keep the list focused. A typical article mentions 1-5 tracked entities. \
   If you are returning more than 10, you are probably overcalling.
7. If the article mentions no tracked entities, return an empty list."""

HUMAN_PROMPT = """\
<entities>
{entity_list}
</entities>

<article>
{article_text}
</article>"""

MENTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_mention_chain():
    """Build a LangChain chain for entity mention identification.

    Returns a chain that accepts entity_list and article_text,
    and returns a MentionResult.
    """
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(MentionResult)
    return MENTION_PROMPT | structured_llm


# ---------------------------------------------------------------------------
# Identification function
# ---------------------------------------------------------------------------

def find_mentions(
    article_text: str,
    entity_list_text: str,
    chain: object | None = None,
    valid_ids: set[str] | None = None,
) -> MentionResult:
    """Identify which tracked entities an article mentions.

    Args:
        article_text: Title + summary of the article.
        entity_list_text: Formatted entity list from format_entity_list().
        chain: Pre-built mention chain. If None, one is built internally.
            Pass a pre-built chain to avoid creating a new LLM client
            per article.
        valid_ids: Set of valid entity IDs. If provided, returned IDs
            are validated against this set and invalid ones are filtered
            out with a warning.

    Returns:
        MentionResult with list of entity IDs found. Returns empty
        result on failure.
    """
    if chain is None:
        chain = build_mention_chain()

    try:
        result = chain.invoke({
            "entity_list": entity_list_text,
            "article_text": article_text,
        })
    except Exception:
        logger.exception("Mention identification failed")
        return MentionResult(mentioned_ids=[])

    # Validate IDs if a valid set was provided
    if valid_ids is not None:
        result = validate_mention_ids(result, valid_ids)

    logger.info("Found %d entity mentions", len(result.mentioned_ids))
    return result


def validate_mention_ids(
    result: MentionResult,
    valid_ids: set[str],
) -> MentionResult:
    """Filter out IDs that don't exist in the entity list.

    Logs warnings for invalid IDs so we can track what the LLM
    is hallucinating.
    """
    valid = []
    for eid in result.mentioned_ids:
        if eid in valid_ids:
            valid.append(eid)
        else:
            logger.warning("Invalid entity ID returned by LLM: %s", eid)

    if len(valid) < len(result.mentioned_ids):
        dropped = len(result.mentioned_ids) - len(valid)
        logger.info("Dropped %d invalid entity IDs", dropped)

    return MentionResult(mentioned_ids=valid)


# ---------------------------------------------------------------------------
# Neo4j edge creation
# ---------------------------------------------------------------------------

def create_mentioned_in_edges(
    graph: GraphStorage,
    article_url: str,
    entity_ids: list[str],
) -> int:
    """Create MENTIONS edges from an article to mentioned entities.

    Args:
        graph: Connected GraphStorage instance.
        article_url: The article's URL (used as its Neo4j id).
        entity_ids: List of entity IDs from find_mentions().

    Returns:
        Number of edges created.
    """
    if not entity_ids:
        return 0

    query = """
        UNWIND $entity_ids AS eid
        MATCH (a:Article {id: $article_url})
        MATCH (e)
        WHERE e.id = eid
          AND any(l IN labels(e) WHERE l IN [
              'Asset', 'Company', 'Index', 'Indicator', 'Commodity',
              'Currency', 'Country', 'CentralBank', 'GovernmentBody',
              'Sector', 'Person', 'Event'
          ])
        MERGE (a)-[r:MENTIONS]->(e)
        SET r.matched_at = datetime()
        RETURN count(r) AS created
    """

    result = graph.run_query(query, {
        "article_url": article_url,
        "entity_ids": entity_ids,
    })

    created = result[0]["created"] if result else 0
    if created > 0:
        logger.debug(
            "Created %d MENTIONS edges for %s",
            created, article_url,
        )
    return created
