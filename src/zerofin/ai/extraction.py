"""Article entity and relationship extraction using Instructor + DeepSeek.

Two-step extraction pipeline:
1. Entity pass: extract entities from article text with type classification
2. Relationship pass: extract relationships using entity anchors from step 1

Uses Instructor for structured output with automatic validation and retry.
Pydantic models enforce our 12 entity types and 16 relationship types.
"""

from __future__ import annotations

import logging
from typing import Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from zerofin.config import settings
from zerofin.constants import ENTITY_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type definitions — must match our schema exactly
# ---------------------------------------------------------------------------

EntityType = Literal[
    "Asset",
    "Indicator",
    "Sector",
    "Event",
    "Company",
    "Index",
    "Commodity",
    "Currency",
    "Country",
    "CentralBank",
    "GovernmentBody",
    "Person",
]

RelationshipType = Literal[
    "CAUSES",
    "BENEFITS",
    "HURTS",
    "LEADS_TO",
    "DEPENDS_ON",
    "CORRELATES_WITH",
    "PREDICTS",
    "BELONGS_TO",
    "HOLDS",
    "REGULATES",
    "LOCATED_IN",
    "COMPETES_WITH",
    "SUPPLIES_TO",
    "SUBSIDIARY_OF",
    "DISRUPTS",
    "PARTNERS_WITH",
    "INVESTS_IN",
]


# ---------------------------------------------------------------------------
# Step 1: Entity extraction models
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """A single entity found in an article."""

    text: str = Field(
        description="The exact text span from the article that refers to this entity.",
    )
    reasoning: str = Field(
        description=(
            "Brief reasoning about classification. Check: Is this a specific "
            "named entity (not a generic term like 'investors')? Is it a "
            "sovereign nation (not a region like 'Middle East')? Is it a "
            "company (discussing business) or asset (discussing price/trading)? "
            "Conclude with the correct type or 'skip' if not an entity."
        ),
    )
    entity_type: EntityType = Field(
        description="The type of entity. Must be one of the allowed types.",
    )
    canonical_name: str = Field(
        description=(
            "The most formal, specific official name. "
            "'Apple' → 'Apple Inc.', 'the Fed' → 'Federal Reserve', "
            "'oil' → 'Crude Oil', 'Goldman' → 'Goldman Sachs Group'."
        ),
    )

    @field_validator("entity_type", mode="before")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Handle common casing issues from LLM output."""
        if isinstance(v, str):
            # Try title case first (most entity types are title case)
            title = v.strip().title().replace(" ", "")
            if title in ENTITY_LABELS:
                return title
            # Try exact match
            for label in ENTITY_LABELS:
                if v.strip().lower() == label.lower():
                    return label
        return v


class EntityExtractionResult(BaseModel):
    """Output from step 1 — all entities found in an article."""

    entities: list[ExtractedEntity] = Field(
        description="All financial entities mentioned in the article.",
    )
    extraction_quality: Literal["high", "medium", "low"] = Field(
        description=(
            "Quality assessment. 'high' = full article with clear entity mentions. "
            "'medium' = summary with some entities identifiable. "
            "'low' = headline or vague text with uncertain entity identification."
        ),
    )


# ---------------------------------------------------------------------------
# Step 2: Relationship extraction models
# ---------------------------------------------------------------------------

class ExtractedRelationship(BaseModel):
    """A single relationship between two entities found in an article."""

    subject: str = Field(
        description="Canonical name of the source entity (the one doing the action).",
    )
    object: str = Field(
        description="Canonical name of the target entity (the one being acted upon).",
    )
    reasoning: str = Field(
        description=(
            "MANDATORY direction check. Answer these three questions: "
            "1) Which entity is the DRIVER (the one causing, providing, or "
            "initiating the action)? 2) Which entity is the RECEIVER (the one "
            "being affected, supplied, or acted upon)? 3) Does the article "
            "explicitly state or strongly imply this connection? "
            "The DRIVER must be the subject and the RECEIVER must be the object. "
            "If you cannot clearly identify driver vs receiver, use "
            "CORRELATES_WITH or skip entirely."
        ),
    )
    relationship_type: RelationshipType = Field(
        description="The type of relationship. Must be one of the allowed types.",
    )
    confidence: float = Field(
        description=(
            "How confident you are that this relationship is stated or clearly "
            "implied in the article. 0.9 = explicitly stated. 0.5 = implied "
            "but ambiguous."
        ),
    )
    evidence: str = Field(
        description=(
            "One sentence from or paraphrasing the article "
            "that supports this relationship."
        ),
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp to [0,1] and reject below minimum threshold.

        When this raises ValueError, Instructor retries with the error
        message, giving the LLM a chance to either raise the confidence
        or drop the relationship.
        """
        if not isinstance(v, (int, float)):
            return v
        clamped = max(0.0, min(1.0, float(v)))
        if clamped < settings.MIN_RELATIONSHIP_CONFIDENCE:
            raise ValueError(
                f"Confidence {clamped:.2f} is below the minimum threshold "
                f"of {settings.MIN_RELATIONSHIP_CONFIDENCE}. Do not extract "
                f"relationships you are not confident about."
            )
        return clamped

    @field_validator("relationship_type", mode="before")
    @classmethod
    def normalize_relationship_type(cls, v: str) -> str:
        """Handle casing and common LLM formatting issues."""
        if isinstance(v, str):
            return v.strip().upper().replace(" ", "_")
        return v


class RelationshipExtractionResult(BaseModel):
    """Output from step 2 — relationships between known entities."""

    relationships: list[ExtractedRelationship] = Field(
        description="Relationships between entities found in the article.",
    )
    skipped: bool = Field(
        default=False,
        description=(
            "True if the article doesn't contain enough context to reliably "
            "determine relationships. Set to true for very short summaries "
            "or vague articles."
        ),
    )
    skip_reason: str = Field(
        default="",
        description="Why relationship extraction was skipped, if applicable.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ENTITY_SYSTEM_PROMPT = """\
Extract financial entities from the article provided and classify each \
one according to the entity type definitions below. For each entity, \
reason about its classification before assigning a type.

<examples>
Example 1 — Correct extraction:
Article: "NVIDIA reported record revenue. TSMC, its chip manufacturer, \
raised capacity."
Entities:
- "NVIDIA" → reasoning: "Discussing business operations (revenue). \
  This is a Company." → Company, canonical: "NVIDIA Corporation"
- "TSMC" → reasoning: "Discussing business role (manufacturer). \
  This is a Company." → Company, canonical: "Taiwan Semiconductor \
  Manufacturing Company"

Example 2 — Correct extraction with tricky types:
Article: "The Federal Reserve held rates steady. Treasury yields fell. \
The S&P 500 rallied 2%. The dollar weakened."
Entities:
- "Federal Reserve" → reasoning: "Monetary authority that sets rates. \
  CentralBank." → CentralBank, canonical: "Federal Reserve"
- "Treasury yields" → reasoning: "Economic data series. Indicator." \
  → Indicator, canonical: "Treasury Yield"
- "S&P 500" → reasoning: "Market benchmark tracking stocks. Not tradable \
  itself. Index." → Index, canonical: "S&P 500"
- "dollar" → reasoning: "National currency. Currency." → Currency, \
  canonical: "US Dollar"

Example 3 — Contrastive (showing what NOT to extract):
Article: "Oil prices surged as tensions in the Middle East escalated. \
Asian refiners scrambled for supply. Investors fled to safe havens."

WRONG extraction:
- "Middle East" → Country ← WRONG. Middle East is a region, not a \
  sovereign nation. Skip it entirely.
- "Asian" → Country ← WRONG. Asia is a continent. Skip it.
- "Investors" → Person ← WRONG. Generic term, not a named individual.

CORRECT extraction:
- "Oil" → reasoning: "Raw material commodity. Not an ETF or futures \
  contract name." → Commodity, canonical: "Crude Oil"
(Only one entity here. The rest are regions or generic terms.)

Example 4 — Contrastive (REGULATES vs recommendations):
Article: "The IEA recommended that consumers reduce energy usage."

WRONG: IEA → GovernmentBody (this is correct), but do NOT create a \
REGULATES relationship later. The IEA is recommending, not regulating. \
It has no legal enforcement authority over consumers.
</examples>

<entity_types>

Company:
  A corporate entity that operates a business.
  Use when discussing operations, earnings, strategy, or management.
  Examples: "Apple Inc.", "Goldman Sachs Group", "TSMC"
  Note: "Apple reported earnings" = Company. "Apple stock fell" = Asset.

Asset:
  A tradable financial instrument — stocks, ETFs, bonds, futures contracts.
  Use when referring to something bought/sold on an exchange.
  Examples: "NVDA stock", "SPY ETF", "TLT bond ETF"
  Note: Futures contracts (WTI futures, gold futures) are Assets.

Index:
  A market benchmark that tracks a basket of securities. Not tradable.
  Examples: "S&P 500", "Nasdaq 100", "Dow Jones", "Russell 2000"
  Note: SPY is an Asset (tradable ETF). The S&P 500 is an Index (benchmark).

Indicator:
  A macroeconomic or market data series that measures the economy.
  Examples: "CPI", "unemployment rate", "10-year Treasury yield", "GDP", \
  "rig count", "initial jobless claims", "consumer confidence"

Commodity:
  A raw material or natural resource — the physical substance.
  Examples: "crude oil", "gold", "copper", "wheat", "natural gas"
  Note: "oil prices", "oil", "crude", "WTI crude", "Brent crude" all \
  refer to oil commodities. Different grades (WTI, Brent, Dubai) are \
  separate commodities.
  NOT a Commodity: "USO" (that's an Asset/ETF).

Sector:
  An industry classification or market segment.
  Use when the article refers to a whole industry, not a specific company.
  Examples: "technology sector", "energy", "healthcare", "financials"
  NOT a Sector: geographic regions. "Middle East" is NOT a Sector.

Currency:
  A national currency or cryptocurrency.
  Examples: "US dollar", "euro", "yen", "Bitcoin", "Ethereum"

Country:
  A specific sovereign nation with an internationally recognized government.
  Extract ONLY sovereign nations: "China", "Japan", "Germany", "Saudi Arabia", \
  "United States", "Iran", "India", "Brazil".
  NEVER extract these — they are regions, not countries:
  - "Asia", "Middle East", "Persian Gulf", "Europe", "Latin America"
  - "Gulf states", "ASEAN", "GCC", "Sub-Saharan Africa"
  NEVER extract US states as Country:
  - Texas, Oklahoma, California, Washington, Florida, New York, Ohio, \
    Pennsylvania, and all other US states are NOT countries. Skip them.
  If you see a region or state name, skip it entirely.

CentralBank:
  A monetary authority that sets interest rates or manages money supply.
  Examples: "Federal Reserve", "ECB", "Bank of Japan"
  Aliases: "the Fed" = "Federal Reserve", "FOMC" = "Federal Reserve"

GovernmentBody:
  A regulatory agency or intergovernmental organization.
  Examples: "SEC", "OPEC", "IEA", "FDA", "European Commission"
  Note: OPEC is GovernmentBody, not Company.

Event:
  A specific occurrence affecting markets.
  Examples: "FOMC rate decision", "Strait of Hormuz disruption", \
  "tariff announcement", "earnings report"
  Use for things that happen, not standing institutions or regions.

Person:
  A named individual relevant to markets.
  Examples: "Jerome Powell", "Warren Buffett", "Jamie Dimon"
  NEVER extract generic terms: "investors", "analysts", "consumers" \
  are NOT Person entities.

</entity_types>

<instructions>
1. Extract ONLY entities explicitly mentioned in the article.
2. Use the reasoning field to check your classification before committing.
3. Do not extract regions, continents, or generic terms.
6. Do not extract the article's publisher or source as an entity. \
   BBC, CNBC, Reuters, MarketWatch, Bloomberg, SCMP, etc. are news \
   outlets reporting the story — they are not financial entities in it.
4. Use the most formal canonical name for each entity.
5. Different text mentions of the same real-world entity = ONE object in \
   your output list. If you assign two entities the same canonical_name, \
   that is a bug — merge them into one and use the first text span as \
   the "text" field. Do not return two objects with the same canonical_name.
</instructions>
"""

RELATIONSHIP_SYSTEM_PROMPT = """\
Extract relationships between the provided entities based on what the \
article states or strongly implies. Use the reasoning field to verify \
your classification before committing.

<examples>
Example 1 — Supply chain:
Entities: [NVIDIA Corporation (Company), TSMC (Company)]
Article: "NVIDIA reported record earnings. Its supplier TSMC announced \
capacity expansion."
Relationships:
- TSMC → NVIDIA Corporation
  reasoning: "Article says 'its supplier TSMC' — direct supplier-customer. \
  SUPPLIES_TO is correct. Direction: TSMC supplies to NVIDIA, not reverse."
  → SUPPLIES_TO (conf: 0.90)

Example 2 — Causal with correct direction:
Entities: [OPEC (GovernmentBody), Crude Oil (Commodity), Energy (Sector)]
Article: "OPEC cut production, sending oil past $100. Energy stocks rallied."
Relationships:
- OPEC → Crude Oil
  reasoning: "OPEC's action directly caused the price surge. CAUSES, not \
  CORRELATES_WITH — there's a direct causal mechanism described."
  → CAUSES (conf: 0.95)
- Crude Oil → Energy
  reasoning: "Oil price rise led to energy stocks rallying. BENEFITS — \
  when oil rises, energy sector does better."
  → BENEFITS (conf: 0.80)

Example 3 — Contrastive (common mistakes):
Entities: [IEA (GovernmentBody), Crude Oil (Commodity)]
Article: "The IEA recommended consumers reduce energy consumption."

WRONG: IEA REGULATES Crude Oil ← WRONG. The IEA has no legal authority. \
It is recommending, not regulating. Recommendations ≠ regulation. \
Only use REGULATES when the subject has formal statutory enforcement \
power (SEC, Federal Reserve, FDA).

CORRECT: No relationship extracted. The IEA's recommendation does not \
constitute a relationship with Crude Oil.

Example 4 — Contrastive (LOCATED_IN restrictions):
Entities: [Crude Oil (Commodity), Saudi Arabia (Country)]
Article: "Saudi Arabian oil production fell."

WRONG: Crude Oil LOCATED_IN Saudi Arabia ← WRONG. Commodities are not \
"located in" countries. LOCATED_IN is ONLY for Company/Institution → Country.

CORRECT: Saudi Arabia SUPPLIES_TO Crude Oil (conf: 0.80) — Saudi Arabia \
is an oil producer, supplying crude to the global market.

Example 5 — Contrastive (data reporting ≠ financial relationship):
Entities: [Baker Hughes Company (Company), Baker Hughes Rig Count (Indicator), \
U.S. Energy Information Administration (GovernmentBody)]
Article: "Baker Hughes reported rig count fell to 543. The latest EIA data \
shows production declining."

WRONG:
- Baker Hughes HOLDS Rig Count ← WRONG. Publishing data is not owning a \
  financial position. HOLDS is only for equity/debt/investment stakes.
- EIA REGULATES Oil & Gas ← WRONG. EIA collects and publishes data. It \
  has no enforcement authority. It does not write rules or impose fines.
- Rig Count DEPENDS_ON United States ← WRONG. That is just geographic \
  scope, not a critical operational dependency.

CORRECT: No relationships extracted. The article describes data being \
published, not business relationships between these entities.

Example 6 — Contrastive (SUPPLIES_TO direction and partnerships):
Entities: [Shandong Airlines (Company), Boeing (Company), Apple Inc. (Company), \
Google (Company)]
Article A: "Shandong Airlines announced plans to lease 10 Boeing 737 aircraft."
WRONG: Shandong Airlines SUPPLIES_TO Boeing ← WRONG. Boeing makes the planes. \
Shandong Airlines is the BUYER.
CORRECT: Boeing SUPPLIES_TO Shandong Airlines (conf: 0.90)

Article B: "Apple plans to open Siri to competing AI assistants including Google."
WRONG: Apple SUPPLIES_TO Google ← WRONG. Opening a platform is a partnership, \
not a supply transaction.
CORRECT: Apple Inc. PARTNERS_WITH Google (conf: 0.85) — mutual platform deal.

Example 7 — Contrastive (military conflict ≠ financial relationship):
Entities: [United States (Country), Israel (Country), Iran (Country), \
Crude Oil (Commodity)]
Article: "U.S. and Israeli forces conducted a joint strike on Iranian \
nuclear facilities. Crude oil surged 4% on supply disruption fears."

WRONG:
- United States COMPETES_WITH Iran ← WRONG. Military strike is not \
  market competition.
- United States SUPPLIES_TO Israel ← WRONG. "Joint forces" is not a \
  supply relationship.

CORRECT:
- United States → Crude Oil
  reasoning: "The strike created supply disruption fears. This is a \
  geopolitical disruption of commodity flow. DISRUPTS is correct."
  → DISRUPTS (conf: 0.85)
- Israel → Crude Oil
  reasoning: "Israel participated in the strike, contributing to the \
  supply disruption."
  → DISRUPTS (conf: 0.80)
</examples>

<relationship_types>

Causal (one thing makes another happen):
- CAUSES: A directly triggers B. Must describe a causal mechanism.
- DISRUPTS: A (an Event, Country, or GovernmentBody) interrupts or blocks \
  the normal flow of B (a Commodity, supply chain, or economic activity). \
  Use for: military conflicts, sanctions, blockades, port closures, \
  production outages. The disruption must be physical or regulatory — \
  not a price move. NOT a DISRUPTS: "rising rates hurt stocks" = HURTS.
- BENEFITS: When A happens/rises, B does better.
- HURTS: When A happens/rises, B does worse.
- LEADS_TO: A comes before B with a time delay.

Dependency/Structural:
- DEPENDS_ON: A critically requires B to operate. A structural dependency \
  where B failing would directly break A. "NVIDIA depends on TSMC" = yes. \
  "Rig count depends on United States" = NO — that is just geographic scope. \
  DO NOT use DEPENDS_ON for: geographic location, data sources, or loose \
  associations. If in doubt, it's probably not DEPENDS_ON.
- BELONGS_TO: A is a member or part of B. A company belongs to a sector. \
  A subsidiary belongs to a parent. An indicator belongs to a data category.
- HOLDS: A owns a financial position (equity, debt, stake) in B. \
  ONLY for investment/ownership relationships. "Berkshire holds Apple stock" \
  = yes. "Baker Hughes publishes rig count" = NO — publishing data is not \
  a financial position. DO NOT use HOLDS for data reporting, research \
  coverage, or monitoring.
- SUBSIDIARY_OF: A is owned by B as a subsidiary. Direction: subsidiary → parent.
- LOCATED_IN: ONLY for Company/CentralBank/GovernmentBody → Country. \
  NEVER for commodities, events, indicators, or regions.

Business/Competitive:
- COMPETES_WITH: A and B fight for the same market or customers.
- SUPPLIES_TO: A sells products, goods, or services to B. Must be a \
  commercial transaction. Direction: supplier → buyer. "Boeing sells planes \
  to airlines" = Boeing SUPPLIES_TO airline. NOT the reverse. \
  Before using SUPPLIES_TO, ask: "Who receives goods/services from whom?" \
  The entity PROVIDING is the subject. The entity RECEIVING must be an \
  institution (Company, Country, GovernmentBody) — NEVER a Commodity. \
  A producer cannot SUPPLIES_TO the commodity it produces. \
  "Aker BP produces crude oil" ≠ Aker BP SUPPLIES_TO Crude Oil. \
  The commodity is the PRODUCT, not the customer. Skip or use a different type. \
  DO NOT use SUPPLIES_TO for: analyst coverage, research reports, \
  price targets, recommendations, or platform access.
- PARTNERS_WITH: A and B have a mutual business relationship — joint \
  venture, strategic alliance, platform deal, technology licensing, or \
  distribution agreement. Both parties benefit. Use when the relationship \
  is collaborative, not one-directional supply. \
  "Apple opens Siri to Google Gemini" = Apple PARTNERS_WITH Google. \
  "Palantir and AWS co-develop government AI" = Palantir PARTNERS_WITH Amazon.
- INVESTS_IN: A puts capital into B — equity stake, funding round, \
  acquisition, or debt issuance. Must involve money flowing from A to B \
  as an investment. "NYSE invests $600M in Polymarket" = yes. \
  "US agency takes 20% stake in Syrah" = yes. \
  DO NOT use for: payments for services, compensation, contracts, or \
  deals where money is exchanged for goods (that's SUPPLIES_TO).
- REGULATES: A has formal statutory enforcement authority over B. \
  A must be able to write rules, impose fines, or enforce compliance. \
  SEC, Federal Reserve, FDA, SAMR = regulators. \
  NOT regulators: EIA, IEA, Baker Hughes, BEA, OECD, IMF = data reporters \
  or research bodies. Analysts, banks, rating agencies = NOT regulators. \
  Statistical agencies that MEASURE or PUBLISH data do NOT regulate the \
  metrics they report. BEA publishing GDP ≠ BEA REGULATES GDP. \
  Filing a disclosure, listing on an exchange, or lobbying a government \
  is NOT regulation.

Statistical:
- CORRELATES_WITH: A and B move together statistically — their prices \
  or data series co-vary over time. ONLY use when the article describes \
  two things moving in tandem with no clear causal direction. \
  "Oil and gas prices move together" = yes. \
  NOT CORRELATES_WITH: institutional oversight, index rankings, being \
  listed on an exchange, recommendations, or any relationship where one \
  entity clearly drives the other (use CAUSES/BENEFITS/HURTS instead).
- PREDICTS: When A does X, B tends to follow. Requires directional \
  evidence with a time component.

</relationship_types>

<instructions>
1. Both subject and object MUST be from the provided entity list.
2. Only extract relationships the article states or strongly implies.
3. DIRECTION CHECK (mandatory for every relationship): Identify the DRIVER \
   (the entity causing, providing, or initiating) and the RECEIVER (the \
   entity being affected, supplied, or acted upon). DRIVER = subject, \
   RECEIVER = object. Common mistakes to avoid: \
   "Airlines buy planes from Boeing" → Boeing is DRIVER (supplier), airline \
   is RECEIVER. "Rising oil hurts airlines" → Oil is DRIVER, airlines is \
   RECEIVER. "China's demand boosts copper" → China is DRIVER, copper is \
   RECEIVER. Write your direction check in the reasoning field.
4. Confidence: 0.9+ = explicitly stated, 0.7-0.9 = strongly implied. \
   Below 0.7 = do not extract the relationship.
5. If the article is too short or vague for reliable extraction, set \
   skipped=true.
6. Prefer specific types. "X supplies to Y" = SUPPLIES_TO, not \
   CORRELATES_WITH.
7. CORRELATES_WITH requires evidence of statistical co-movement. It is \
   NOT for institutional oversight, monitoring, or advisory relationships. \
   If an institution monitors a market without co-moving with it, extract \
   nothing.
8. Military conflicts, sanctions, and blockades = DISRUPTS, not \
   COMPETES_WITH or SUPPLIES_TO.
9. If no relationship type fits, extract nothing. Do not force a financial \
   type onto a geopolitical event.
10. Both subject and object MUST exactly match a canonical_name from the \
    entity list. Do not use entity names that were not extracted. If the \
    entity was not extracted, do not create a relationship with it.
11. Do not extract relationships involving the article's own publisher \
    or source (BBC, CNBC, Reuters, etc.). News outlets reporting on \
    events are not participants in those events.
12. SKIP procedural actions. If the relationship is about filing, \
    disclosing, lobbying, announcing, coordinating, urging, or listing, \
    extract nothing. Only extract relationships that describe direct \
    financial or operational impact between two entities. \
    "Company filed a notice with exchange" = skip. \
    "Company urged government to change policy" = skip. \
    "Company announced joint findings with agency" = skip.
13. Analyst ratings, price targets, and research coverage = PREDICTS, \
    not CORRELATES_WITH. "TipRanks rates VOOG as Strong Buy" = \
    TipRanks PREDICTS VOOG. "Macquarie says copper is overpriced" = \
    Macquarie PREDICTS Copper.
</instructions>
"""


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def _get_instructor_client() -> instructor.Instructor:
    """Create an Instructor client using DeepSeek via OpenRouter."""
    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    base_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return instructor.from_openai(base_client)


# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

RelevanceLabel = Literal["EXTRACT", "SKIP", "UNCERTAIN"]


class RelevanceResult(BaseModel):
    """Quick relevance check on an article before spending extraction calls."""

    reasoning: str = Field(
        description=(
            "One sentence: does this article contain concrete financial "
            "information (earnings, prices, deals, policy changes, supply "
            "chain events, economic data) or is it opinion/commentary/lifestyle?"
        ),
    )
    relevance: RelevanceLabel = Field(
        description=(
            "EXTRACT = concrete financial content worth extracting. "
            "SKIP = opinion, commentary, lifestyle, no actionable financial data. "
            "UNCERTAIN = might contain something, extract to be safe."
        ),
    )


RELEVANCE_PROMPT = """\
You are a financial news filter. Given an article title and summary, \
determine if the article contains concrete, extractable financial \
information or if it is opinion, commentary, or fluff.

EXTRACT — the article discusses specific:
- Earnings, revenue, or financial results
- Price movements with named assets/commodities
- Mergers, acquisitions, partnerships, or deals
- Policy decisions (rate changes, sanctions, tariffs, regulations)
- Supply chain events (production changes, disruptions, new facilities)
- Economic data releases (CPI, jobs, GDP, housing data)
- Company strategy changes with specific details

SKIP — the article is:
- General opinion or market commentary without specific data
- Anniversary/milestone pieces ("Apple turns 50")
- Lifestyle, personal finance, or consumer advice
- Vague warnings or predictions without concrete claims
- Roundup/digest articles that just list headlines

UNCERTAIN — the article might have extractable content but you're \
not sure from the title/summary alone. Err on the side of UNCERTAIN \
over SKIP — it's cheaper to extract from a borderline article than \
to miss real financial data.

Be aggressive with SKIP. Most opinion pieces and commentary add \
nothing to a financial knowledge graph.
"""


def check_relevance(title: str, summary: str = "") -> RelevanceResult:
    """Quick relevance filter before spending extraction API calls.

    Args:
        title: Article title.
        summary: Article summary (can be empty for headline-only).

    Returns:
        RelevanceResult with EXTRACT, SKIP, or UNCERTAIN label.
    """
    client = _get_instructor_client()

    text = title
    if summary:
        text += f"\n{summary[:200]}"  # First 200 chars of summary is enough

    result = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        temperature=0.1,
        messages=[
            {"role": "system", "content": RELEVANCE_PROMPT},
            {"role": "user", "content": text},
        ],
        response_model=RelevanceResult,
    )

    logger.info("Relevance: %s (%s)", result.relevance, result.reasoning[:80])
    return result


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_entities(
    article_text: str,
    existing_entities: str = "",
) -> EntityExtractionResult:
    """Step 1: Extract entities from article text.

    Args:
        article_text: The full article or summary text.
        existing_entities: Optional context block listing entities already
            in the knowledge graph, to help the LLM use canonical names.

    Returns:
        EntityExtractionResult with list of extracted entities.
    """
    client = _get_instructor_client()

    user_content = f"Extract all financial entities from this article:\n\n{article_text}"
    if existing_entities:
        user_content += (
            f"\n\nThe following entities already exist in our knowledge graph. "
            f"If an entity in the article matches one of these, use the exact "
            f"canonical name shown:\n{existing_entities}"
        )

    result = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        temperature=0.1,
        messages=[
            {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_model=EntityExtractionResult,
    )

    logger.info(
        "Extracted %d entities (quality: %s)",
        len(result.entities),
        result.extraction_quality,
    )
    return result


def extract_relationships(
    article_text: str,
    entities: list[ExtractedEntity],
) -> RelationshipExtractionResult:
    """Step 2: Extract relationships between known entities.

    Args:
        article_text: The full article or summary text.
        entities: Entity list from step 1 — the LLM must use these
            canonical names as subject/object.

    Returns:
        RelationshipExtractionResult with list of relationships.
    """
    client = _get_instructor_client()

    entity_list = "\n".join(
        f"- {e.canonical_name} [{e.entity_type}]"
        for e in entities
    )

    user_content = (
        f"Given this article and the entities identified in it, extract "
        f"the relationships between them.\n\n"
        f"ENTITIES FOUND:\n{entity_list}\n\n"
        f"ARTICLE:\n{article_text}"
    )

    result = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        temperature=0.1,
        messages=[
            {"role": "system", "content": RELATIONSHIP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_model=RelationshipExtractionResult,
    )

    if result.skipped:
        logger.info("Relationship extraction skipped: %s", result.skip_reason)
    else:
        logger.info("Extracted %d relationships", len(result.relationships))
    return result


def deduplicate_entities(
    entities: list[ExtractedEntity],
) -> list[ExtractedEntity]:
    """Remove duplicate entities by canonical_name, keeping first occurrence."""
    seen: dict[str, ExtractedEntity] = {}

    for entity in entities:
        key = entity.canonical_name.strip().lower()
        if key not in seen:
            seen[key] = entity
        else:
            logger.debug(
                "Dedup: dropped '%s' (duplicate of '%s', canonical='%s')",
                entity.text,
                seen[key].text,
                entity.canonical_name,
            )

    removed = len(entities) - len(seen)
    if removed > 0:
        logger.info("Dedup removed %d duplicate entities", removed)

    return list(seen.values())


def deduplicate_relationships(
    relationships: list[ExtractedRelationship],
) -> list[ExtractedRelationship]:
    """Remove duplicate relationships by (subject, object, type) triple."""
    seen: set[tuple[str, str, str]] = set()
    result: list[ExtractedRelationship] = []

    for rel in relationships:
        key = (
            rel.subject.strip().lower(),
            rel.object.strip().lower(),
            rel.relationship_type,
        )
        if key not in seen:
            seen.add(key)
            result.append(rel)
        else:
            logger.debug(
                "Dedup: dropped duplicate relationship %s→%s %s",
                rel.subject, rel.object, rel.relationship_type,
            )

    return result


def extract_from_article(
    article_text: str,
    existing_entities: str = "",
    skip_relationships_under: int = 50,
) -> tuple[EntityExtractionResult, RelationshipExtractionResult | None]:
    """Full two-step extraction from a single article.

    Args:
        article_text: The article text to extract from.
        existing_entities: Optional context of known graph entities.
        skip_relationships_under: Skip relationship extraction if the
            article has fewer words than this threshold.

    Returns:
        Tuple of (entity_result, relationship_result). The relationship
        result is None if the article was too short.
    """
    # Step 1: Extract entities
    entity_result = extract_entities(article_text, existing_entities)

    # Post-processing: deduplicate entities
    entity_result.entities = deduplicate_entities(entity_result.entities)

    if not entity_result.entities:
        logger.info("No entities found — skipping relationship extraction")
        return entity_result, None

    # Skip relationship extraction for very short articles
    word_count = len(article_text.split())
    if word_count < skip_relationships_under:
        logger.info(
            "Article too short for relationships (%d words < %d threshold)",
            word_count, skip_relationships_under,
        )
        return entity_result, None

    # Step 2: Extract relationships using entity anchors
    rel_result = extract_relationships(article_text, entity_result.entities)

    # Post-processing: deduplicate relationships
    if rel_result and not rel_result.skipped:
        rel_result.relationships = deduplicate_relationships(rel_result.relationships)

    return entity_result, rel_result
