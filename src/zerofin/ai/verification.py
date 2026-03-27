"""Correlation verification pipeline — LLM judges whether statistical
relationships are financially plausible.

Takes pending_verification relationships from Neo4j, asks an LLM
to evaluate whether each one reflects a real economic relationship
or is statistical noise.

The prompt is designed around research from 2025-2026 on LLM calibration:
- Null hypothesis framing (skeptical default, not affirmative)
- 5-point categorical verdict instead of continuous 0-1 score
- Mechanism-first reasoning (name the pathway before the verdict)
- Few-shot examples spanning the full range including negatives
- Temperature 0.1 (not 0.0 — deterministic ≠ calibrated)
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from zerofin.ai.provider import get_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

Verdict = Literal[
    "confirmed_plausible",
    "likely_plausible",
    "uncertain",
    "likely_spurious",
    "confirmed_spurious",
]

RelationshipCategory = Literal[
    "sector_peer",
    "supply_chain",
    "macro_sensitivity",
    "competition",
    "risk_factor",
    "etf_composition",
    "none",
]


class VerificationResult(BaseModel):
    """The LLM's structured judgment on a single correlation."""

    mechanism: str = Field(
        description=(
            "The specific economic mechanism that would explain this "
            "relationship, or 'No direct mechanism identified' if none exists."
        ),
    )
    alternative_explanations: str = Field(
        description=(
            "Common factors, confounds, or coincidences that could produce "
            "this correlation without a direct relationship."
        ),
    )
    verdict: Verdict = Field(
        description="Categorical judgment on plausibility. Must be lowercase.",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def lowercase_verdict(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the verdict. 0.5 = genuinely uncertain. "
            "Above 0.8 only when mechanism is well-documented and unambiguous. "
            "Below 0.3 when no mechanism is evident."
        ),
    )
    reasoning: str = Field(
        description="1-2 sentence summary of why this verdict is correct.",
    )
    relationship_category: RelationshipCategory = Field(
        description="Type of financial relationship, or 'none' if spurious. Must be lowercase.",
    )

    @field_validator("relationship_category", mode="before")
    @classmethod
    def lowercase_category(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a quantitative research analyst specializing in spurious relationship \
detection. Your mandate is to identify statistical artifacts in financial data \
that, if treated as real, would generate false trading signals.

CONTEXT:
- Partial correlations measure co-movement AFTER removing the effects of all \
other variables in the model, including broad market factors and sector ETFs.
- The range 0.10-0.15 is weak but meaningful. Because common factors have \
already been removed, even small residual correlations can reflect genuine \
direct relationships. Do NOT dismiss a clear mechanism just because the \
magnitude is low.
- "Plausible" requires a specific, identifiable economic mechanism — not just \
theoretical possibility.
- A relationship driven by a removed common factor (e.g., both respond to Fed \
policy) is NOT a direct relationship — that signal was already partialled out.
- DIRECT vs INDIRECT: A one-hop relationship (e.g., Tesla buys lithium, \
Deere sells to corn farmers) is DIRECT. A three-hop chain through multiple \
intermediaries is indirect. Do not downgrade one-hop supply chains.
- SAME-INDUSTRY PEERS: Companies in the same broad industry share regulatory, \
institutional, and policy dynamics that count as sector_peer mechanisms — \
even when their specific products or business models differ.
- FACTOR EXPOSURE: Shared secondary factor exposure (yield sensitivity, size \
factor, cyclicality, defensive characteristics) is a valid macro_sensitivity \
mechanism after partial correlation removes broad market effects.
- COMPOSITIONAL: Part-whole relationships (ETF contains another ETF, index \
contains another index, instrument tracks the same underlying) are valid \
etf_composition relationships, not spurious.
- VALUE CHAIN: Entities connected through a production value chain (raw \
material to processor, producer to refiner, manufacturer to distributor) \
have a direct supply_chain relationship even when their business models \
and revenue drivers differ.
- GEOGRAPHIC PRODUCTION: When a country is a major producer of a commodity, \
country indices have a valid compositional or macro_sensitivity link to \
that commodity.

THE NULL HYPOTHESIS: This correlation is noise. You are looking for evidence \
strong enough to reject that null hypothesis. If you cannot articulate a \
direct mechanism in one sentence, your verdict must be UNCERTAIN or SPURIOUS.

VERDICT SCALE:
- CONFIRMED_PLAUSIBLE: Well-documented mechanism with strong precedent. \
Reserve for relationships where the economic link is established in \
literature or industry practice.
- LIKELY_PLAUSIBLE: Clear mechanism but weaker evidence. The pathway is \
identifiable and direct, but correlation magnitude is weak or historical \
evidence is limited.
- UNCERTAIN: A theoretical mechanism may exist but evidence is ambiguous. \
THIS IS THE CORRECT DEFAULT for partial correlations of 0.10-0.15 when \
the mechanism is indirect or debatable.
- LIKELY_SPURIOUS: No clear mechanism, or the most likely explanation is a \
confound or coincidence. Use when you can explain the correlation WITHOUT \
invoking a direct relationship.
- CONFIRMED_SPURIOUS: No conceivable mechanism. Entities are from unrelated \
sectors with no economic pathway. If you cannot identify ANY mechanism, use \
this verdict — do not hedge with LIKELY_SPURIOUS. However, if you do not \
recognize an entity, are uncertain about its sector, or cannot determine \
what it does, use UNCERTAIN instead — never CONFIRMED_SPURIOUS for pairs \
where you lack knowledge about one or both entities.

CALIBRATION EXAMPLES:

Example 1 — CONFIRMED_PLAUSIBLE (confidence: 0.92)
Entities: XOM (ExxonMobil) <-> CL=F (WTI Crude Oil Futures)
Partial correlation: 0.14, positive
Mechanism: ExxonMobil's revenue is directly driven by crude oil prices. \
This is a textbook commodity-producer relationship documented in hundreds \
of studies.
Alternative explanations: Could be partially driven by energy sector \
sentiment, but the direct revenue linkage exists independently.
Verdict: CONFIRMED_PLAUSIBLE
Category: supply_chain

Example 2 — LIKELY_PLAUSIBLE (confidence: 0.65)
Entities: CAT (Caterpillar) <-> HG=F (Copper Futures)
Partial correlation: 0.11, positive
Mechanism: Caterpillar sells heavy equipment to mining and construction \
sectors that consume copper. Higher copper prices signal construction \
activity that drives equipment demand.
Alternative explanations: Both could respond to global growth expectations \
without a direct link. The connection is real but indirect — 2 hops through \
construction activity.
Verdict: LIKELY_PLAUSIBLE
Category: supply_chain

Example 3 — UNCERTAIN (confidence: 0.45)
Entities: NFLX (Netflix) <-> DGS10 (10-Year Treasury Yield)
Partial correlation: -0.11, negative
Mechanism: Growth stocks are theoretically sensitive to discount rates. \
Higher yields reduce present value of future earnings.
Alternative explanations: This is a generic growth-stock-vs-rates argument \
that applies to ALL growth stocks equally. After partialling out market \
factors, it's unclear why Netflix specifically would retain this signal. \
The mechanism is real at the sector level but not specific to this pair.
Verdict: UNCERTAIN
Category: macro_sensitivity

Example 4 — LIKELY_SPURIOUS (confidence: 0.30)
Entities: COST (Costco) <-> SI=F (Silver Futures)
Partial correlation: 0.10, positive
Mechanism: No direct mechanism identified. Costco is a retailer; silver is \
a precious metal with industrial and store-of-value uses.
Alternative explanations: Both could respond to inflation expectations — \
Costco as a consumer spending proxy, silver as an inflation hedge. But \
this common factor should already be partialled out. Residual correlation \
of 0.10 at this sample size is within the noise floor.
Verdict: LIKELY_SPURIOUS
Category: none

Example 5 — CONFIRMED_SPURIOUS (confidence: 0.12)
Entities: MRNA (Moderna) <-> ZC=F (Corn Futures)
Partial correlation: 0.10, positive
Mechanism: No direct mechanism identified. Biotech vaccine manufacturer \
and agricultural commodity have no shared supply chain, regulatory \
exposure, or investor base.
Alternative explanations: Pure coincidence at this correlation magnitude. \
0.10 with 47 observations is well within sampling noise.
Verdict: CONFIRMED_SPURIOUS
Category: none

Example 6 — LIKELY_PLAUSIBLE (confidence: 0.68)
Entities: GDX (VanEck Gold Miners ETF) <-> DGS10 (10-Year Treasury Yield)
Partial correlation: -0.12, negative
Mechanism: Higher treasury yields increase the opportunity cost of holding \
non-yielding assets like gold. Lower gold prices compress gold miner revenues \
and margins, pushing GDX down. This is a well-documented macro-sensitivity \
relationship in financial literature.
Alternative explanations: Both could respond to inflation expectations or \
Fed policy, but these common factors should already be partialled out. The \
residual negative correlation is consistent with the direct yield-to-gold \
transmission channel.
Verdict: LIKELY_PLAUSIBLE
Category: macro_sensitivity

Example 7 — CONFIRMED_PLAUSIBLE (confidence: 0.90)
Entities: JETS (US Global Jets ETF) <-> CL=F (WTI Crude Oil Futures)
Partial correlation: -0.13, negative
Mechanism: Airlines are among the most oil-intensive businesses. Jet fuel, \
refined from crude oil, is 20-30% of airline operating costs. Higher crude \
prices directly compress airline margins and reduce profitability.
Alternative explanations: Both could respond to economic growth expectations, \
but in opposite directions — growth boosts travel demand but also raises oil \
prices. After partialling, the residual negative correlation reflects the \
direct cost-input relationship.
Verdict: CONFIRMED_PLAUSIBLE
Category: supply_chain"""

HUMAN_PROMPT = """\
Entity A: {entity_a_id} — {entity_a_desc} (type: {entity_a_type})
Entity B: {entity_b_id} — {entity_b_desc} (type: {entity_b_type})
Partial correlation: {correlation}, {direction}
Window: {window_days} trading days
Observations: {observation_count}

Evaluate this relationship. Start by identifying the mechanism (or lack of \
one), then consider alternative explanations, then render your verdict."""

VERIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_verification_chain():
    """Build a LangChain chain that verifies a single correlation.

    Returns a chain that accepts a dict of correlation properties
    and returns a VerificationResult.

    Usage:
        chain = build_verification_chain()
        result = chain.invoke({
            "entity_a_id": "NVDA",
            "entity_a_type": "asset",
            "entity_b_id": "AMD",
            "entity_b_type": "asset",
            "correlation": 0.12,
            "direction": "positive",
            "window_days": 252,
            "observation_count": 189,
        })
        print(result.verdict)     # "likely_plausible"
        print(result.mechanism)   # "Both compete in GPU/semiconductor market"
        print(result.confidence)  # 0.7
    """
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(VerificationResult)
    return VERIFICATION_PROMPT | structured_llm
