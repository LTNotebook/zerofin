"""Test the verification pipeline with 50 diverse correlations.

Run with: python scripts/test_verification.py

Tests a mix of:
- Obvious connections (should be plausible, high confidence)
- Obvious nonsense (should be implausible, high confidence)
- Gray zone pairs (could go either way — this is what we really want to evaluate)

Uses DeepSeek via OpenRouter (paid, but ~$0.10 for 50 calls).
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["LLM_PROVIDER"] = "openrouter"
os.environ["LLM_MODEL"] = "deepseek/deepseek-chat"

from zerofin.ai.verification import build_verification_chain  # noqa: E402

# fmt: off
TEST_CASES = [
    # =========================================================================
    # OBVIOUS — should be plausible, high confidence
    # =========================================================================
    {"label": "NVDA vs AMD (semiconductor competitors)", "entity_a_id": "NVDA", "entity_a_desc": "NVIDIA — GPUs, AI accelerators, data center chips", "entity_a_type": "asset", "entity_b_id": "AMD", "entity_b_desc": "Advanced Micro Devices — GPUs, CPUs, data center chips", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "XOM vs CL=F (oil company vs crude futures)", "entity_a_id": "XOM", "entity_a_desc": "ExxonMobil — integrated oil & gas major", "entity_a_type": "asset", "entity_b_id": "CL=F", "entity_b_desc": "WTI Crude Oil Futures", "entity_b_type": "asset", "correlation": 0.14, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "JPM vs GS (bank peers)", "entity_a_id": "JPM", "entity_a_desc": "JPMorgan Chase — largest US bank, investment banking", "entity_a_type": "asset", "entity_b_id": "GS", "entity_b_desc": "Goldman Sachs — investment bank, trading, asset management", "entity_b_type": "asset", "correlation": 0.13, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "GC=F vs GLD (gold futures vs gold ETF)", "entity_a_id": "GC=F", "entity_a_desc": "Gold Futures (COMEX)", "entity_a_type": "asset", "entity_b_id": "GLD", "entity_b_desc": "SPDR Gold Trust ETF — holds physical gold bullion", "entity_b_type": "asset", "correlation": 0.15, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "MSFT vs GOOGL (mega-cap tech peers)", "entity_a_id": "MSFT", "entity_a_desc": "Microsoft — cloud (Azure), software, AI", "entity_a_type": "asset", "entity_b_id": "GOOGL", "entity_b_desc": "Alphabet/Google — search, cloud, advertising, AI", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "CVX vs COP (energy peers)", "entity_a_id": "CVX", "entity_a_desc": "Chevron — integrated oil & gas major", "entity_a_type": "asset", "entity_b_id": "COP", "entity_b_desc": "ConocoPhillips — oil & gas exploration and production", "entity_b_type": "asset", "correlation": 0.13, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "LMT vs RTX (defense peers)", "entity_a_id": "LMT", "entity_a_desc": "Lockheed Martin — defense contractor, fighter jets, missiles", "entity_a_type": "asset", "entity_b_id": "RTX", "entity_b_desc": "RTX (Raytheon) — defense contractor, missiles, avionics", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "V vs MA (payment network duopoly)", "entity_a_id": "V", "entity_a_desc": "Visa — global payment network", "entity_a_type": "asset", "entity_b_id": "MA", "entity_b_desc": "Mastercard — global payment network", "entity_b_type": "asset", "correlation": 0.14, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "CRWD vs PANW (cybersecurity peers)", "entity_a_id": "CRWD", "entity_a_desc": "CrowdStrike — endpoint cybersecurity", "entity_a_type": "asset", "entity_b_id": "PANW", "entity_b_desc": "Palo Alto Networks — enterprise cybersecurity", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "WMT vs COST (retail peers)", "entity_a_id": "WMT", "entity_a_desc": "Walmart — largest US retailer, groceries, general merchandise", "entity_a_type": "asset", "entity_b_id": "COST", "entity_b_desc": "Costco — warehouse club retailer, bulk goods", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501

    # =========================================================================
    # OBVIOUS NONSENSE — should be implausible, high confidence
    # =========================================================================
    {"label": "NKE vs ZC=F (Nike vs corn futures)", "entity_a_id": "NKE", "entity_a_desc": "Nike — athletic footwear and apparel", "entity_a_type": "asset", "entity_b_id": "ZC=F", "entity_b_desc": "Corn Futures (CBOT)", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "NFLX vs ZW=F (Netflix vs wheat futures)", "entity_a_id": "NFLX", "entity_a_desc": "Netflix — streaming entertainment", "entity_a_type": "asset", "entity_b_id": "ZW=F", "entity_b_desc": "Wheat Futures (CBOT)", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "MRNA vs GC=F (Moderna vs gold)", "entity_a_id": "MRNA", "entity_a_desc": "Moderna — mRNA vaccines and therapeutics", "entity_a_type": "asset", "entity_b_id": "GC=F", "entity_b_desc": "Gold Futures (COMEX)", "entity_b_type": "asset", "correlation": 0.11, "direction": "negative", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "DIS vs NG=F (Disney vs natural gas)", "entity_a_id": "DIS", "entity_a_desc": "Walt Disney — entertainment, theme parks, streaming", "entity_a_type": "asset", "entity_b_id": "NG=F", "entity_b_desc": "Natural Gas Futures (Henry Hub)", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "PLTR vs ZS=F (Palantir vs soybean futures)", "entity_a_id": "PLTR", "entity_a_desc": "Palantir — data analytics software for government and enterprise", "entity_a_type": "asset", "entity_b_id": "ZS=F", "entity_b_desc": "Soybean Futures (CBOT)", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "LLY vs SI=F (Eli Lilly vs silver)", "entity_a_id": "LLY", "entity_a_desc": "Eli Lilly — pharmaceuticals, diabetes and obesity drugs", "entity_a_type": "asset", "entity_b_id": "SI=F", "entity_b_desc": "Silver Futures (COMEX)", "entity_b_type": "asset", "correlation": 0.10, "direction": "negative", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "MCD vs URA (McDonald's vs uranium ETF)", "entity_a_id": "MCD", "entity_a_desc": "McDonald's — fast food restaurant chain", "entity_a_type": "asset", "entity_b_id": "URA", "entity_b_desc": "Global X Uranium ETF — uranium miners and nuclear fuel", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "ARM vs DBA (ARM Holdings vs agriculture ETF)", "entity_a_id": "ARM", "entity_a_desc": "ARM Holdings — semiconductor IP licensing, chip design", "entity_a_type": "asset", "entity_b_id": "DBA", "entity_b_desc": "Invesco DB Agriculture Fund — agricultural commodity futures", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "COIN vs JNJ (Coinbase vs Johnson & Johnson)", "entity_a_id": "COIN", "entity_a_desc": "Coinbase — cryptocurrency exchange", "entity_a_type": "asset", "entity_b_id": "JNJ", "entity_b_desc": "Johnson & Johnson — diversified healthcare, pharmaceuticals", "entity_b_type": "asset", "correlation": 0.11, "direction": "negative", "window_days": 63, "observation_count": 47},  # noqa: E501
    {"label": "PFE vs LIT (Pfizer vs lithium ETF)", "entity_a_id": "PFE", "entity_a_desc": "Pfizer — pharmaceuticals, vaccines", "entity_a_type": "asset", "entity_b_id": "LIT", "entity_b_desc": "Global X Lithium & Battery Tech ETF", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 63, "observation_count": 47},  # noqa: E501

    # =========================================================================
    # GRAY ZONE — this is what matters. Could go either way.
    # =========================================================================
    {"label": "AAPL vs DGS10 (Apple vs 10yr treasury)", "entity_a_id": "AAPL", "entity_a_desc": "Apple — consumer electronics, software, services", "entity_a_type": "asset", "entity_b_id": "DGS10", "entity_b_desc": "10-Year US Treasury Yield", "entity_b_type": "indicator", "correlation": -0.11, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "TSLA vs LIT (Tesla vs lithium ETF)", "entity_a_id": "TSLA", "entity_a_desc": "Tesla — electric vehicles, batteries, energy storage", "entity_a_type": "asset", "entity_b_id": "LIT", "entity_b_desc": "Global X Lithium & Battery Tech ETF — lithium miners and battery makers", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "HD vs UNRATE (Home Depot vs unemployment)", "entity_a_id": "HD", "entity_a_desc": "Home Depot — home improvement retail", "entity_a_type": "asset", "entity_b_id": "UNRATE", "entity_b_desc": "US Unemployment Rate", "entity_b_type": "indicator", "correlation": -0.13, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "CAT vs HG=F (Caterpillar vs copper)", "entity_a_id": "CAT", "entity_a_desc": "Caterpillar — heavy equipment for construction, mining, agriculture", "entity_a_type": "asset", "entity_b_id": "HG=F", "entity_b_desc": "Copper Futures (COMEX) — industrial metal, global growth proxy", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "NEE vs NG=F (NextEra Energy vs natural gas)", "entity_a_id": "NEE", "entity_a_desc": "NextEra Energy — utility, largest US renewable energy producer", "entity_a_type": "asset", "entity_b_id": "NG=F", "entity_b_desc": "Natural Gas Futures (Henry Hub)", "entity_b_type": "asset", "correlation": -0.10, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "AMZN vs XLP (Amazon vs consumer staples)", "entity_a_id": "AMZN", "entity_a_desc": "Amazon — e-commerce, cloud (AWS), logistics", "entity_a_type": "asset", "entity_b_id": "XLP", "entity_b_desc": "Consumer Staples Select Sector SPDR ETF", "entity_b_type": "asset", "correlation": -0.11, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "CEG vs NVDA (Constellation Energy vs Nvidia)", "entity_a_id": "CEG", "entity_a_desc": "Constellation Energy — nuclear and natural gas power generation", "entity_a_type": "asset", "entity_b_id": "NVDA", "entity_b_desc": "NVIDIA — GPUs, AI accelerators, data center chips", "entity_b_type": "asset", "correlation": 0.13, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "GDX vs DGS10 (gold miners vs 10yr yield)", "entity_a_id": "GDX", "entity_a_desc": "VanEck Gold Miners ETF", "entity_a_type": "asset", "entity_b_id": "DGS10", "entity_b_desc": "10-Year US Treasury Yield", "entity_b_type": "indicator", "correlation": -0.12, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "KRE vs FEDFUNDS (regional banks vs fed funds rate)", "entity_a_id": "KRE", "entity_a_desc": "SPDR S&P Regional Banking ETF", "entity_a_type": "asset", "entity_b_id": "FEDFUNDS", "entity_b_desc": "Federal Funds Effective Rate", "entity_b_type": "indicator", "correlation": -0.14, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "JETS vs CL=F (airlines vs crude oil)", "entity_a_id": "JETS", "entity_a_desc": "US Global Jets ETF — airline stocks", "entity_a_type": "asset", "entity_b_id": "CL=F", "entity_b_desc": "WTI Crude Oil Futures", "entity_b_type": "asset", "correlation": -0.13, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "BABA vs FXI (Alibaba vs China large-cap ETF)", "entity_a_id": "BABA", "entity_a_desc": "Alibaba — China e-commerce, cloud computing", "entity_a_type": "asset", "entity_b_id": "FXI", "entity_b_desc": "iShares China Large-Cap ETF (FTSE China 50)", "entity_b_type": "asset", "correlation": 0.14, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "BDRY vs HG=F (dry bulk shipping vs copper)", "entity_a_id": "BDRY", "entity_a_desc": "Breakwave Dry Bulk Shipping ETF — global trade volume proxy", "entity_a_type": "asset", "entity_b_id": "HG=F", "entity_b_desc": "Copper Futures (COMEX) — industrial metal, global growth proxy", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "TLT vs ^VIX (long bonds vs volatility)", "entity_a_id": "TLT", "entity_a_desc": "iShares 20+ Year Treasury Bond ETF", "entity_a_type": "asset", "entity_b_id": "^VIX", "entity_b_desc": "CBOE Volatility Index — equity market fear gauge", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "HYG vs UNRATE (high yield bonds vs unemployment)", "entity_a_id": "HYG", "entity_a_desc": "iShares High Yield Corporate Bond ETF", "entity_a_type": "asset", "entity_b_id": "UNRATE", "entity_b_desc": "US Unemployment Rate", "entity_b_type": "indicator", "correlation": -0.12, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "XHB vs MORTGAGE30US (homebuilders vs 30yr mortgage)", "entity_a_id": "XHB", "entity_a_desc": "SPDR S&P Homebuilders ETF", "entity_a_type": "asset", "entity_b_id": "MORTGAGE30US", "entity_b_desc": "30-Year Fixed Rate Mortgage Average", "entity_b_type": "indicator", "correlation": -0.13, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "MPC vs BZ=F (Marathon Petroleum vs Brent crude)", "entity_a_id": "MPC", "entity_a_desc": "Marathon Petroleum — oil refining and marketing", "entity_a_type": "asset", "entity_b_id": "BZ=F", "entity_b_desc": "Brent Crude Oil Futures", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "DE vs ZC=F (Deere vs corn futures)", "entity_a_id": "DE", "entity_a_desc": "Deere & Company — world's largest agricultural equipment manufacturer", "entity_a_type": "asset", "entity_b_id": "ZC=F", "entity_b_desc": "Corn Futures (CBOT)", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "EQIX vs NVDA (Equinix data centers vs Nvidia)", "entity_a_id": "EQIX", "entity_a_desc": "Equinix — data center REIT, colocation and interconnection", "entity_a_type": "asset", "entity_b_id": "NVDA", "entity_b_desc": "NVIDIA — GPUs, AI accelerators, data center chips", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "VRT vs CEG (Vertiv vs Constellation — AI power)", "entity_a_id": "VRT", "entity_a_desc": "Vertiv Holdings — data center power management and cooling infrastructure", "entity_a_type": "asset", "entity_b_id": "CEG", "entity_b_desc": "Constellation Energy — largest US nuclear power generator, data center power supplier", "entity_b_type": "asset", "correlation": 0.13, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "EMB vs DXY (EM bonds vs dollar index)", "entity_a_id": "EMB", "entity_a_desc": "iShares JP Morgan USD Emerging Markets Bond ETF", "entity_a_type": "asset", "entity_b_id": "DTWEXBGS", "entity_b_desc": "Trade-Weighted US Dollar Index (Broad)", "entity_b_type": "indicator", "correlation": -0.14, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501

    # =========================================================================
    # TRICKY — plausible but indirect, or surprising
    # =========================================================================
    {"label": "COST vs UNRATE (Costco vs unemployment)", "entity_a_id": "COST", "entity_a_desc": "Costco — warehouse club retailer, bulk goods", "entity_a_type": "asset", "entity_b_id": "UNRATE", "entity_b_desc": "US Unemployment Rate", "entity_b_type": "indicator", "correlation": -0.10, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "PLD vs AMZN (Prologis warehouses vs Amazon)", "entity_a_id": "PLD", "entity_a_desc": "Prologis — industrial/logistics warehouse REIT", "entity_a_type": "asset", "entity_b_id": "AMZN", "entity_b_desc": "Amazon — e-commerce, cloud (AWS), logistics", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "SPG vs UNRATE (Simon Property malls vs unemployment)", "entity_a_id": "SPG", "entity_a_desc": "Simon Property Group — largest US mall/retail REIT", "entity_a_type": "asset", "entity_b_id": "UNRATE", "entity_b_desc": "US Unemployment Rate", "entity_b_type": "indicator", "correlation": -0.12, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "COIN vs BTC-USD (Coinbase vs Bitcoin)", "entity_a_id": "COIN", "entity_a_desc": "Coinbase — cryptocurrency exchange", "entity_a_type": "asset", "entity_b_id": "BTC-USD", "entity_b_desc": "Bitcoin price in USD", "entity_b_type": "asset", "correlation": 0.14, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "UNP vs DBC (Union Pacific vs commodities)", "entity_a_id": "UNP", "entity_a_desc": "Union Pacific — railroad, transports coal, grain, chemicals, intermodal freight", "entity_a_type": "asset", "entity_b_id": "DBC", "entity_b_desc": "Invesco DB Commodity Index — broad commodity futures basket", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "WELL vs AGING (Welltower vs aging demographics)", "entity_a_id": "WELL", "entity_a_desc": "Welltower — healthcare REIT, senior housing and medical facilities", "entity_a_type": "asset", "entity_b_id": "POPTHM", "entity_b_desc": "US Population Growth Rate (monthly)", "entity_b_type": "indicator", "correlation": 0.10, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "TTD vs META (Trade Desk vs Meta — ad tech)", "entity_a_id": "TTD", "entity_a_desc": "The Trade Desk — programmatic advertising platform (demand-side)", "entity_a_type": "asset", "entity_b_id": "META", "entity_b_desc": "Meta Platforms — social media, digital advertising (Facebook, Instagram)", "entity_b_type": "asset", "correlation": 0.11, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "KMI vs NG=F (Kinder Morgan vs natural gas)", "entity_a_id": "KMI", "entity_a_desc": "Kinder Morgan — natural gas pipeline and storage infrastructure", "entity_a_type": "asset", "entity_b_id": "NG=F", "entity_b_desc": "Natural Gas Futures (Henry Hub)", "entity_b_type": "asset", "correlation": 0.12, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "O vs DGS10 (Realty Income vs 10yr yield)", "entity_a_id": "O", "entity_a_desc": "Realty Income — net lease REIT, bond-like monthly dividends", "entity_a_type": "asset", "entity_b_id": "DGS10", "entity_b_desc": "10-Year US Treasury Yield", "entity_b_type": "indicator", "correlation": -0.13, "direction": "negative", "window_days": 252, "observation_count": 189},  # noqa: E501
    {"label": "SNOW vs NOW (Snowflake vs ServiceNow — cloud peers?)", "entity_a_id": "SNOW", "entity_a_desc": "Snowflake — cloud data warehousing and analytics", "entity_a_type": "asset", "entity_b_id": "NOW", "entity_b_desc": "ServiceNow — enterprise IT workflow automation (SaaS)", "entity_b_type": "asset", "correlation": 0.10, "direction": "positive", "window_days": 252, "observation_count": 189},  # noqa: E501
]
# fmt: on


def main() -> None:
    chain = build_verification_chain()

    # Separate labels from inputs
    labels = [case.pop("label") for case in TEST_CASES]
    inputs = TEST_CASES

    print(f"Sending {len(inputs)} cases in parallel (max_concurrency=20)...\n")
    start = time.time()

    # Batch sends all requests with up to 20 running at once
    results = chain.batch(inputs, config={"max_concurrency": 20})

    elapsed = time.time() - start

    # Track verdict distribution
    verdict_counts: dict[str, int] = {}
    total_input_chars = 0
    total_output_chars = 0

    for label, result in zip(labels, results):
        print(f"{label}")
        print(f"  Verdict: {result.verdict.upper()} (confidence: {result.confidence})")
        print(f"  Mechanism: {result.mechanism}")
        print(f"  Alternatives: {result.alternative_explanations}")
        print(f"  Category: {result.relationship_category}")
        print(f"  Reasoning: {result.reasoning}")
        print()

        verdict_counts[result.verdict] = verdict_counts.get(result.verdict, 0) + 1

        # Rough token estimate (1 token ~ 4 chars)
        total_output_chars += len(result.mechanism) + len(result.alternative_explanations)
        total_output_chars += len(result.reasoning) + len(result.verdict)

    # The system prompt is the same for all calls (~2500 chars)
    system_prompt_chars = 2500
    human_prompt_chars_each = 200
    total_input_chars = (system_prompt_chars + human_prompt_chars_each) * len(inputs)

    est_input_tokens = total_input_chars // 4
    est_output_tokens = total_output_chars // 4
    est_total_tokens = est_input_tokens + est_output_tokens

    # DeepSeek via OpenRouter pricing (approx)
    input_cost = (est_input_tokens / 1_000_000) * 0.30
    output_cost = (est_output_tokens / 1_000_000) * 0.44
    total_cost = (input_cost + output_cost) * 1.055  # 5.5% OpenRouter fee

    print("=" * 60)
    print(f"RESULTS ({len(inputs)} pairs in {elapsed:.1f}s)")
    print("-" * 60)
    print("Verdict distribution:")
    for verdict in ["confirmed_plausible", "likely_plausible", "uncertain",
                     "likely_spurious", "confirmed_spurious"]:
        count = verdict_counts.get(verdict, 0)
        pct = count / len(inputs) * 100
        bar = "#" * count
        print(f"  {verdict:25s} {count:3d} ({pct:4.1f}%) {bar}")
    print("-" * 60)
    print(
        f"Estimated tokens:  ~{est_input_tokens:,} input"
        f" + ~{est_output_tokens:,} output = ~{est_total_tokens:,} total"
    )
    print(f"Estimated cost:    ~${total_cost:.4f}")


if __name__ == "__main__":
    main()
