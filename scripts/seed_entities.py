"""Seed Neo4j with all tracked entities and basic structural relationships.

Run once at project setup, safe to re-run (uses MERGE so no duplicates).

Usage:
    uv run python scripts/seed_entities.py
"""

from __future__ import annotations

import logging
import sys

from zerofin.data.tickers import FRED_INDICATORS
from zerofin.models.entities import EntityCreate
from zerofin.storage.graph import GraphStorage

# ── Logging setup ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Entity Seed Data
# =============================================================================
# Every ticker from tickers.py mapped to its Neo4j entity info.
# Descriptions come from the tracking list (12 - Tracking List.md).

ENTITY_SEED_DATA: list[dict] = [
    # ── US Indices ──────────────────────────────────────────────────────
    {
        "id": "^GSPC",
        "label": "Index",
        "name": "S&P 500",
        "description": (
            "THE benchmark for US large-cap equities; 500 companies, ~80% of US market cap"
        ),
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^DJI",
        "label": "Index",
        "name": "Dow Jones Industrial Average",
        "description": "30 blue-chip stocks; oldest and most widely quoted index",
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^IXIC",
        "label": "Index",
        "name": "Nasdaq Composite",
        "description": "~3,000 stocks, heavy tech/growth weighting; barometer for tech sentiment",
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^NDX",
        "label": "Index",
        "name": "Nasdaq 100",
        "description": "Top 100 non-financial Nasdaq stocks; pure large-cap tech/growth proxy",
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^RUT",
        "label": "Index",
        "name": "Russell 2000",
        "description": "Small-cap benchmark; sensitive to domestic economic conditions and credit",
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^VIX",
        "label": "Index",
        "name": "CBOE Volatility Index",
        "description": "Fear gauge — implied volatility of S&P 500 options over next 30 days",
        "metadata": {"category": "us_index", "subtype": "volatility"},
    },
    {
        "id": "^GSPTSE",
        "label": "Index",
        "name": "S&P/TSX Composite",
        "description": "Largest US trading partner; commodity-heavy Canadian index",
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^W5000",
        "label": "Index",
        "name": "Wilshire 5000",
        "description": "Broadest measure of total US stock market",
        "metadata": {"category": "us_index"},
    },
    {
        "id": "^VVIX",
        "label": "Index",
        "name": "CBOE VIX of VIX",
        "description": (
            "Volatility of volatility — leads VIX, measures uncertainty"
        ),
        "metadata": {"category": "us_index", "subtype": "volatility"},
    },
    {
        "id": "^MOVE",
        "label": "Index",
        "name": "ICE BofAML MOVE Index",
        "description": "Bond market volatility — VIX for Treasuries, fixed income stress signal",
        "metadata": {"category": "us_index", "subtype": "volatility"},
    },
    {
        "id": "^GVZ",
        "label": "Index",
        "name": "CBOE Gold Volatility Index",
        "description": "Gold options implied volatility — macro panic signal",
        "metadata": {"category": "us_index", "subtype": "volatility"},
    },
    {
        "id": "^OVX",
        "label": "Index",
        "name": "CBOE Crude Oil Volatility Index",
        "description": "Oil implied volatility — energy sector stress signal",
        "metadata": {"category": "us_index", "subtype": "volatility"},
    },
    # ── Sector ETFs (11 GICS) ──────────────────────────────────────────
    {
        "id": "XLK",
        "label": "Asset",
        "name": "Technology Select Sector SPDR",
        "description": "Largest sector weight in S&P 500; AAPL, MSFT, NVDA",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Technology"},
    },
    {
        "id": "XLF",
        "label": "Asset",
        "name": "Financial Select Sector SPDR",
        "description": "Banks, insurers, asset managers; rate-sensitive",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Financials"},
    },
    {
        "id": "XLV",
        "label": "Asset",
        "name": "Health Care Select Sector SPDR",
        "description": "Defensive growth; pharma, biotech, insurance",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Healthcare"},
    },
    {
        "id": "XLE",
        "label": "Asset",
        "name": "Energy Select Sector SPDR",
        "description": "Oil & gas producers/servicers; commodity-driven",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Energy"},
    },
    {
        "id": "XLI",
        "label": "Asset",
        "name": "Industrial Select Sector SPDR",
        "description": "Capex bellwether; aerospace, machinery, transports",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Industrials"},
    },
    {
        "id": "XLY",
        "label": "Asset",
        "name": "Consumer Discretionary Select Sector SPDR",
        "description": "Cyclical spending; AMZN, TSLA, HD",
        "metadata": {
            "subtype": "etf",
            "asset_type": "sector_etf",
            "sector": "Consumer Discretionary",
        },
    },
    {
        "id": "XLP",
        "label": "Asset",
        "name": "Consumer Staples Select Sector SPDR",
        "description": "Defensive; food, beverage, household goods",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Consumer Staples"},
    },
    {
        "id": "XLU",
        "label": "Asset",
        "name": "Utilities Select Sector SPDR",
        "description": "Rate-sensitive defensive; dividend proxy",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Utilities"},
    },
    {
        "id": "XLB",
        "label": "Asset",
        "name": "Materials Select Sector SPDR",
        "description": "Chemicals, metals, mining; inflation/commodity play",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Materials"},
    },
    {
        "id": "XLRE",
        "label": "Asset",
        "name": "Real Estate Select Sector SPDR",
        "description": "REITs; interest rate and credit sensitive",
        "metadata": {"subtype": "etf", "asset_type": "sector_etf", "sector": "Real Estate"},
    },
    {
        "id": "XLC",
        "label": "Asset",
        "name": "Communication Services Select Sector SPDR",
        "description": "META, GOOGL, NFLX; ad spend and streaming",
        "metadata": {
            "subtype": "etf",
            "asset_type": "sector_etf",
            "sector": "Communication Services",
        },
    },
    # ── Thematic / Sub-Sector ETFs ─────────────────────────────────────
    {
        "id": "SMH",
        "label": "Asset",
        "name": "VanEck Semiconductor ETF",
        "description": "AI/chip cycle proxy; NVDA, AVGO, TSM, ASML",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "semiconductors"},
    },
    {
        "id": "AIQ",
        "label": "Asset",
        "name": "Global X Artificial Intelligence & Technology ETF",
        "description": "Largest pure-play AI ETF; 92 holdings across AI software, chips, cloud",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "ai"},
    },
    {
        "id": "BOTZ",
        "label": "Asset",
        "name": "Global X Robotics & AI ETF",
        "description": "Industrial robotics + AI; ABB, Fanuc, NVDA, Keyence",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "robotics"},
    },
    {
        "id": "ROBO",
        "label": "Asset",
        "name": "ROBO Global Robotics & Automation ETF",
        "description": "Equal-weighted robotics/automation; small/mid-cap AI exposure",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "robotics"},
    },
    {
        "id": "ITA",
        "label": "Asset",
        "name": "iShares US Aerospace & Defense ETF",
        "description": "Defense contractors; government-mandated multi-year spending",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "defense"},
    },
    {
        "id": "HACK",
        "label": "Asset",
        "name": "ETFMG Prime Cyber Security ETF",
        "description": "Cybersecurity stocks; AI agent attack surface expansion",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "cybersecurity"},
    },
    {
        "id": "IBB",
        "label": "Asset",
        "name": "iShares Biotechnology ETF",
        "description": "High-beta healthcare; drug pipeline sentiment",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "biotech"},
    },
    {
        "id": "XHB",
        "label": "Asset",
        "name": "SPDR S&P Homebuilders ETF",
        "description": "Housing cycle bellwether",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "housing"},
    },
    {
        "id": "KRE",
        "label": "Asset",
        "name": "SPDR S&P Regional Banking ETF",
        "description": "Credit stress indicator; rate spread sensitive",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "regional_banks"},
    },
    {
        "id": "RSP",
        "label": "Asset",
        "name": "Invesco S&P 500 Equal Weight ETF",
        "description": "Equal-weighted S&P; rotation and breadth indicator",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "equal_weight"},
    },
    # ── Factor ETFs ───────────────────────────────────────────────────
    {
        "id": "MTUM",
        "label": "Asset",
        "name": "iShares MSCI USA Momentum Factor ETF",
        "description": (
            "Momentum factor — tracks 6/12-month momentum, regime signal"
        ),
        "metadata": {"subtype": "etf", "asset_type": "factor_etf", "factor": "momentum"},
    },
    {
        "id": "QUAL",
        "label": "Asset",
        "name": "iShares MSCI USA Quality Factor ETF",
        "description": "Quality factor — high ROE, low debt; late-cycle signal",
        "metadata": {"subtype": "etf", "asset_type": "factor_etf", "factor": "quality"},
    },
    {
        "id": "USMV",
        "label": "Asset",
        "name": "iShares MSCI USA Min Vol Factor ETF",
        "description": "Min volatility factor — risk-off equity positioning signal",
        "metadata": {"subtype": "etf", "asset_type": "factor_etf", "factor": "min_vol"},
    },
    # ── Alternative / Thematic Proxy ETFs ─────────────────────────────
    {
        "id": "JETS",
        "label": "Asset",
        "name": "U.S. Global Jets ETF",
        "description": "Airline demand — consumer spending and travel activity proxy",
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "airlines"},
    },
    {
        "id": "BDRY",
        "label": "Asset",
        "name": "Breakwave Dry Bulk Shipping ETF",
        "description": (
            "Dry bulk freight futures — global trade volume proxy"
        ),
        "metadata": {"subtype": "etf", "asset_type": "thematic_etf", "theme": "shipping"},
    },
    # ── Key Stocks — Mega-Cap Tech ─────────────────────────────────────
    {
        "id": "AAPL",
        "label": "Company",
        "name": "Apple Inc.",
        "description": "Largest company by market cap; consumer tech bellwether",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "MSFT",
        "label": "Company",
        "name": "Microsoft Corporation",
        "description": "Enterprise software, cloud (Azure), AI (OpenAI partner)",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "NVDA",
        "label": "Company",
        "name": "NVIDIA Corporation",
        "description": "AI infrastructure leader; GPU monopoly for AI training",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "GOOGL",
        "label": "Company",
        "name": "Alphabet Inc.",
        "description": "Search, cloud, advertising, AI (Gemini/DeepMind)",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "AMZN",
        "label": "Company",
        "name": "Amazon.com Inc.",
        "description": "E-commerce, AWS cloud, logistics; consumer spending proxy",
        "metadata": {"sector": "Consumer Discretionary"},
    },
    {
        "id": "META",
        "label": "Company",
        "name": "Meta Platforms Inc.",
        "description": "Social media, digital ads, AI, metaverse",
        "metadata": {"sector": "Communication Services"},
    },
    {
        "id": "TSLA",
        "label": "Company",
        "name": "Tesla Inc.",
        "description": "EV leader; high retail ownership, massive vol, Musk factor",
        "metadata": {"sector": "Consumer Discretionary"},
    },
    # ── Key Stocks — Semiconductors ────────────────────────────────────
    {
        "id": "AVGO",
        "label": "Company",
        "name": "Broadcom Inc.",
        "description": "AI networking, custom chips; diversified semi",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "AMD",
        "label": "Company",
        "name": "Advanced Micro Devices Inc.",
        "description": "CPU/GPU competitor to Intel and NVIDIA",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "TSM",
        "label": "Company",
        "name": "Taiwan Semiconductor Manufacturing",
        "description": "Manufactures chips for AAPL, NVDA, AMD — global chokepoint",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "ASML",
        "label": "Company",
        "name": "ASML Holding NV",
        "description": "Only maker of EUV lithography machines; semi capex proxy",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "MU",
        "label": "Company",
        "name": "Micron Technology Inc.",
        "description": "Memory/DRAM leader; AI memory demand",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "QCOM",
        "label": "Company",
        "name": "Qualcomm Inc.",
        "description": "Mobile chips, AI edge, licensing",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "INTC",
        "label": "Company",
        "name": "Intel Corporation",
        "description": "Legacy chipmaker; US onshoring / CHIPS Act beneficiary",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "ARM",
        "label": "Company",
        "name": "ARM Holdings plc",
        "description": "CPU architecture licensor; ubiquitous in mobile and AI edge",
        "metadata": {"sector": "Technology"},
    },
    # ── Key Stocks — Banks & Finance ───────────────────────────────────
    {
        "id": "JPM",
        "label": "Company",
        "name": "JPMorgan Chase & Co.",
        "description": "Largest US bank; credit cycle bellwether",
        "metadata": {"sector": "Financials"},
    },
    {
        "id": "GS",
        "label": "Company",
        "name": "Goldman Sachs Group Inc.",
        "description": "Investment banking, trading; capital markets proxy",
        "metadata": {"sector": "Financials"},
    },
    {
        "id": "BAC",
        "label": "Company",
        "name": "Bank of America Corporation",
        "description": "Consumer banking giant; rate sensitivity",
        "metadata": {"sector": "Financials"},
    },
    {
        "id": "BRK-B",
        "label": "Company",
        "name": "Berkshire Hathaway Inc.",
        "description": "Buffett's conglomerate; market sentiment and insurance",
        "metadata": {"sector": "Financials"},
    },
    {
        "id": "V",
        "label": "Company",
        "name": "Visa Inc.",
        "description": "Global payments network; consumer spending proxy",
        "metadata": {"sector": "Financials"},
    },
    {
        "id": "MA",
        "label": "Company",
        "name": "Mastercard Inc.",
        "description": "Payments network; international spending gauge",
        "metadata": {"sector": "Financials"},
    },
    # ── Key Stocks — Energy ────────────────────────────────────────────
    {
        "id": "XOM",
        "label": "Company",
        "name": "Exxon Mobil Corporation",
        "description": "Largest US oil major; integrated energy bellwether",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "CVX",
        "label": "Company",
        "name": "Chevron Corporation",
        "description": "Second-largest US oil major",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "COP",
        "label": "Company",
        "name": "ConocoPhillips",
        "description": "Largest independent E&P; pure upstream proxy",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "SLB",
        "label": "Company",
        "name": "SLB (Schlumberger)",
        "description": "Oilfield services leader; capex cycle indicator",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "OXY",
        "label": "Company",
        "name": "Occidental Petroleum",
        "description": "E&P + carbon capture; major Berkshire Hathaway holding",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "MPC",
        "label": "Company",
        "name": "Marathon Petroleum",
        "description": "Refining margins signal; distinct from upstream E&P",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "PSX",
        "label": "Company",
        "name": "Phillips 66",
        "description": "Midstream + refining; different signal from upstream producers",
        "metadata": {"sector": "Energy"},
    },
    {
        "id": "KMI",
        "label": "Company",
        "name": "Kinder Morgan",
        "description": "Natural gas pipeline infrastructure; midstream signal",
        "metadata": {"sector": "Energy"},
    },
    # ── Key Stocks — Defense & Aerospace ───────────────────────────────
    {
        "id": "LMT",
        "label": "Company",
        "name": "Lockheed Martin Corporation",
        "description": "Largest defense contractor; F-35, missile defense",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "RTX",
        "label": "Company",
        "name": "RTX Corporation",
        "description": "Defense + commercial aero (Pratt & Whitney, Raytheon)",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "NOC",
        "label": "Company",
        "name": "Northrop Grumman Corporation",
        "description": "Stealth bombers, space systems, nuclear deterrence",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "GD",
        "label": "Company",
        "name": "General Dynamics Corporation",
        "description": "Submarines, Gulfstream jets, IT services",
        "metadata": {"sector": "Industrials"},
    },
    # ── Key Stocks — Healthcare ────────────────────────────────────────
    {
        "id": "UNH",
        "label": "Company",
        "name": "UnitedHealth Group Inc.",
        "description": "Largest health insurer; healthcare cost bellwether",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "LLY",
        "label": "Company",
        "name": "Eli Lilly and Company",
        "description": "GLP-1 obesity/diabetes drugs (Mounjaro/Zepbound); top pharma",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "JNJ",
        "label": "Company",
        "name": "Johnson & Johnson",
        "description": "Diversified healthcare; defensive stalwart",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "PFE",
        "label": "Company",
        "name": "Pfizer Inc.",
        "description": "Major pharma; vaccine franchise, pipeline",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "ABBV",
        "label": "Company",
        "name": "AbbVie Inc.",
        "description": "Immunology (Humira successor), oncology",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "MRNA",
        "label": "Company",
        "name": "Moderna Inc.",
        "description": "mRNA platform; vaccines, therapeutics pipeline",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "AMGN",
        "label": "Company",
        "name": "Amgen Inc.",
        "description": "Large-cap biotech; Dow 30 component, obesity pipeline",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "MRK",
        "label": "Company",
        "name": "Merck & Co.",
        "description": "Oncology leader (Keytruda); top-selling drug globally",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "GILD",
        "label": "Company",
        "name": "Gilead Sciences",
        "description": "Antivirals / HIV franchise; virology signal",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "REGN",
        "label": "Company",
        "name": "Regeneron Pharmaceuticals",
        "description": "Immunology (Dupixent) + ophthalmology (Eylea)",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "VRTX",
        "label": "Company",
        "name": "Vertex Pharmaceuticals",
        "description": "Cystic fibrosis monopoly + gene editing pipeline",
        "metadata": {"sector": "Healthcare"},
    },
    {
        "id": "BIIB",
        "label": "Company",
        "name": "Biogen Inc.",
        "description": "Neurology / Alzheimer's; high idiosyncratic risk",
        "metadata": {"sector": "Healthcare"},
    },
    # ── Key Stocks — Retail & Consumer ─────────────────────────────────
    {
        "id": "WMT",
        "label": "Company",
        "name": "Walmart Inc.",
        "description": "Largest retailer; low-end consumer health proxy",
        "metadata": {"sector": "Consumer Staples"},
    },
    {
        "id": "COST",
        "label": "Company",
        "name": "Costco Wholesale Corporation",
        "description": "Membership warehouse; middle/upper consumer proxy",
        "metadata": {"sector": "Consumer Staples"},
    },
    {
        "id": "HD",
        "label": "Company",
        "name": "The Home Depot Inc.",
        "description": "Housing and renovation cycle bellwether",
        "metadata": {"sector": "Consumer Discretionary"},
    },
    {
        "id": "NKE",
        "label": "Company",
        "name": "Nike Inc.",
        "description": "Global consumer brand; international demand proxy",
        "metadata": {"sector": "Consumer Discretionary"},
    },
    {
        "id": "MCD",
        "label": "Company",
        "name": "McDonald's Corporation",
        "description": "Consumer discretionary/defensive hybrid; Dow 30, fast food bellwether",
        "metadata": {"sector": "Consumer Discretionary"},
    },
    {
        "id": "PG",
        "label": "Company",
        "name": "Procter & Gamble",
        "description": "Defensive consumer staples bellwether; Dow 30",
        "metadata": {"sector": "Consumer Staples"},
    },
    {
        "id": "KO",
        "label": "Company",
        "name": "Coca-Cola Company",
        "description": "Classic defensive; Buffett holding, consumer staples anchor",
        "metadata": {"sector": "Consumer Staples"},
    },
    # ── Key Stocks — Industrials ───────────────────────────────────────
    {
        "id": "CAT",
        "label": "Company",
        "name": "Caterpillar Inc.",
        "description": "Heavy equipment; global infrastructure/mining cycle",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "GE",
        "label": "Company",
        "name": "GE Aerospace",
        "description": "Aviation engines and services",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "HON",
        "label": "Company",
        "name": "Honeywell International Inc.",
        "description": "Diversified industrial; automation, building tech",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "DE",
        "label": "Company",
        "name": "Deere & Company",
        "description": "Agriculture equipment; farm economy proxy",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "UNP",
        "label": "Company",
        "name": "Union Pacific Corporation",
        "description": "Largest US railroad; freight and economic activity gauge",
        "metadata": {"sector": "Industrials"},
    },
    # ── Key Stocks — AI / Cloud / Software ─────────────────────────────
    {
        "id": "CRM",
        "label": "Company",
        "name": "Salesforce Inc.",
        "description": "Enterprise SaaS leader; AI agent platform",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "PLTR",
        "label": "Company",
        "name": "Palantir Technologies Inc.",
        "description": "AI/data analytics for government and enterprise",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "NOW",
        "label": "Company",
        "name": "ServiceNow Inc.",
        "description": "IT workflow automation; enterprise AI adoption",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "SNOW",
        "label": "Company",
        "name": "Snowflake Inc.",
        "description": "Cloud data platform; data/AI infrastructure",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "ORCL",
        "label": "Company",
        "name": "Oracle Corporation",
        "description": "Cloud infrastructure, databases; AI cloud buildout",
        "metadata": {"sector": "Technology"},
    },
    # ── Key Stocks — AI Infrastructure ─────────────────────────────────
    {
        "id": "CEG",
        "label": "Company",
        "name": "Constellation Energy Corporation",
        "description": "Largest US nuclear operator; AI data center power contracts",
        "metadata": {"sector": "Utilities"},
    },
    {
        "id": "VRT",
        "label": "Company",
        "name": "Vertiv Holdings Co.",
        "description": "Data center cooling and power management; 76% of AI servers liquid-cooled",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "ETN",
        "label": "Company",
        "name": "Eaton Corporation plc",
        "description": "Power infrastructure and grid equipment for data centers",
        "metadata": {"sector": "Industrials"},
    },
    {
        "id": "ANET",
        "label": "Company",
        "name": "Arista Networks Inc.",
        "description": "AI data center networking switches; high-speed cluster interconnect",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "MRVL",
        "label": "Company",
        "name": "Marvell Technology Inc.",
        "description": "Custom AI chips + networking silicon; hyperscaler design partner",
        "metadata": {"sector": "Technology"},
    },
    # ── Key Stocks — Cybersecurity ─────────────────────────────────────
    {
        "id": "PANW",
        "label": "Company",
        "name": "Palo Alto Networks Inc.",
        "description": "Largest pure-play cybersecurity; AI security spending bellwether",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "CRWD",
        "label": "Company",
        "name": "CrowdStrike Holdings Inc.",
        "description": "Endpoint security leader; enterprise security spending gauge",
        "metadata": {"sector": "Technology"},
    },
    # ── Key Stocks — Media & Telecom ───────────────────────────────────
    {
        "id": "NFLX",
        "label": "Company",
        "name": "Netflix Inc.",
        "description": "Streaming leader; consumer discretionary spending, ad market",
        "metadata": {"sector": "Communication Services"},
    },
    {
        "id": "DIS",
        "label": "Company",
        "name": "The Walt Disney Company",
        "description": "Media/entertainment conglomerate; parks, streaming, content",
        "metadata": {"sector": "Communication Services"},
    },
    {
        "id": "VZ",
        "label": "Company",
        "name": "Verizon Communications Inc.",
        "description": "Largest US telecom; dividend, debt/refinancing risk signal",
        "metadata": {"sector": "Communication Services"},
    },
    # ── Key Stocks — Signal Stocks ─────────────────────────────────────
    {
        "id": "KKR",
        "label": "Company",
        "name": "KKR & Co. Inc.",
        "description": "Alternative asset manager; private credit cycle, insider buying signal",
        "metadata": {"sector": "Financials"},
    },
    {
        "id": "TTD",
        "label": "Company",
        "name": "The Trade Desk Inc.",
        "description": "Programmatic ads; CEO $148M insider buy, open-web AI advertising",
        "metadata": {"sector": "Technology"},
    },
    {
        "id": "COIN",
        "label": "Company",
        "name": "Coinbase Global Inc.",
        "description": "Crypto exchange; bridges crypto to traditional equities",
        "metadata": {"sector": "Financials"},
    },
    # ── REITs ─────────────────────────────────────────────────────────
    {
        "id": "PLD",
        "label": "Company",
        "name": "Prologis Inc.",
        "description": "Industrial/logistics REIT; e-commerce and warehouse demand proxy",
        "metadata": {"sector": "Real Estate"},
    },
    {
        "id": "AMT",
        "label": "Company",
        "name": "American Tower Corporation",
        "description": "Cell tower REIT; interest rate sensitive + wireless infrastructure",
        "metadata": {"sector": "Real Estate"},
    },
    {
        "id": "EQIX",
        "label": "Company",
        "name": "Equinix Inc.",
        "description": "Data center REIT; AI infrastructure demand signal",
        "metadata": {"sector": "Real Estate"},
    },
    {
        "id": "DLR",
        "label": "Company",
        "name": "Digital Realty Trust",
        "description": "Data center REIT; AI buildout signal",
        "metadata": {"sector": "Real Estate"},
    },
    {
        "id": "WELL",
        "label": "Company",
        "name": "Welltower Inc.",
        "description": "Healthcare REIT; aging demographics play",
        "metadata": {"sector": "Real Estate"},
    },
    {
        "id": "SPG",
        "label": "Company",
        "name": "Simon Property Group",
        "description": "Retail/mall REIT; consumer spending signal",
        "metadata": {"sector": "Real Estate"},
    },
    {
        "id": "O",
        "label": "Company",
        "name": "Realty Income Corporation",
        "description": "Net lease REIT; bond-like defensive income signal",
        "metadata": {"sector": "Real Estate"},
    },
    # ── Utilities ─────────────────────────────────────────────────────
    {
        "id": "NEE",
        "label": "Company",
        "name": "NextEra Energy",
        "description": "Largest utility; renewables + AI datacenter power contracts",
        "metadata": {"sector": "Utilities"},
    },
    {
        "id": "VST",
        "label": "Company",
        "name": "Vistra Corp.",
        "description": "Deregulated power generator; direct AI energy demand play",
        "metadata": {"sector": "Utilities"},
    },
    {
        "id": "SO",
        "label": "Company",
        "name": "Southern Company",
        "description": "Traditional regulated utility; pure rate-sensitivity signal",
        "metadata": {"sector": "Utilities"},
    },
    {
        "id": "DUK",
        "label": "Company",
        "name": "Duke Energy",
        "description": "Traditional utility with AI power transition angle",
        "metadata": {"sector": "Utilities"},
    },
    # ── China Individual Stocks ───────────────────────────────────────
    {
        "id": "BABA",
        "label": "Company",
        "name": "Alibaba Group",
        "description": "China e-commerce bellwether; ADR on NYSE",
        "metadata": {"region": "China"},
    },
    {
        "id": "PDD",
        "label": "Company",
        "name": "PDD Holdings (Temu)",
        "description": "China e-commerce + international expansion via Temu",
        "metadata": {"region": "China"},
    },
    # ── Commodity Futures ──────────────────────────────────────────────
    {
        "id": "CL=F",
        "label": "Commodity",
        "name": "WTI Crude Oil Futures",
        "description": "Global energy benchmark; inflation input, geopolitics",
        "metadata": {"commodity_type": "energy"},
    },
    {
        "id": "BZ=F",
        "label": "Commodity",
        "name": "Brent Crude Oil Futures",
        "description": "International oil benchmark; premium to WTI signals supply",
        "metadata": {"commodity_type": "energy"},
    },
    {
        "id": "GC=F",
        "label": "Commodity",
        "name": "Gold Futures",
        "description": "Safe haven, inflation hedge, central bank buying",
        "metadata": {"commodity_type": "precious_metal"},
    },
    {
        "id": "SI=F",
        "label": "Commodity",
        "name": "Silver Futures",
        "description": "Industrial + precious metal; more volatile than gold",
        "metadata": {"commodity_type": "precious_metal"},
    },
    {
        "id": "HG=F",
        "label": "Commodity",
        "name": "Copper Futures",
        "description": "Dr. Copper — industrial demand and global growth proxy",
        "metadata": {"commodity_type": "industrial_metal"},
    },
    {
        "id": "NG=F",
        "label": "Commodity",
        "name": "Natural Gas Futures",
        "description": "US energy costs, heating/cooling, LNG exports",
        "metadata": {"commodity_type": "energy"},
    },
    {
        "id": "ZC=F",
        "label": "Commodity",
        "name": "Corn Futures",
        "description": "Key food/feed/ethanol crop; ag inflation indicator",
        "metadata": {"commodity_type": "agriculture"},
    },
    {
        "id": "ZW=F",
        "label": "Commodity",
        "name": "Wheat Futures",
        "description": "Global food security; geopolitical flashpoint (Black Sea)",
        "metadata": {"commodity_type": "agriculture"},
    },
    {
        "id": "ZS=F",
        "label": "Commodity",
        "name": "Soybean Futures",
        "description": "Major US export; China demand proxy",
        "metadata": {"commodity_type": "agriculture"},
    },
    # ── Commodity ETFs ─────────────────────────────────────────────────
    {
        "id": "GLD",
        "label": "Asset",
        "name": "SPDR Gold Trust ETF",
        "description": "Gold ETF; safe haven, inflation hedge, central bank demand",
        "metadata": {"subtype": "commodity_etf", "commodity": "gold"},
    },
    {
        "id": "URA",
        "label": "Asset",
        "name": "Global X Uranium ETF",
        "description": "Nuclear energy demand; no liquid uranium futures on yfinance",
        "metadata": {"subtype": "commodity_etf", "commodity": "uranium"},
    },
    {
        "id": "DBA",
        "label": "Asset",
        "name": "Invesco DB Agriculture Fund",
        "description": "Broad agriculture basket ETF",
        "metadata": {"subtype": "commodity_etf", "commodity": "agriculture"},
    },
    {
        "id": "DBC",
        "label": "Asset",
        "name": "Invesco DB Commodity Index Tracking Fund",
        "description": "Broad commodity basket; overall commodity cycle",
        "metadata": {"subtype": "commodity_etf", "commodity": "broad"},
    },
    {
        "id": "LIT",
        "label": "Asset",
        "name": "Global X Lithium & Battery Tech ETF",
        "description": "EV and energy storage supply chain; battery metals demand",
        "metadata": {"subtype": "commodity_etf", "commodity": "lithium"},
    },
    {
        "id": "GDX",
        "label": "Asset",
        "name": "VanEck Gold Miners ETF",
        "description": "Large-cap gold miners; amplified gold signal with equity beta",
        "metadata": {"subtype": "commodity_etf", "commodity": "gold_miners"},
    },
    # Yield indices (^TNX, ^TYX, ^FVX, ^IRX) removed — tautological duplicates
    # of FRED indicators (DGS10, DGS30, DGS5, DGS3MO)
    # ── Bonds & Yields — Bond ETFs ─────────────────────────────────────
    {
        "id": "TLT",
        "label": "Asset",
        "name": "iShares 20+ Year Treasury Bond ETF",
        "description": "Long-duration bond ETF; rate sensitivity, flight to safety",
        "metadata": {"subtype": "bond_etf", "duration": "long"},
    },
    {
        "id": "SHY",
        "label": "Asset",
        "name": "iShares 1-3 Year Treasury Bond ETF",
        "description": "Short-duration; cash-like, Fed rate proxy",
        "metadata": {"subtype": "bond_etf", "duration": "short"},
    },
    {
        "id": "IEF",
        "label": "Asset",
        "name": "iShares 7-10 Year Treasury Bond ETF",
        "description": "Intermediate duration; balanced rate exposure",
        "metadata": {"subtype": "bond_etf", "duration": "intermediate"},
    },
    {
        "id": "HYG",
        "label": "Asset",
        "name": "iShares iBoxx High Yield Corporate Bond ETF",
        "description": "Junk bond ETF; credit risk appetite, distress canary",
        "metadata": {"subtype": "bond_etf", "credit_quality": "high_yield"},
    },
    {
        "id": "LQD",
        "label": "Asset",
        "name": "iShares iBoxx Investment Grade Corporate Bond ETF",
        "description": "Investment-grade corporate bonds; credit spread proxy",
        "metadata": {"subtype": "bond_etf", "credit_quality": "investment_grade"},
    },
    {
        "id": "AGG",
        "label": "Asset",
        "name": "iShares Core US Aggregate Bond ETF",
        "description": "Broadest US bond market ETF; total fixed income proxy",
        "metadata": {"subtype": "bond_etf", "credit_quality": "aggregate"},
    },
    {
        "id": "TIP",
        "label": "Asset",
        "name": "iShares TIPS Bond ETF",
        "description": "Inflation-protected Treasuries; real yield / breakeven proxy",
        "metadata": {"subtype": "bond_etf", "credit_quality": "tips"},
    },
    {
        "id": "JNK",
        "label": "Asset",
        "name": "SPDR Bloomberg High Yield Bond ETF",
        "description": "Second HY benchmark alongside HYG; different underlying index",
        "metadata": {"subtype": "bond_etf", "credit_quality": "high_yield"},
    },
    {
        "id": "BKLN",
        "label": "Asset",
        "name": "Invesco Senior Loan ETF",
        "description": "Floating-rate leveraged loans; different rate sensitivity than fixed HY",
        "metadata": {"subtype": "bond_etf", "credit_quality": "senior_loans"},
    },
    {
        "id": "EMB",
        "label": "Asset",
        "name": "iShares J.P. Morgan EM Bond ETF",
        "description": "Emerging market sovereign bonds; EM credit stress signal",
        "metadata": {"subtype": "bond_etf", "credit_quality": "em_sovereign"},
    },
    # ── Currency ETFs ─────────────────────────────────────────────────
    {
        "id": "DX-Y.NYB",
        "label": "Index",
        "name": "US Dollar Index (DXY)",
        "description": "Broad dollar strength; inverse to gold, commodities, EM",
        "metadata": {"category": "currency"},
    },
    {
        "id": "UUP",
        "label": "Asset",
        "name": "Invesco DB US Dollar Bullish ETF",
        "description": "Long USD position; liquid DXY proxy",
        "metadata": {"subtype": "currency_etf", "currency": "USD"},
    },
    {
        "id": "FXE",
        "label": "Asset",
        "name": "CurrencyShares Euro Trust",
        "description": "Euro pure play; EUR/USD proxy",
        "metadata": {"subtype": "currency_etf", "currency": "EUR"},
    },
    {
        "id": "FXY",
        "label": "Asset",
        "name": "CurrencyShares Japanese Yen Trust",
        "description": "Yen signal; BoJ policy sensitivity, carry trade indicator",
        "metadata": {"subtype": "currency_etf", "currency": "JPY"},
    },
    # ── International ETFs ─────────────────────────────────────────────
    {
        "id": "IEFA",
        "label": "Asset",
        "name": "iShares Core MSCI EAFE ETF",
        "description": "Developed markets ex-US/Canada; Europe, Japan, Australia",
        "metadata": {"subtype": "international_etf", "region": "developed_ex_us"},
    },
    {
        "id": "VEA",
        "label": "Asset",
        "name": "Vanguard FTSE Developed Markets ETF",
        "description": "Similar developed ex-US; slightly different index",
        "metadata": {"subtype": "international_etf", "region": "developed_ex_us"},
    },
    {
        "id": "IEMG",
        "label": "Asset",
        "name": "iShares Core MSCI Emerging Markets ETF",
        "description": "Broad EM including China, India, Brazil, Taiwan",
        "metadata": {"subtype": "international_etf", "region": "emerging"},
    },
    {
        "id": "EEM",
        "label": "Asset",
        "name": "iShares MSCI Emerging Markets ETF",
        "description": "Older, more liquid EM ETF; heavier China weight",
        "metadata": {"subtype": "international_etf", "region": "emerging"},
    },
    {
        "id": "EMXC",
        "label": "Asset",
        "name": "iShares MSCI EM ex China ETF",
        "description": "EM without China risk; India, Taiwan, Korea, Brazil",
        "metadata": {"subtype": "international_etf", "region": "emerging_ex_china"},
    },
    {
        "id": "VXUS",
        "label": "Asset",
        "name": "Vanguard Total International Stock ETF",
        "description": "Entire world ex-US; one-stop international allocation",
        "metadata": {"subtype": "international_etf", "region": "global_ex_us"},
    },
    {
        "id": "EWJ",
        "label": "Asset",
        "name": "iShares MSCI Japan ETF",
        "description": "Japan equities; yen carry trade, BOJ policy sensitive",
        "metadata": {"subtype": "international_etf", "country": "Japan"},
    },
    {
        "id": "FXI",
        "label": "Asset",
        "name": "iShares China Large-Cap ETF",
        "description": "China H-shares; geopolitics, stimulus, property crisis",
        "metadata": {"subtype": "international_etf", "country": "China"},
    },
    {
        "id": "KWEB",
        "label": "Asset",
        "name": "KraneShares CSI China Internet ETF",
        "description": "Chinese tech (Alibaba, Tencent, JD); regulatory risk",
        "metadata": {"subtype": "international_etf", "country": "China"},
    },
    {
        "id": "INDA",
        "label": "Asset",
        "name": "iShares MSCI India ETF",
        "description": "India equities; fastest-growing large economy",
        "metadata": {"subtype": "international_etf", "country": "India"},
    },
    {
        "id": "EWG",
        "label": "Asset",
        "name": "iShares MSCI Germany ETF",
        "description": "Germany/Europe manufacturing; auto, industrial proxy",
        "metadata": {"subtype": "international_etf", "country": "Germany"},
    },
    {
        "id": "VNM",
        "label": "Asset",
        "name": "VanEck Vietnam ETF",
        "description": "Vietnam; supply chain shift beneficiary",
        "metadata": {"subtype": "international_etf", "country": "Vietnam"},
    },
    {
        "id": "KSA",
        "label": "Asset",
        "name": "iShares MSCI Saudi Arabia ETF",
        "description": "Saudi equities; oil revenue, Vision 2030 diversification",
        "metadata": {"subtype": "international_etf", "country": "Saudi Arabia"},
    },
    {
        "id": "EWZ",
        "label": "Asset",
        "name": "iShares MSCI Brazil ETF",
        "description": "Brazil; commodity exporter, EM carry trade",
        "metadata": {"subtype": "international_etf", "country": "Brazil"},
    },
    {
        "id": "EMLC",
        "label": "Asset",
        "name": "VanEck EM Local Currency Bond ETF",
        "description": "EM sovereign debt in local currency; FX + credit risk",
        "metadata": {"subtype": "international_etf", "region": "emerging_bonds"},
    },
    {
        "id": "EWT",
        "label": "Asset",
        "name": "iShares MSCI Taiwan ETF",
        "description": "TSMC home market; semiconductor supply chain, geopolitical risk",
        "metadata": {"subtype": "international_etf", "country": "Taiwan"},
    },
    {
        "id": "EWY",
        "label": "Asset",
        "name": "iShares MSCI South Korea ETF",
        "description": "Samsung, SK Hynix; memory chips, manufacturing",
        "metadata": {"subtype": "international_etf", "country": "South Korea"},
    },
    # ── Crypto ─────────────────────────────────────────────────────────
    {
        "id": "BTC-USD",
        "label": "Currency",
        "name": "Bitcoin",
        "description": "Digital gold narrative; institutional adoption, halving cycle",
        "metadata": {"subtype": "crypto"},
    },
    {
        "id": "ETH-USD",
        "label": "Currency",
        "name": "Ethereum",
        "description": "Smart contract platform; DeFi, staking yield, L2 ecosystem",
        "metadata": {"subtype": "crypto"},
    },
    {
        "id": "SOL-USD",
        "label": "Currency",
        "name": "Solana",
        "description": "High-throughput L1; DeFi/NFT/meme coin activity, retail proxy",
        "metadata": {"subtype": "crypto"},
    },
    # ── FRED Economic Indicators ────────────────────────────────────────
    # Auto-generated from FRED_INDICATORS metadata in tickers.py.
    # These need to exist as nodes so the correlation engine can create
    # CORRELATES_WITH edges between indicators and assets.
    *[
        {
            "id": series_id,
            "label": "Indicator",
            "name": meta["name"],
            "description": f"{meta['category']} indicator ({meta['frequency']})",
            "metadata": {
                "unit": meta["unit"],
                "metric": meta["metric"],
                "frequency": meta["frequency"],
                "category": meta["category"],
            },
        }
        for series_id, meta in FRED_INDICATORS.items()
    ],
    # ── Sector Entities ────────────────────────────────────────────────
    # These are the logical sectors that companies BELONG_TO.
    {
        "id": "TECHNOLOGY",
        "label": "Sector",
        "name": "Technology",
        "description": "Information technology, software, semiconductors, hardware",
        "metadata": {},
    },
    {
        "id": "FINANCIALS",
        "label": "Sector",
        "name": "Financials",
        "description": "Banks, insurance, asset management, payments",
        "metadata": {},
    },
    {
        "id": "HEALTHCARE",
        "label": "Sector",
        "name": "Healthcare",
        "description": "Pharmaceuticals, biotech, health insurance, medical devices",
        "metadata": {},
    },
    {
        "id": "ENERGY",
        "label": "Sector",
        "name": "Energy",
        "description": "Oil, gas, coal, and renewable energy producers and servicers",
        "metadata": {},
    },
    {
        "id": "INDUSTRIALS",
        "label": "Sector",
        "name": "Industrials",
        "description": "Aerospace, defense, machinery, transportation, construction",
        "metadata": {},
    },
    {
        "id": "CONSUMER_DISCRETIONARY",
        "label": "Sector",
        "name": "Consumer Discretionary",
        "description": "Non-essential consumer goods and services; cyclical spending",
        "metadata": {},
    },
    {
        "id": "CONSUMER_STAPLES",
        "label": "Sector",
        "name": "Consumer Staples",
        "description": "Essential consumer goods; food, beverage, household products",
        "metadata": {},
    },
    {
        "id": "UTILITIES",
        "label": "Sector",
        "name": "Utilities",
        "description": "Electric, gas, water utilities; regulated, dividend-heavy",
        "metadata": {},
    },
    {
        "id": "MATERIALS",
        "label": "Sector",
        "name": "Materials",
        "description": "Chemicals, metals, mining, packaging, building materials",
        "metadata": {},
    },
    {
        "id": "REAL_ESTATE",
        "label": "Sector",
        "name": "Real Estate",
        "description": "REITs and real estate management/development",
        "metadata": {},
    },
    {
        "id": "COMMUNICATION_SERVICES",
        "label": "Sector",
        "name": "Communication Services",
        "description": "Media, entertainment, interactive media, telecom",
        "metadata": {},
    },
    # ── Country Entities ───────────────────────────────────────────────
    # Key countries for international ETF relationships and geopolitics.
    {
        "id": "US",
        "label": "Country",
        "name": "United States",
        "description": "World's largest economy; reserve currency issuer",
        "metadata": {},
    },
    {
        "id": "CHINA",
        "label": "Country",
        "name": "China",
        "description": "Second-largest economy; manufacturing, trade tensions",
        "metadata": {},
    },
    {
        "id": "JAPAN",
        "label": "Country",
        "name": "Japan",
        "description": "Third-largest economy; BOJ monetary policy, aging demographics",
        "metadata": {},
    },
    {
        "id": "INDIA",
        "label": "Country",
        "name": "India",
        "description": "Fastest-growing large economy; demographics, tech services",
        "metadata": {},
    },
    {
        "id": "GERMANY",
        "label": "Country",
        "name": "Germany",
        "description": "Europe's largest economy; auto, manufacturing, export engine",
        "metadata": {},
    },
    {
        "id": "BRAZIL",
        "label": "Country",
        "name": "Brazil",
        "description": "Largest Latin American economy; commodities, agriculture",
        "metadata": {},
    },
    {
        "id": "SAUDI_ARABIA",
        "label": "Country",
        "name": "Saudi Arabia",
        "description": "Top oil exporter; OPEC leader, Vision 2030 diversification",
        "metadata": {},
    },
    {
        "id": "TAIWAN",
        "label": "Country",
        "name": "Taiwan",
        "description": "Global semiconductor hub; TSMC, geopolitical flashpoint",
        "metadata": {},
    },
    {
        "id": "SOUTH_KOREA",
        "label": "Country",
        "name": "South Korea",
        "description": "Memory chips, consumer electronics, shipbuilding",
        "metadata": {},
    },
    {
        "id": "VIETNAM",
        "label": "Country",
        "name": "Vietnam",
        "description": "Supply chain diversification beneficiary; manufacturing growth",
        "metadata": {},
    },
    {
        "id": "NETHERLANDS",
        "label": "Country",
        "name": "Netherlands",
        "description": "ASML headquarters; key node in global semiconductor supply chain",
        "metadata": {},
    },
    {
        "id": "IRAN",
        "label": "Country",
        "name": "Iran",
        "description": "Major oil producer; geopolitical hotspot, OPEC member",
        "metadata": {},
    },
    {
        "id": "RUSSIA",
        "label": "Country",
        "name": "Russia",
        "description": "Major oil/gas exporter; sanctions, geopolitical risk",
        "metadata": {},
    },
    {
        "id": "UK",
        "label": "Country",
        "name": "United Kingdom",
        "description": "Major economy; BOE monetary policy, financial services hub",
        "metadata": {},
    },
    {
        "id": "AUSTRALIA",
        "label": "Country",
        "name": "Australia",
        "description": "Mining, commodities, Pacific trade, iron ore exporter",
        "metadata": {},
    },
    {
        "id": "ISRAEL",
        "label": "Country",
        "name": "Israel",
        "description": "Defense tech, geopolitical flashpoint, Middle East conflicts",
        "metadata": {},
    },
    {
        "id": "FRANCE",
        "label": "Country",
        "name": "France",
        "description": "Major EU economy; defense, luxury goods, energy",
        "metadata": {},
    },
    {
        "id": "CANADA",
        "label": "Country",
        "name": "Canada",
        "description": "Energy, mining, banking; major US trade partner",
        "metadata": {},
    },
    {
        "id": "MEXICO",
        "label": "Country",
        "name": "Mexico",
        "description": "Manufacturing, nearshoring, US trade partner",
        "metadata": {},
    },
    {
        "id": "INDONESIA",
        "label": "Country",
        "name": "Indonesia",
        "description": "Largest Southeast Asian economy; nickel, palm oil",
        "metadata": {},
    },
    {
        "id": "TURKEY",
        "label": "Country",
        "name": "Turkey",
        "description": "Emerging market; geopolitical bridge, currency volatility",
        "metadata": {},
    },
    {
        "id": "HONG_KONG",
        "label": "Country",
        "name": "Hong Kong",
        "description": "Financial hub; gateway to Chinese markets",
        "metadata": {},
    },
]


# =============================================================================
# Structural Relationships
# =============================================================================
# Obvious structural connections seeded at setup. The AI will discover more
# nuanced relationships over time through analysis.
#
# Each tuple: (from_label, from_id, to_label, to_id, rel_type, properties)

STRUCTURAL_RELATIONSHIPS: list[tuple[str, str, str, str, str, dict]] = [
    # ── BELONGS_TO: Company → Sector ───────────────────────────────────
    # Mega-cap tech
    ("Company", "AAPL", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "MSFT", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "NVDA", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "GOOGL", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "META", "Sector", "COMMUNICATION_SERVICES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "AMZN", "Sector", "CONSUMER_DISCRETIONARY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "TSLA", "Sector", "CONSUMER_DISCRETIONARY", "BELONGS_TO", {"source": "seed"}),
    # Semiconductors
    ("Company", "AVGO", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "AMD", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "TSM", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "ASML", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "MU", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "QCOM", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "INTC", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "ARM", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    # Banks / Finance
    ("Company", "JPM", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "GS", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "BAC", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "BRK-B", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "V", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "MA", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "KKR", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "COIN", "Sector", "FINANCIALS", "BELONGS_TO", {"source": "seed"}),
    # Energy
    ("Company", "XOM", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "CVX", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "COP", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "SLB", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    # Defense / Aerospace
    ("Company", "LMT", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "RTX", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "NOC", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "GD", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    # Industrials
    ("Company", "CAT", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "GE", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "HON", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "DE", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "UNP", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "VRT", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    ("Company", "ETN", "Sector", "INDUSTRIALS", "BELONGS_TO", {"source": "seed"}),
    # Healthcare
    ("Company", "UNH", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "LLY", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "JNJ", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "PFE", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "ABBV", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    # Retail / Consumer
    ("Company", "WMT", "Sector", "CONSUMER_STAPLES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "COST", "Sector", "CONSUMER_STAPLES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "HD", "Sector", "CONSUMER_DISCRETIONARY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "NKE", "Sector", "CONSUMER_DISCRETIONARY", "BELONGS_TO", {"source": "seed"}),
    # AI / Cloud / Software
    ("Company", "CRM", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "PLTR", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "NOW", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "SNOW", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "ORCL", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "ANET", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "MRVL", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "TTD", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    # Cybersecurity
    ("Company", "PANW", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "CRWD", "Sector", "TECHNOLOGY", "BELONGS_TO", {"source": "seed"}),
    # Media / Telecom
    ("Company", "NFLX", "Sector", "COMMUNICATION_SERVICES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "DIS", "Sector", "COMMUNICATION_SERVICES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "VZ", "Sector", "COMMUNICATION_SERVICES", "BELONGS_TO", {"source": "seed"}),
    # Energy (remaining)
    ("Company", "OXY", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "MPC", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "PSX", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    ("Company", "KMI", "Sector", "ENERGY", "BELONGS_TO", {"source": "seed"}),
    # Healthcare (remaining)
    ("Company", "MRNA", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "AMGN", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "MRK", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "GILD", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "REGN", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "VRTX", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "BIIB", "Sector", "HEALTHCARE", "BELONGS_TO", {"source": "seed"}),
    # Consumer Discretionary (remaining)
    ("Company", "MCD", "Sector", "CONSUMER_DISCRETIONARY", "BELONGS_TO", {"source": "seed"}),
    # Consumer Staples (remaining)
    ("Company", "PG", "Sector", "CONSUMER_STAPLES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "KO", "Sector", "CONSUMER_STAPLES", "BELONGS_TO", {"source": "seed"}),
    # Real Estate
    ("Company", "PLD", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "AMT", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "EQIX", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "DLR", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "WELL", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "SPG", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    ("Company", "O", "Sector", "REAL_ESTATE", "BELONGS_TO", {"source": "seed"}),
    # Utilities
    ("Company", "CEG", "Sector", "UTILITIES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "NEE", "Sector", "UTILITIES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "VST", "Sector", "UTILITIES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "SO", "Sector", "UTILITIES", "BELONGS_TO", {"source": "seed"}),
    ("Company", "DUK", "Sector", "UTILITIES", "BELONGS_TO", {"source": "seed"}),
    # ── LOCATED_IN: International ETFs → Country ───────────────────────
    ("Asset", "EWJ", "Country", "JAPAN", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "FXI", "Country", "CHINA", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "KWEB", "Country", "CHINA", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "INDA", "Country", "INDIA", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "EWG", "Country", "GERMANY", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "VNM", "Country", "VIETNAM", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "KSA", "Country", "SAUDI_ARABIA", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "EWZ", "Country", "BRAZIL", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "EWT", "Country", "TAIWAN", "LOCATED_IN", {"source": "seed"}),
    ("Asset", "EWY", "Country", "SOUTH_KOREA", "LOCATED_IN", {"source": "seed"}),
    # TSM is headquartered in Taiwan
    ("Company", "TSM", "Country", "TAIWAN", "LOCATED_IN", {"source": "seed"}),
    ("Company", "ASML", "Country", "NETHERLANDS", "LOCATED_IN", {"source": "seed"}),
    # ── DEPENDS_ON: Key supply chain dependencies ──────────────────────
    # NVIDIA, AMD, Apple all depend on TSMC to fabricate their chips
    (
        "Company",
        "NVDA",
        "Company",
        "TSM",
        "DEPENDS_ON",
        {"source": "seed", "reason": "fab_partner"},
    ),
    ("Company", "AMD", "Company", "TSM", "DEPENDS_ON", {"source": "seed", "reason": "fab_partner"}),
    (
        "Company",
        "AAPL",
        "Company",
        "TSM",
        "DEPENDS_ON",
        {"source": "seed", "reason": "fab_partner"},
    ),
    # TSMC depends on ASML — only source of EUV lithography machines
    (
        "Company",
        "TSM",
        "Company",
        "ASML",
        "DEPENDS_ON",
        {"source": "seed", "reason": "sole_supplier_euv"},
    ),
    # ── SUPPLIES_TO: Manufacturing relationships ───────────────────────
    ("Company", "TSM", "Company", "NVDA", "SUPPLIES_TO", {"source": "seed", "product": "chips"}),
    ("Company", "TSM", "Company", "AMD", "SUPPLIES_TO", {"source": "seed", "product": "chips"}),
    ("Company", "TSM", "Company", "AAPL", "SUPPLIES_TO", {"source": "seed", "product": "chips"}),
    ("Company", "TSM", "Company", "QCOM", "SUPPLIES_TO", {"source": "seed", "product": "chips"}),
    (
        "Company",
        "ASML",
        "Company",
        "TSM",
        "SUPPLIES_TO",
        {"source": "seed", "product": "euv_machines"},
    ),
    (
        "Company",
        "ASML",
        "Company",
        "INTC",
        "SUPPLIES_TO",
        {"source": "seed", "product": "euv_machines"},
    ),
    # ── COMPETES_WITH: Direct competitors ──────────────────────────────
    ("Company", "NVDA", "Company", "AMD", "COMPETES_WITH", {"source": "seed", "arena": "gpu"}),
    ("Company", "AMD", "Company", "INTC", "COMPETES_WITH", {"source": "seed", "arena": "cpu"}),
    (
        "Company",
        "MSFT",
        "Company",
        "GOOGL",
        "COMPETES_WITH",
        {"source": "seed", "arena": "cloud_ai"},
    ),
    ("Company", "AMZN", "Company", "MSFT", "COMPETES_WITH", {"source": "seed", "arena": "cloud"}),
    ("Company", "V", "Company", "MA", "COMPETES_WITH", {"source": "seed", "arena": "payments"}),
    ("Company", "XOM", "Company", "CVX", "COMPETES_WITH", {"source": "seed", "arena": "oil_major"}),
    ("Company", "LMT", "Company", "RTX", "COMPETES_WITH", {"source": "seed", "arena": "defense"}),
    (
        "Company",
        "PANW",
        "Company",
        "CRWD",
        "COMPETES_WITH",
        {"source": "seed", "arena": "cybersecurity"},
    ),
    (
        "Company",
        "CRM",
        "Company",
        "NOW",
        "COMPETES_WITH",
        {"source": "seed", "arena": "enterprise_saas"},
    ),
    (
        "Company",
        "NFLX",
        "Company",
        "DIS",
        "COMPETES_WITH",
        {"source": "seed", "arena": "streaming"},
    ),
    ("Company", "WMT", "Company", "COST", "COMPETES_WITH", {"source": "seed", "arena": "retail"}),
]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Seed Neo4j with all tracked entities and structural relationships."""
    logger.info("Starting entity seed...")

    with GraphStorage() as graph:
        # Ensure indexes/constraints exist before inserting
        graph.setup_indexes()

        # ── Validate all entities through Pydantic ─────────────────────
        # Group validated entities by label so we can batch-insert per type
        entities_by_label: dict[str, list[dict]] = {}
        validation_errors = 0

        for raw in ENTITY_SEED_DATA:
            try:
                entity = EntityCreate(**raw)
                label = entity.label
                properties = entity.to_graph_properties()

                if label not in entities_by_label:
                    entities_by_label[label] = []
                entities_by_label[label].append(properties)

            except Exception:
                validation_errors += 1
                logger.exception("Validation failed for entity: %s", raw.get("id", "UNKNOWN"))

        if validation_errors > 0:
            logger.error("%d entities failed validation — aborting", validation_errors)
            sys.exit(1)

        # ── Batch insert entities by label ─────────────────────────────
        total_entities = 0
        for label, entities in entities_by_label.items():
            count = graph.create_entities_batch(label, entities)
            total_entities += count
            logger.info("  %s: %d entities", label, count)

        # ── Batch create structural relationships ─────────────────────
        relationship_batch = []
        for _, from_id, _, to_id, rel_type, props in STRUCTURAL_RELATIONSHIPS:
            relationship_batch.append({
                "from_id": from_id,
                "to_id": to_id,
                "rel_type": rel_type,
                "props": props,
            })

        total_relationships = graph.create_relationships_batch(relationship_batch)

        # ── Summary ────────────────────────────────────────────────────
        logger.info("Seed complete!")
        logger.info("  Entities created/updated:      %d", total_entities)
        logger.info("  Relationships created/updated:  %d", total_relationships)


if __name__ == "__main__":
    main()
