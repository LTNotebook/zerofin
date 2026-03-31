"""Single source of truth for all tracked tickers and FRED indicators.

Every collector pulls its list from here — no hardcoded tickers anywhere else.
"""

from __future__ import annotations

from typing import TypedDict

# =============================================================================
# yfinance — Market Data Tickers
# =============================================================================

# Major US stock market indices
US_INDICES: list[str] = [
    "^GSPC",  # S&P 500
    "^DJI",  # Dow Jones Industrial Average
    "^IXIC",  # Nasdaq Composite
    "^NDX",  # Nasdaq 100
    "^RUT",  # Russell 2000 (small-cap)
    "^VIX",  # CBOE Volatility Index ("fear gauge")
    "^GSPTSE",  # S&P/TSX Composite (Canada)
    "^W5000",  # Wilshire 5000 (broadest US market)
    # Volatility indices
    "^VVIX",  # VIX of VIX — volatility of volatility, leads VIX
    "^MOVE",  # ICE BofAML Bond Volatility — "VIX for Treasuries"
    "^GVZ",  # CBOE Gold Volatility
    "^OVX",  # CBOE Crude Oil Volatility
]

# Select Sector SPDRs (all 11 GICS sectors) plus thematic/sub-sector ETFs
SECTOR_ETFS: list[str] = [
    # 11 GICS sectors
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLB",  # Materials
    "XLRE",  # Real Estate
    "XLC",  # Communication Services
    # Sub-sector / thematic
    "SMH",  # Semiconductors (VanEck)
    "AIQ",  # AI & Technology (Global X)
    "BOTZ",  # Robotics & AI (Global X)
    "ROBO",  # Robotics & Automation
    "ITA",  # US Aerospace & Defense
    "HACK",  # Cybersecurity (ETFMG)
    "IBB",  # Biotech (iShares)
    "XHB",  # Homebuilders (SPDR)
    "KRE",  # Regional Banks (SPDR)
    "RSP",  # S&P 500 Equal Weight
    # Factor ETFs — regime detection
    "MTUM",  # Momentum factor (trend-following regime signal)
    "QUAL",  # Quality factor (high ROE, low debt — late-cycle signal)
    "USMV",  # Min Volatility factor (risk-off equity signal)
    # Alternative / thematic proxies
    "JETS",  # U.S. Global Jets ETF (airline demand / travel spending)
    "BDRY",  # Breakwave Dry Bulk Shipping (global trade volume proxy)
]

# Individual US stocks organized by sector/theme
KEY_STOCKS: list[str] = [
    # Mega-cap tech
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    # Semiconductors
    "AVGO",
    "AMD",
    "TSM",
    "ASML",
    "MU",
    "QCOM",
    "INTC",
    "ARM",
    # Banks / Finance
    "JPM",
    "GS",
    "BAC",
    "BRK-B",
    "V",
    "MA",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "OXY",
    "MPC",  # Marathon Petroleum — refining margins signal
    "PSX",  # Phillips 66 — midstream + refining
    "KMI",  # Kinder Morgan — natural gas pipeline infrastructure
    # Defense / Aerospace
    "BA",   # Boeing — commercial + defense aerospace
    "LMT",
    "RTX",
    "NOC",
    "GD",
    # Healthcare
    "UNH",
    "LLY",
    "JNJ",
    "PFE",
    "ABBV",
    "MRNA",
    "AMGN",
    "MRK",  # Merck — oncology (Keytruda), top-selling drug globally
    "GILD",  # Gilead Sciences — antivirals / HIV franchise
    "REGN",  # Regeneron — immunology (Dupixent) + ophthalmology
    "VRTX",  # Vertex Pharma — cystic fibrosis monopoly + gene editing
    "BIIB",  # Biogen — neurology / Alzheimer's
    # Retail / Consumer
    "WMT",
    "COST",
    "HD",
    "NKE",
    "MCD",
    # Consumer Staples
    "PG",
    "KO",
    # Industrials
    "CAT",
    "GE",
    "HON",
    "DE",
    "UNP",
    # AI / Cloud / Software
    "CRM",
    "PLTR",
    "NOW",
    "SNOW",
    "ORCL",
    # AI Infrastructure
    "CEG",
    "VRT",
    "ETN",
    "ANET",
    "MRVL",
    # Cybersecurity
    "PANW",
    "CRWD",
    # Media / Telecom
    "NFLX",
    "DIS",
    "VZ",
    # REITs — individual names beyond XLRE sector ETF
    "PLD",  # Prologis — industrial/logistics, e-commerce proxy
    "AMT",  # American Tower — cell towers, rate-sensitive + wireless demand
    "EQIX",  # Equinix — data centers, AI infrastructure demand
    "DLR",  # Digital Realty — data centers, AI buildout signal
    "WELL",  # Welltower — healthcare REIT, aging demographics
    "SPG",  # Simon Property Group — retail/malls, consumer spending
    "O",  # Realty Income — net lease, bond-like behavior
    # Utilities — traditional + AI power demand
    "NEE",  # NextEra Energy — renewables + AI datacenter power
    "VST",  # Vistra — deregulated power, AI energy demand (pairs with CEG)
    "SO",  # Southern Company — traditional regulated utility
    "DUK",  # Duke Energy — traditional + AI power transition
    # Signal stocks (unusual insider activity or thematic bets)
    "KKR",
    "TTD",
    "COIN",
    # China individual stocks
    "BABA",  # Alibaba — China e-commerce bellwether
    "PDD",  # PDD Holdings / Temu — China e-commerce + international expansion
]

# Commodity futures and commodity-focused ETFs.
# GLD is an ETF (not a futures contract), separate from GC=F gold futures.
COMMODITIES: list[str] = [
    # Futures
    "CL=F",  # WTI Crude Oil
    "BZ=F",  # Brent Crude Oil
    "GC=F",  # Gold
    "SI=F",  # Silver
    "HG=F",  # Copper ("Dr. Copper" — global growth proxy)
    "NG=F",  # Natural Gas
    "ZC=F",  # Corn
    "ZW=F",  # Wheat
    "ZS=F",  # Soybeans
    # ETFs (used where no liquid futures exist on yfinance)
    "GLD",  # SPDR Gold Trust ETF — different instrument than GC=F futures
    "URA",  # Global X Uranium ETF
    "DBA",  # Invesco DB Agriculture Fund
    "DBC",  # Invesco DB Commodity Index
    "LIT",  # Global X Lithium & Battery ETF
    "GDX",  # VanEck Gold Miners ETF — amplified gold signal with equity beta
]

# Treasury yields and fixed-income ETFs
BONDS_YIELDS: list[str] = [
    # Yields now tracked via FRED indicators (DGS10, DGS30, DGS5, DGS3MO)
    # ^TNX, ^TYX, ^FVX, ^IRX removed — tautological duplicates of FRED series
    # Bond ETFs
    "TLT",  # iShares 20+ Year Treasury
    "SHY",  # iShares 1-3 Year Treasury
    "IEF",  # iShares 7-10 Year Treasury
    "HYG",  # iShares High Yield Corporate
    "LQD",  # iShares Investment Grade Corporate
    "AGG",  # iShares Core US Aggregate Bond
    "TIP",  # iShares TIPS (inflation-protected)
    # Credit / loans
    "JNK",  # SPDR Bloomberg High Yield (second HY benchmark alongside HYG)
    "BKLN",  # Invesco Senior Loan ETF (floating-rate leveraged loans)
    "EMB",  # iShares EM Bond ETF (emerging market sovereign bonds)
]

# International / regional equity and bond ETFs
INTERNATIONAL: list[str] = [
    "IEFA",  # iShares Core MSCI EAFE (developed ex-US)
    "VEA",  # Vanguard FTSE Developed Markets
    "IEMG",  # iShares Core MSCI Emerging Markets
    "EEM",  # iShares MSCI Emerging Markets (older, more liquid)
    "EMXC",  # iShares MSCI EM ex China
    "VXUS",  # Vanguard Total International
    "EWJ",  # iShares MSCI Japan
    "FXI",  # iShares China Large-Cap
    "KWEB",  # KraneShares CSI China Internet
    "INDA",  # iShares MSCI India
    "EWG",  # iShares MSCI Germany
    "VNM",  # VanEck Vietnam ETF
    "KSA",  # iShares MSCI Saudi Arabia
    "EWZ",  # iShares MSCI Brazil
    "EMLC",  # VanEck EM Local Currency Bond
    "EWT",  # iShares MSCI Taiwan
    "EWY",  # iShares MSCI South Korea
]

# Currency ETFs and indices
CURRENCIES: list[str] = [
    "DX-Y.NYB",  # US Dollar Index (DXY)
    "UUP",  # Invesco DB US Dollar Bullish ETF
    "FXE",  # CurrencyShares Euro Trust
    "FXY",  # CurrencyShares Japanese Yen Trust
]

# Major cryptocurrencies
CRYPTO: list[str] = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
]

# =============================================================================
# Stock-to-Sector ETF Mapping
# =============================================================================
# Maps each individual stock to its GICS sector ETF and optional sub-sector ETF.
# Used by the correlation engine for market + sector beta removal.
# Source: stockanalysis.com ETF holdings, GICS classifications (verified March 2026)

STOCK_SECTOR_MAP: dict[str, dict[str, str]] = {
    # Technology
    "AAPL":  {"sector": "XLK"},
    "MSFT":  {"sector": "XLK"},
    "NVDA":  {"sector": "XLK", "sub_sector": "SMH"},
    "AVGO":  {"sector": "XLK", "sub_sector": "SMH"},
    "AMD":   {"sector": "XLK", "sub_sector": "SMH"},
    "TSM":   {"sector": "XLK", "sub_sector": "SMH"},
    "ASML":  {"sector": "XLK", "sub_sector": "SMH"},
    "MU":    {"sector": "XLK", "sub_sector": "SMH"},
    "QCOM":  {"sector": "XLK", "sub_sector": "SMH"},
    "INTC":  {"sector": "XLK", "sub_sector": "SMH"},
    "ARM":   {"sector": "XLK", "sub_sector": "SMH"},
    "CRWD":  {"sector": "XLK", "sub_sector": "HACK"},
    "PANW":  {"sector": "XLK", "sub_sector": "HACK"},
    # Communication Services
    "GOOGL": {"sector": "XLC"},
    "META":  {"sector": "XLC"},
    # Consumer Discretionary
    "AMZN":  {"sector": "XLY"},
    "TSLA":  {"sector": "XLY"},
    "HD":    {"sector": "XLY"},
    "MCD":   {"sector": "XLY"},
    # Chinese ADRs — use KWEB (China internet ETF) as sector proxy
    "BABA":  {"sector": "KWEB"},
    "PDD":   {"sector": "KWEB"},
    # Financials
    "JPM":   {"sector": "XLF"},
    "GS":    {"sector": "XLF"},
    "BAC":   {"sector": "XLF"},
    "BRK-B": {"sector": "XLF"},
    "V":     {"sector": "XLF"},
    "MA":    {"sector": "XLF"},
    # Energy
    "XOM":   {"sector": "XLE"},
    "CVX":   {"sector": "XLE"},
    "COP":   {"sector": "XLE"},
    "OXY":   {"sector": "XLE"},
    "MPC":   {"sector": "XLE"},
    "PSX":   {"sector": "XLE"},
    "KMI":   {"sector": "XLE"},
    # Industrials
    "BA":    {"sector": "XLI", "sub_sector": "ITA"},
    "LMT":   {"sector": "XLI", "sub_sector": "ITA"},
    "RTX":   {"sector": "XLI", "sub_sector": "ITA"},
    "GD":    {"sector": "XLI", "sub_sector": "ITA"},
    "NOC":   {"sector": "XLI", "sub_sector": "ITA"},
    # Health Care
    "UNH":   {"sector": "XLV"},
    "JNJ":   {"sector": "XLV"},
    "LLY":   {"sector": "XLV"},
    "PFE":   {"sector": "XLV"},
    "ABBV":  {"sector": "XLV"},
    "MRNA":  {"sector": "XLV", "sub_sector": "IBB"},
    "AMGN":  {"sector": "XLV", "sub_sector": "IBB"},
    "MRK":   {"sector": "XLV"},
    "GILD":  {"sector": "XLV", "sub_sector": "IBB"},
    "REGN":  {"sector": "XLV", "sub_sector": "IBB"},
    "VRTX":  {"sector": "XLV", "sub_sector": "IBB"},
    "BIIB":  {"sector": "XLV", "sub_sector": "IBB"},
    # Consumer Staples
    "WMT":   {"sector": "XLP"},
    "COST":  {"sector": "XLP"},
    "PG":    {"sector": "XLP"},
    "KO":    {"sector": "XLP"},
    # Energy Services
    "SLB":   {"sector": "XLE"},
    # Consumer Discretionary
    "NKE":   {"sector": "XLY"},
    # Industrials
    "CAT":   {"sector": "XLI"},
    "GE":    {"sector": "XLI"},
    "HON":   {"sector": "XLI"},
    "DE":    {"sector": "XLI"},
    "UNP":   {"sector": "XLI"},
    "VRT":   {"sector": "XLI"},
    "ETN":   {"sector": "XLI"},
    # Technology / Software
    "CRM":   {"sector": "XLK"},
    "PLTR":  {"sector": "XLK"},
    "NOW":   {"sector": "XLK"},
    "SNOW":  {"sector": "XLK"},
    "ORCL":  {"sector": "XLK"},
    "ANET":  {"sector": "XLK"},
    "MRVL":  {"sector": "XLK", "sub_sector": "SMH"},
    "TTD":   {"sector": "XLK"},
    # Communication Services
    "NFLX":  {"sector": "XLC"},
    "DIS":   {"sector": "XLC"},
    "VZ":    {"sector": "XLC"},
    # Financials
    "KKR":   {"sector": "XLF"},
    "COIN":  {"sector": "XLF"},
    # Utilities
    "CEG":   {"sector": "XLU"},
    "NEE":   {"sector": "XLU"},
    "VST":   {"sector": "XLU"},
    "SO":    {"sector": "XLU"},
    "DUK":   {"sector": "XLU"},
    # Real Estate
    "PLD":   {"sector": "XLRE"},
    "AMT":   {"sector": "XLRE"},
    "EQIX":  {"sector": "XLRE"},
    "DLR":   {"sector": "XLRE"},
    "WELL":  {"sector": "XLRE"},
    "SPG":   {"sector": "XLRE"},
    "O":     {"sector": "XLRE"},
}


# =============================================================================
# Redundancy Groups — near-identical securities that flood the correlation matrix
# =============================================================================
# When two securities track the same underlying, correlating them is meaningless
# (r > 0.95). We group them and pick one representative per group.
# The correlation engine skips intra-group pairs entirely.
# Review quarterly with scripts/check_redundancy.py (future).
# Source: ETF deduplication research, March 2026

REDUNDANCY_GROUPS: dict[str, dict] = {
    "em_equity": {
        "representative": "IEMG",
        "members": ["IEMG", "EEM", "EMXC"],
        "reason": "All track emerging markets equity; IEMG has largest AUM",
    },
    "developed_intl": {
        "representative": "IEFA",
        "members": ["IEFA", "VEA"],
        "reason": "Both track developed ex-US markets; near-identical holdings",
    },
    "broad_intl": {
        "representative": "VXUS",
        "members": ["VXUS"],
        "reason": "Broad international (EM + developed); overlaps with other intl groups",
    },
    "treasury_10y": {
        "representative": "DGS10",
        "members": ["DGS10"],
        "reason": "10-year Treasury yield (^TNX removed — tautological duplicate)",
    },
    "fed_funds_bounds": {
        "representative": "DFEDTARU",
        "members": ["DFEDTARU", "DFEDTARL"],
        "reason": "Upper and lower bounds move in lockstep — effectively the same series",
    },
    "cpi_variants": {
        "representative": "CPIAUCSL",
        "members": ["CPIAUCSL", "CPIAUCNS"],
        "reason": "Same CPI data, seasonally adjusted vs not; SA version is standard",
    },
    "thematic_ai_robotics": {
        "representative": "AIQ",
        "members": ["AIQ", "BOTZ", "ROBO"],
        "reason": "All track AI/robotics/automation themes; high overlap in holdings and behavior",
    },
    "corporate_profits": {
        "representative": "CPGDPAI",
        "members": ["CPGDPAI", "CP"],
        "reason": "Both measure corporate profits after tax; CPGDPAI includes IVA+CCAdj",
    },
}

# Build a quick lookup: ticker -> group name (for fast filtering in correlation engine)
REDUNDANCY_LOOKUP: dict[str, str] = {}
for group_name, group_info in REDUNDANCY_GROUPS.items():
    for member in group_info["members"]:
        REDUNDANCY_LOOKUP[member] = group_name




# All yfinance tickers combined (deduplicated, order preserved)
ALL_TICKERS: list[str] = list(
    dict.fromkeys(
        US_INDICES + SECTOR_ETFS + KEY_STOCKS + COMMODITIES
        + BONDS_YIELDS + INTERNATIONAL + CURRENCIES + CRYPTO
    )
)


# =============================================================================
# FRED — Economic Data Series
# =============================================================================

# CPI, PCE, PPI, and market-implied inflation expectations
FRED_INFLATION: list[str] = [
    "CPIAUCSL",  # CPI All Items (seasonally adjusted)
    "CPILFESL",  # Core CPI (ex food & energy)
    "CPIAUCNS",  # CPI All Items (not seasonally adjusted — for YoY)
    "PCEPI",  # PCE Price Index (headline)
    "PCEPILFE",  # Core PCE (the Fed's 2% target measure)
    "T5YIE",  # 5-Year Breakeven Inflation Rate
    "T10YIE",  # 10-Year Breakeven Inflation Rate
    "PPIFIS",  # PPI Final Demand (leads CPI by 1-3 months)
]

# Jobs, wages, and labor market health
FRED_EMPLOYMENT: list[str] = [
    "PAYEMS",  # Total Nonfarm Payrolls
    "UNRATE",  # Unemployment Rate
    "ICSA",  # Initial Jobless Claims (weekly, highest-frequency)
    "CCSA",  # Continued Claims
    "JTSJOL",  # JOLTS Job Openings
    "CES0500000003",  # Average Hourly Earnings, Total Private
    "CIVPART",  # Labor Force Participation Rate
    "U6RATE",  # U-6 Unemployment (broadest measure)
]

# GDP, industrial output, retail sales, business investment
FRED_GROWTH: list[str] = [
    "GDPC1",  # Real GDP
    "INDPRO",  # Industrial Production Index
    "RSAFS",  # Advance Retail Sales
    "RSXFS",  # Retail Sales Ex Food Services
    "DGORDER",  # Durable Goods Orders (business capex proxy)
    "CPGDPAI",  # Corporate Profits After Tax
    "TCU",  # Capacity Utilization: Total Index (overheating/slack signal)
    "CP",  # Corporate Profits After Tax (alt measure, leads S&P ~1 quarter)
]

# Housing starts, permits, prices, mortgage rates
FRED_HOUSING: list[str] = [
    "HOUST",  # Housing Starts
    "PERMIT",  # Building Permits (leads starts)
    "EXHOSLUSM495S",  # Existing Home Sales
    "CSUSHPINSA",  # Case-Shiller National Home Price Index
    "MORTGAGE30US",  # 30-Year Fixed Mortgage Rate
]

# Consumer confidence, income, spending, savings
FRED_CONSUMER: list[str] = [
    "UMCSENT",  # Univ of Michigan Consumer Sentiment
    "PI",  # Personal Income
    "PSAVERT",  # Personal Savings Rate
    "PCE",  # Personal Consumption Expenditures
]

# Factory orders and manufacturing hours.
# DGORDER intentionally omitted — already in FRED_GROWTH.
FRED_MANUFACTURING: list[str] = [
    "NEWORDER",  # Manufacturers' New Orders: All Manufacturing
    "AMTMNO",  # Manufacturers' New Orders: Total Manufacturing
    "AWHMAN",  # Avg Weekly Hours: Manufacturing (leading indicator)
]

# Fed funds rate, Treasury yields across the curve, SOFR.
# DFF (daily) kept instead of FEDFUNDS (monthly) for granularity.
FRED_RATES: list[str] = [
    "DFF",  # Effective Fed Funds Rate (daily)
    "DGS2",  # 2-Year Treasury
    "DGS5",  # 5-Year Treasury
    "DGS10",  # 10-Year Treasury
    "DGS30",  # 30-Year Treasury
    "DGS3MO",  # 3-Month Treasury
    "DFEDTARU",  # Fed Funds Target Range Upper Bound
    "DFEDTARL",  # Fed Funds Target Range Lower Bound
    "SOFR",  # Secured Overnight Financing Rate (replaced LIBOR)
]

# Corporate bond spreads, bank lending, and credit health
FRED_CREDIT: list[str] = [
    "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS
    "BAMLC0A0CM",  # ICE BofA US Investment Grade OAS
    "BAMLH0A0HYM2EY",  # ICE BofA High Yield Effective Yield
    # Bank lending and credit conditions
    "DRTSCILM",  # Banks Tightening C&I Loan Standards (Large Firms) — SLOOS proxy
    "DRALACBS",  # Delinquency Rate: All Loans, All Commercial Banks
    "TOTALSL",  # Total Consumer Credit Outstanding
    "TOTBKCR",  # Bank Credit, All Commercial Banks (credit impulse)
]

# Money supply, velocity, Fed balance sheet, reverse repo
FRED_MONEY: list[str] = [
    "M2SL",  # M2 Money Supply
    "M2V",  # Velocity of M2
    "WALCL",  # Fed Total Assets (balance sheet — QE/QT proxy)
    "RRPONTSYD",  # Overnight Reverse Repo (excess liquidity)
]

# Trade balance and dollar indices
FRED_TRADE: list[str] = [
    "BOPGSTB",  # Trade Balance: Goods & Services
    "DTWEXBGS",  # Nominal Broad US Dollar Index
    "DTWEXAFEGS",  # Nominal Advanced Foreign Economies Dollar Index
    "IPG21112N",  # Industrial Production: Crude Oil Mining (oil production index)
]

# Composite activity, financial stress, and recession indicators
FRED_LEADING: list[str] = [
    "CFNAI",  # Chicago Fed National Activity Index (85 indicators)
    "STLFSI2",  # St. Louis Fed Financial Stress Index
    "NFCI",  # Chicago Fed National Financial Conditions Index
    # ANFCI removed — tautological duplicate of NFCI (adjusted version)
    "SAHMCURRENT",  # Sahm Rule Recession Indicator
]

# Yield curve spreads — recession predictors
FRED_YIELD_CURVE: list[str] = [
    "T10Y2Y",  # 10Y minus 2Y (classic inversion signal)
    "T10Y3M",  # 10Y minus 3M (NY Fed preferred recession predictor)
    "T10YFF",  # 10Y minus Fed Funds Rate
]

# All FRED series combined (deduplicated, order preserved)
FRED_ALL: list[str] = list(
    dict.fromkeys(
        FRED_INFLATION
        + FRED_EMPLOYMENT
        + FRED_GROWTH
        + FRED_HOUSING
        + FRED_CONSUMER
        + FRED_MANUFACTURING
        + FRED_RATES
        + FRED_CREDIT
        + FRED_MONEY
        + FRED_TRADE
        + FRED_LEADING
        + FRED_YIELD_CURVE
    )
)


# =============================================================================
# FRED Indicator Metadata
# =============================================================================
# Maps each FRED series to structured metadata so downstream code (storage,
# analysis, briefings) knows what each number means without calling the API.


class FredMeta(TypedDict):
    """Metadata for a single FRED economic indicator."""

    name: str  # Human-readable name
    unit: str  # What it's measured in (percent, index_points, etc.)
    metric: str  # Kind of measurement (value, rate, spread, index, count)
    frequency: str  # Update cadence (daily, weekly, monthly, quarterly)
    category: str  # Which group it belongs to


FRED_INDICATORS: dict[str, FredMeta] = {
    # --- Inflation ---
    "CPIAUCSL": {
        "name": "CPI: All Items (Seasonally Adjusted)",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "inflation",
    },
    "CPILFESL": {
        "name": "Core CPI (Ex Food & Energy)",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "inflation",
    },
    "CPIAUCNS": {
        "name": "CPI: All Items (Not Seasonally Adjusted)",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "inflation",
    },
    "PCEPI": {
        "name": "PCE Price Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "inflation",
    },
    "PCEPILFE": {
        "name": "Core PCE Price Index (Ex Food & Energy)",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "inflation",
    },
    "T5YIE": {
        "name": "5-Year Breakeven Inflation Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "inflation",
    },
    "T10YIE": {
        "name": "10-Year Breakeven Inflation Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "inflation",
    },
    "PPIFIS": {
        "name": "PPI: Final Demand",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "inflation",
    },
    # --- Employment ---
    "PAYEMS": {
        "name": "Total Nonfarm Payrolls",
        "unit": "thousands",
        "metric": "count",
        "frequency": "monthly",
        "category": "employment",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "monthly",
        "category": "employment",
    },
    "ICSA": {
        "name": "Initial Jobless Claims",
        "unit": "claims",
        "metric": "count",
        "frequency": "weekly",
        "category": "employment",
    },
    "CCSA": {
        "name": "Continued Claims",
        "unit": "claims",
        "metric": "count",
        "frequency": "weekly",
        "category": "employment",
    },
    "JTSJOL": {
        "name": "JOLTS Job Openings",
        "unit": "thousands",
        "metric": "count",
        "frequency": "monthly",
        "category": "employment",
    },
    "CES0500000003": {
        "name": "Average Hourly Earnings, Total Private",
        "unit": "dollars",
        "metric": "value",
        "frequency": "monthly",
        "category": "employment",
    },
    "CIVPART": {
        "name": "Labor Force Participation Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "monthly",
        "category": "employment",
    },
    "U6RATE": {
        "name": "U-6 Unemployment Rate (Broadest)",
        "unit": "percent",
        "metric": "rate",
        "frequency": "monthly",
        "category": "employment",
    },
    # --- Growth & Output ---
    "GDPC1": {
        "name": "Real Gross Domestic Product",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "quarterly",
        "category": "growth",
    },
    "INDPRO": {
        "name": "Industrial Production Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "growth",
    },
    "RSAFS": {
        "name": "Advance Retail Sales",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "growth",
    },
    "RSXFS": {
        "name": "Retail Sales Ex Food Services",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "growth",
    },
    "DGORDER": {
        "name": "Durable Goods Orders",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "growth",
    },
    "CPGDPAI": {
        "name": "Corporate Profits After Tax (w/ IVA+CCAdj)",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "quarterly",
        "category": "growth",
    },
    "TCU": {
        "name": "Capacity Utilization: Total Index",
        "unit": "percent",
        "metric": "rate",
        "frequency": "monthly",
        "category": "growth",
    },
    "CP": {
        "name": "Corporate Profits After Tax",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "quarterly",
        "category": "growth",
    },
    # --- Housing ---
    "HOUST": {
        "name": "Housing Starts",
        "unit": "thousands",
        "metric": "count",
        "frequency": "monthly",
        "category": "housing",
    },
    "PERMIT": {
        "name": "Building Permits",
        "unit": "thousands",
        "metric": "count",
        "frequency": "monthly",
        "category": "housing",
    },
    "EXHOSLUSM495S": {
        "name": "Existing Home Sales",
        "unit": "millions",
        "metric": "count",
        "frequency": "monthly",
        "category": "housing",
    },
    "CSUSHPINSA": {
        "name": "Case-Shiller National Home Price Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "housing",
    },
    "MORTGAGE30US": {
        "name": "30-Year Fixed Mortgage Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "weekly",
        "category": "housing",
    },
    # --- Consumer Sentiment ---
    "UMCSENT": {
        "name": "University of Michigan Consumer Sentiment",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "consumer",
    },
    "PI": {
        "name": "Personal Income",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "consumer",
    },
    "PSAVERT": {
        "name": "Personal Savings Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "monthly",
        "category": "consumer",
    },
    "PCE": {
        "name": "Personal Consumption Expenditures",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "consumer",
    },
    # --- Manufacturing ---
    "NEWORDER": {
        "name": "Manufacturers' New Orders: All Manufacturing",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "manufacturing",
    },
    "AMTMNO": {
        "name": "Manufacturers' New Orders: Total Manufacturing",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "manufacturing",
    },
    "AWHMAN": {
        "name": "Average Weekly Hours: Manufacturing",
        "unit": "hours",
        "metric": "value",
        "frequency": "monthly",
        "category": "manufacturing",
    },
    # --- Interest Rates ---
    "DFF": {
        "name": "Effective Federal Funds Rate (Daily)",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DGS2": {
        "name": "2-Year Treasury Constant Maturity Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DGS5": {
        "name": "5-Year Treasury Constant Maturity Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DGS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DGS30": {
        "name": "30-Year Treasury Constant Maturity Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DGS3MO": {
        "name": "3-Month Treasury Constant Maturity Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DFEDTARU": {
        "name": "Fed Funds Target Range Upper Bound",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "DFEDTARL": {
        "name": "Fed Funds Target Range Lower Bound",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    "SOFR": {
        "name": "Secured Overnight Financing Rate",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "rates",
    },
    # --- Credit Spreads ---
    "BAMLH0A0HYM2": {
        "name": "ICE BofA US High Yield OAS",
        "unit": "percent",
        "metric": "spread",
        "frequency": "daily",
        "category": "credit",
    },
    "BAMLC0A0CM": {
        "name": "ICE BofA US Investment Grade OAS",
        "unit": "percent",
        "metric": "spread",
        "frequency": "daily",
        "category": "credit",
    },
    "BAMLH0A0HYM2EY": {
        "name": "ICE BofA High Yield Effective Yield",
        "unit": "percent",
        "metric": "rate",
        "frequency": "daily",
        "category": "credit",
    },
    "DRTSCILM": {
        "name": "Banks Tightening C&I Loan Standards (Large Firms)",
        "unit": "percent",
        "metric": "rate",
        "frequency": "quarterly",
        "category": "credit",
    },
    "DRALACBS": {
        "name": "Delinquency Rate: All Loans, All Commercial Banks",
        "unit": "percent",
        "metric": "rate",
        "frequency": "quarterly",
        "category": "credit",
    },
    "TOTALSL": {
        "name": "Total Consumer Credit Outstanding",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "credit",
    },
    "TOTBKCR": {
        "name": "Bank Credit, All Commercial Banks",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "weekly",
        "category": "credit",
    },
    # --- Money Supply ---
    "M2SL": {
        "name": "M2 Money Supply",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "money",
    },
    "M2V": {
        "name": "Velocity of M2 Money Stock",
        "unit": "ratio",
        "metric": "value",
        "frequency": "quarterly",
        "category": "money",
    },
    "WALCL": {
        "name": "Federal Reserve Total Assets",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "weekly",
        "category": "money",
    },
    "RRPONTSYD": {
        "name": "Overnight Reverse Repo (ON RRP)",
        "unit": "billions_usd",
        "metric": "value",
        "frequency": "daily",
        "category": "money",
    },
    # --- Trade & Dollar ---
    "BOPGSTB": {
        "name": "Trade Balance: Goods & Services",
        "unit": "millions_usd",
        "metric": "value",
        "frequency": "monthly",
        "category": "trade",
    },
    "DTWEXBGS": {
        "name": "Nominal Broad US Dollar Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "daily",
        "category": "trade",
    },
    "DTWEXAFEGS": {
        "name": "Nominal Advanced Foreign Economies Dollar Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "daily",
        "category": "trade",
    },
    "IPG21112N": {
        "name": "Industrial Production: Mining: Crude Oil",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "trade",
    },
    # --- Leading / Composite Indicators ---
    "CFNAI": {
        "name": "Chicago Fed National Activity Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "monthly",
        "category": "leading",
    },
    "STLFSI2": {
        "name": "St. Louis Fed Financial Stress Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "weekly",
        "category": "leading",
    },
    "NFCI": {
        "name": "Chicago Fed National Financial Conditions Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "weekly",
        "category": "leading",
    },
    # ANFCI removed — tautological duplicate of NFCI
    "SAHMCURRENT": {
        "name": "Sahm Rule Recession Indicator",
        "unit": "percentage_points",
        "metric": "value",
        "frequency": "monthly",
        "category": "leading",
    },
    # --- Yield Curve ---
    "T10Y2Y": {
        "name": "10-Year Minus 2-Year Treasury Spread",
        "unit": "percent",
        "metric": "spread",
        "frequency": "daily",
        "category": "yield_curve",
    },
    "T10Y3M": {
        "name": "10-Year Minus 3-Month Treasury Spread",
        "unit": "percent",
        "metric": "spread",
        "frequency": "daily",
        "category": "yield_curve",
    },
    "T10YFF": {
        "name": "10-Year Minus Fed Funds Rate",
        "unit": "percent",
        "metric": "spread",
        "frequency": "daily",
        "category": "yield_curve",
    },
}


# =============================================================================
# Non-Daily Indicators — excluded from the daily correlation pipeline
# =============================================================================
# Monthly, quarterly, and weekly FRED indicators can't be correlated at daily
# frequency because forward-filling creates artifacts. These go through the
# separate monthly pipeline instead.

NON_DAILY_INDICATORS: set[str] = {
    series_id
    for series_id, meta in FRED_INDICATORS.items()
    if meta["frequency"] in ("monthly", "quarterly")
}

# Indicators that go through the monthly pipeline — includes weekly,
# monthly, and quarterly series. Separate from NON_DAILY_INDICATORS
# because the daily pipeline can handle weekly data (forward-fill is
# only 1-4 days) but monthly/quarterly can't be forward-filled daily.
MONTHLY_PIPELINE_INDICATORS: set[str] = {
    series_id
    for series_id, meta in FRED_INDICATORS.items()
    if meta["frequency"] in ("weekly", "monthly", "quarterly")
}
