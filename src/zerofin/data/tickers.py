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
    # Defense / Aerospace
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
    # Retail / Consumer
    "WMT",
    "COST",
    "HD",
    "NKE",
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
    # Signal stocks (unusual insider activity or thematic bets)
    "KKR",
    "TTD",
    "COIN",
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
]

# Treasury yields and fixed-income ETFs
BONDS_YIELDS: list[str] = [
    # Yields (index-style tickers)
    "^TNX",  # 10-Year Treasury Yield
    "^TYX",  # 30-Year Treasury Yield
    "^FVX",  # 5-Year Treasury Yield
    "^IRX",  # 13-Week Treasury Bill Yield
    # Bond ETFs
    "TLT",  # iShares 20+ Year Treasury
    "SHY",  # iShares 1-3 Year Treasury
    "IEF",  # iShares 7-10 Year Treasury
    "HYG",  # iShares High Yield Corporate
    "LQD",  # iShares Investment Grade Corporate
    "AGG",  # iShares Core US Aggregate Bond
    "TIP",  # iShares TIPS (inflation-protected)
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

# Major cryptocurrencies
CRYPTO: list[str] = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
]

# All yfinance tickers combined (deduplicated, order preserved)
ALL_TICKERS: list[str] = list(
    dict.fromkeys(
        US_INDICES + SECTOR_ETFS + KEY_STOCKS + COMMODITIES + BONDS_YIELDS + INTERNATIONAL + CRYPTO
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

# Corporate bond option-adjusted spreads — credit stress gauges
FRED_CREDIT: list[str] = [
    "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS
    "BAMLC0A0CM",  # ICE BofA US Investment Grade OAS
    "BAMLH0A0HYM2EY",  # ICE BofA High Yield Effective Yield
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
]

# Composite activity, financial stress, and recession indicators
FRED_LEADING: list[str] = [
    "CFNAI",  # Chicago Fed National Activity Index (85 indicators)
    "STLFSI2",  # St. Louis Fed Financial Stress Index
    "NFCI",  # Chicago Fed National Financial Conditions Index
    "ANFCI",  # Adjusted NFCI (purer financial signal)
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
    "ANFCI": {
        "name": "Adjusted National Financial Conditions Index",
        "unit": "index_points",
        "metric": "index",
        "frequency": "weekly",
        "category": "leading",
    },
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
