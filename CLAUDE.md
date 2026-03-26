# Zerofin — Financial Intelligence System

## Project Overview
A personal financial intelligence system that collects market data, discovers relationships between financial entities using a knowledge graph (Neo4j), and generates plain-English briefings. Built with Python, two databases (PostgreSQL, Neo4j), DeepSeek API for automated analysis, and Claude Code for interactive exploration.

Full design docs: `C:/Users/B/Desktop/The Base/Projects/Financial Project/` (15+ Obsidian files covering architecture, data model, build plan, tracking list, RSS feeds, brainstorming, etc.)

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** uv (NOT pip or Poetry)
- **Databases:** PostgreSQL (time-series + settings), Neo4j (knowledge graph + vector search + article storage)
- **Math:** Polars (NOT Pandas)
- **Embeddings:** Voyage AI (`voyage-finance-2`)
- **AI (automated):** DeepSeek Speciale API (`deepseek-chat`) via LangChain
- **AI (interactive):** Claude Code + MCP tools
- **LLM Orchestration:** LangChain (for DeepSeek pipeline and future multi-model workflows)
- **Covariance Estimation:** scikit-learn (Ledoit-Wolf, GraphicalLasso)
- **Web Backend:** FastAPI
- **Web Frontend:** React (later phases)
- **Linting:** Ruff
- **Testing:** Pytest
- **Deployment:** Docker Desktop locally (cloud VPS later for 24/7 operation)

## Project Structure
```
zerofin/
├── src/
│   └── zerofin/           # Main Python package
│       ├── config.py      # Settings from .env via pydantic-settings
│       ├── data/          # Data collection plugins
│       │   ├── collector.py   # BaseCollector template
│       │   ├── tickers.py     # All tracked tickers + FRED indicators
│       │   ├── prices.py      # yfinance batch price collector
│       │   ├── economic.py    # FRED API economic collector
│       │   └── news.py        # RSS feed news collector
│       ├── storage/       # Database connections (postgres, neo4j)
│       ├── models/        # Pydantic data models
│       ├── analysis/      # Correlation engine, partial correlation, relationship discovery
│       │   ├── correlations.py  # Pearson pipeline (beta-removed)
│       │   ├── partial.py       # Partial correlation (precision matrix)
│       │   ├── monthly.py       # Monthly FRED pipeline
│       │   ├── filters.py       # FDR, stability, plausibility filters
│       │   └── transforms.py    # Returns, z-score, winsorize, beta removal
│       ├── ai/            # DeepSeek integration, prompts (Phase 2)
│       └── delivery/      # Briefing generation, alerts (Phase 4)
├── scripts/               # Runnable scripts (setup, seed, daily run)
├── web/                   # FastAPI + React dashboard (Phase 5)
├── tests/                 # Pytest tests
├── docker-compose.yml     # Database infrastructure (reads from .env)
├── pyproject.toml         # uv config
└── .env                   # API keys (NEVER committed)
```

## Development Commands
- `uv sync` — install all dependencies
- `uv run pytest` — run tests
- `uv run ruff check .` — lint
- `uv run ruff format .` — format code
- `uv run python scripts/daily_collect.py` — run daily pipeline
- `uv run python scripts/setup_databases.py` — initialize databases
- `uv run python scripts/seed_entities.py` — load initial entities

## Code Conventions

### Style
- Type hints on ALL function parameters and return values
- Docstrings on functions that aren't obvious — skip if the name already says it all
- Comment where it adds value — explain the "why", skip the obvious
- Aim for functions under 30 lines — split if it helps readability, but don't force it
- Descriptive names: `get_daily_prices()` not `gdp()` or `fetch()`
- No abbreviations: `article_count` not `ac` or `artCnt`
- No magic numbers — give numbers a name (`MIN_CONFIDENCE = 0.7` not just `0.7`)
- Follow PEP 8

### Imports
- Order: stdlib → third-party → local
- Use `from __future__ import annotations` in every file
- Prefer `pathlib.Path` over `os.path`
- Use `pendulum` for dates, not `datetime`
- Use `httpx` for HTTP requests, not `requests`

### Logging
- Use `logging` module, NEVER `print()`
- Set up logger: `logger = logging.getLogger(__name__)`
- Use appropriate levels: DEBUG for details, INFO for operations, ERROR for failures

### Data Validation
- Use Pydantic models for ALL external data (API responses, database reads)
- Validate data BEFORE storing — never trust raw API responses
- See Data Model doc for validation rules (price bounds, economic data ranges)

### Database Access
- Each database has its own module in `storage/`
- Neo4j: use Cypher queries, batch operations (UNWIND), always index queried properties
- Neo4j vector search: articles stored as nodes with Voyage AI embeddings, connected via MENTIONED_IN edges
- PostgreSQL: use parameterized queries, never string concatenation
- NEVER hardcode connection strings — load from config.py

### Error Handling
- Every external API call gets try/except with logging
- Retry once on failure, then log and continue
- Worker pipeline: collect all results, verify, then store in one batch
- If a pipeline step fails, don't store partial results

## Architecture Rules
- Worker and web app are SEPARATE processes sharing databases
- Worker NEVER talks to FastAPI. FastAPI NEVER talks to worker. Both talk to databases.
- Settings stored in PostgreSQL. Worker reads settings from DB.
- All AI analysis goes through DeepSeek API, NOT Claude API (cost)
- Data collection plugins follow a standard interface (see `data/collector.py`)

## Entity and Relationship Schema
- 12 entity types and 16 relationship types based on FinDKG ontology (+ DEPENDS_ON)
- Full schema details in Obsidian docs: `10 - Entity and Relationship Types.md`
- All relationships have: confidence, times_tested, times_confirmed, valid_from, valid_until, source, status

## Key Rules
- NEVER commit .env or any file with API keys
- NEVER use print() — always use logger
- NEVER use Pandas — use Polars
- NEVER store data without validation (always route through Pydantic models)
- NEVER insert into Neo4j one record at a time — always batch with UNWIND
- NEVER hardcode thresholds in code — put them in config.py so they're tunable
- NEVER edit files that another agent might be working on simultaneously
- Ruff runs automatically via pre-commit hook — don't run it manually
- ALWAYS write tests for new functionality
- ALWAYS use uv commands, never raw pip
- ALWAYS verify data quality after changes (run engine, check Neo4j, audit results)
- ALWAYS suggest research when uncertain about domain-specific decisions
- ALWAYS clear the market_data table before re-running backfills (backfill creates duplicates, not upserts)
- ALWAYS flag new library installations before running them — explain what's being added and why
- ALWAYS verify script paths exist before citing them — don't guess filenames
- NEVER use Python for-loops for numeric computation when numpy can vectorize it
- NEVER stack more statistical filters when the problem is the estimator — fix the math upstream
- Prefer structural sparsity (glasso) over arbitrary thresholding (Ledoit-Wolf + cutoff)
- DeepSeek pipeline MUST use LangChain for orchestration — decided 2026-03-25

## Working Style
- Explain things in plain English, not jargon
- Be concise and direct — avoid walls of text
- Push back when something is wrong — don't just agree
- Use visual diagrams for architecture discussions
- Comment where it adds value — explain the "why", not the obvious
- Ask before making big architectural decisions
- Use parallel agents when building multiple files
- When generating code, prioritize readability over cleverness
- Ask before implementing — we discuss and agree before you write code
- Don't be pushy — if we're exploring or brainstorming, go with it
- Always ask before committing or pushing — user reviews code first
- Don't take silent actions — explain what you're doing before creating files or running commands
- Build one file at a time, then walk through what it does — don't dump 9 files at once
- When the user asks a question, STOP and answer it fully — don't rush to the next task
- Let the user decide when to move on — they'll say when they're ready
- Give complete answers the first time — don't drip-feed pieces that require follow-ups
- This is a learning project — the user wants to understand WHY, not just see working code
- When uncertain about domain-specific decisions (thresholds, algorithms, financial math), don't guess — suggest research. Research has been the most valuable part of this project. Every major design decision should be backed by research, not assumptions
- Suggest specific research prompts — the user runs them in separate terminals
- Research prompts must start with "Research this topic thoroughly — search the web" and include date range if relevant (e.g., "2025-2026 only")
- When suggesting prompts for separate terminals (code reviews, research, fixes), preface with: `claude --dangerously-skip-permissions --model "claude-opus-4-6[1m]"` — we use the `[1m]` suffix for the 1M context window, otherwise it defaults to 200k
- Chain of verification: build → test → research if results look wrong → fix → test again
- The user runs multiple agents in parallel — don't edit files another agent might be editing
- When delegating to other agents, specify what's safe to delegate vs what needs our context
- Run correlation audits via agents after every threshold or filter change
- Don't assume the user remembers technical terms — re-explain simply when asked
- Don't say "you'll probably never need this" — the user wants to understand everything
- All documents save to Obsidian (never project docs/ folder):
  - Research: `C:/Users/B/Desktop/The Base/Projects/Financial Project/Research/`
  - Quality audits: `C:/Users/B/Desktop/The Base/Projects/Financial Project/Quality Control/`
  - Code reviews: `C:/Users/B/Desktop/The Base/Projects/Financial Project/Code Reviews/`
  - Prompts: `C:/Users/B/Desktop/The Base/Projects/Financial Project/Prompts/`