# Zerofin — Financial Intelligence System

## Project Overview
A personal financial intelligence system that collects market data, discovers relationships between financial entities using a knowledge graph (Neo4j), and generates plain-English briefings. Built with Python, two databases (PostgreSQL, Neo4j), DeepSeek API for automated analysis, and Claude Code for interactive exploration.

Full design docs: `C:/Users/B/Desktop/The Base/Financial Project/` (11 Obsidian files covering architecture, data model, build plan, etc.)

## Tech Stack
- **Language:** Python 3.11+
- **Package Manager:** uv (NOT pip or Poetry)
- **Databases:** PostgreSQL (time-series + settings), Neo4j (knowledge graph + vector search + article storage)
- **Math:** Polars (NOT Pandas)
- **Embeddings:** Voyage AI (`voyage-finance-2`)
- **AI (automated):** DeepSeek Speciale API (`deepseek-chat`)
- **AI (interactive):** Claude Code + MCP tools
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
│       ├── storage/       # Database connections (postgres, neo4j)
│       ├── models/        # Pydantic data models
│       ├── analysis/      # Correlation engine, relationship discovery
│       ├── ai/            # DeepSeek integration, prompts
│       └── delivery/      # Briefing generation, alerts
├── scripts/               # Runnable scripts (setup, seed, daily run)
├── web/                   # FastAPI + React dashboard (later phases)
├── tests/                 # Pytest tests
├── docker-compose.yml     # Database infrastructure
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
- NEVER store data without validation
- NEVER insert into Neo4j one record at a time — always batch with UNWIND
- Ruff runs automatically via pre-commit hook — don't run it manually
- ALWAYS write tests for new functionality
- ALWAYS use uv commands, never raw pip

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