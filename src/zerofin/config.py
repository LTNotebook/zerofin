"""
Application settings — loads everything from the .env file.

All database credentials, API keys, and configuration live here.
Other modules import `settings` from this file instead of reading
.env themselves. One source of truth for all config.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

# Walk up from this file to find the project root where .env lives
# src/zerofin/config.py -> src/zerofin -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """All app settings, loaded automatically from .env file."""

    # --- PostgreSQL ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "zerofin"
    POSTGRES_USER: str = "zerofin"
    POSTGRES_PASSWORD: str = ""

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # --- API Keys ---
    FRED_API_KEY: str = ""
    NEWSAPI_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Don't crash if .env has extra variables
    }

    def postgres_connection_params(self) -> dict[str, str | int]:
        """Return Postgres connection parameters as separate keys.

        Using individual parameters instead of a URL string means the password
        is never concatenated into a readable value that could appear in logs,
        error messages, or debug output.

        psycopg.connect() accepts these keyword arguments directly:
            psycopg.connect(**settings.postgres_connection_params())
        """
        return {
            "host": self.POSTGRES_HOST,
            "port": self.POSTGRES_PORT,
            "dbname": self.POSTGRES_DB,
            "user": self.POSTGRES_USER,
            "password": self.POSTGRES_PASSWORD,
        }


# Create one global instance — import this everywhere
# Usage: from zerofin.config import settings
settings = Settings()
