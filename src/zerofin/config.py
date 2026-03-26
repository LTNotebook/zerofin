"""
Application settings — loads everything from the .env file.

All database credentials, API keys, and configuration live here.
Other modules import `settings` from this file instead of reading
.env themselves. One source of truth for all config.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

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

    # --- Correlation Engine (long-term relationship discovery) ---

    # Rolling windows for relationship storage (medium + long only)
    # 21-day is reserved for future signal/alert detection, not stored as candidates
    CORRELATION_WINDOWS: list[int] = [63, 252]
    # Minimum overlapping data points as a fraction of the window size.
    # 0.75 means we need at least 75% of the window's days to have data.
    # For a 252-day window that's ~189 days. For 63 days that's ~47.
    CORRELATION_MIN_OBSERVATIONS_RATIO: float = 0.75

    # Tiered strength thresholds
    CORRELATION_TIER_STORE: float = 0.4       # Minimum for daily windows (63/252 day)
    CORRELATION_TIER_STORE_MONTHLY: float = 0.4  # Stability filter + Gate 2 handle quality
    CORRELATION_TIER_ACTIONABLE: float = 0.5  # Moderate, use in analysis
    CORRELATION_TIER_STRONG: float = 0.7      # High-confidence

    # FDR alpha — 0.10 for exploratory discovery (we validate later via review queue)
    CORRELATION_FDR_ALPHA: float = 0.10

    # Lag days to test — stock-to-stock vs macro-to-stock
    CORRELATION_LAGS_EQUITY: list[int] = [0, 1, 2, 3, 5, 10]
    CORRELATION_LAGS_MACRO: list[int] = [0, 1, 2, 3, 5, 10, 21, 63]

    # Winsorize: cap extreme daily moves at these percentiles before correlating
    CORRELATION_WINSORIZE_LOW: float = 0.01   # 1st percentile
    CORRELATION_WINSORIZE_HIGH: float = 0.99  # 99th percentile

    # Stability filter: require correlation in both halves of the window
    CORRELATION_STABILITY_FILTER: bool = True

    # Run Spearman alongside Pearson as a sanity check
    CORRELATION_SPEARMAN_CHECK: bool = True

    # Magnitude stability ratio for monthly filter — the weakest half-window
    # correlation must be at least this fraction of the full-window strength.
    # 0.4 = weakest half must be >= 40% of full strength.
    MAGNITUDE_STABILITY_RATIO: float = 0.4

    # Starting confidence for newly discovered correlations.
    # Starts neutral at 0.5 — validation raises or lowers it.
    CORRELATION_INITIAL_CONFIDENCE: float = 0.5

    # Partial correlation — tiered thresholds
    # Tier 1 (>= active): goes straight into graph as candidate
    # Tier 2 (pending floor to active): stored as pending_verification
    #   for DeepSeek to confirm later
    # Below pending floor: not stored
    PARTIAL_CORRELATION_TIER1: float = 0.15
    PARTIAL_CORRELATION_TIER2_FLOOR: float = 0.10

    # EBIC gamma — controls sparsity aggressiveness for Graphical Lasso.
    # 0.5 = standard for very high-dimensional data (p >> n).
    # 0.1 = appropriate when p/n < 1 and we accept ~15-20% noise
    # that downstream AI verification handles.
    EBIC_GAMMA: float = 0.1

    # Number of alpha values to test in the EBIC search grid.
    # 25 is enough to find the EBIC minimum on a smooth curve.
    EBIC_N_ALPHAS: int = 25

    # Graphical Lasso solver settings
    GLASSO_MAX_ITER: int = 200
    GLASSO_CONVERGENCE_TOL: float = 1e-4
    GLASSO_ALPHA_RANGE_RATIO: float = 0.01

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Don't crash if .env has extra variables
    }

    @model_validator(mode="after")
    def _warn_empty_db_passwords(self) -> Settings:
        """Log a warning if database passwords are empty."""
        if not self.POSTGRES_PASSWORD:
            logger.warning(
                "POSTGRES_PASSWORD is empty — set it in .env for production"
            )
        if not self.NEO4J_PASSWORD:
            logger.warning(
                "NEO4J_PASSWORD is empty — set it in .env for production"
            )
        return self

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
