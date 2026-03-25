"""
Monthly FRED correlation pipeline — separate from the daily correlation engine.

Monthly/quarterly FRED indicators can't be correlated at daily frequency
because forward-filling creates 20 zeros and 1 jump per month. Instead,
we downsample daily stock prices to monthly returns and correlate directly
against monthly indicator changes using Spearman rank correlation.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import pendulum
import polars as pl
from scipy import stats

from zerofin.analysis.correlations import (
    _build_candidates,
    _clear_old_candidates,
    _store_candidates_batch,
)
from zerofin.analysis.filters import _apply_fdr_correction, apply_monthly_stability_filter
from zerofin.config import settings
from zerofin.data.tickers import NON_DAILY_INDICATORS
from zerofin.models.correlations import CorrelationRunSummary
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


# Window tag for monthly correlations stored in Neo4j.
# Used consistently across build, clear, and store operations.
MONTHLY_WINDOW_DAYS = 30

# Minimum months of overlapping data to trust a correlation
MIN_MONTHLY_OBSERVATIONS = 10


# ─── Main entry point ────────────────────────────────────────────────


def run_monthly_correlation_pipeline(
    db: PostgresStorage,
    graph: GraphStorage,
) -> CorrelationRunSummary:
    """Correlate monthly FRED indicators with monthly stock returns.

    The daily pipeline can't handle monthly data because forward-filling
    monthly values to daily creates 20 zeros and 1 jump per month.
    Instead, we downsample daily stock prices to monthly returns and
    correlate directly against monthly indicator changes.

    Uses Spearman (rank correlation) because monthly data has heavy
    tails and small sample sizes where Pearson is unreliable.
    """
    start_time = time.monotonic()
    window_end = pendulum.now("UTC")
    window_start = window_end.subtract(years=3)

    logger.info(
        "Starting monthly FRED pipeline (%s to %s)",
        window_start.to_date_string(),
        window_end.to_date_string(),
    )

    # Fetch all data
    raw_rows = db.get_all_market_data_range(
        start=window_start, end=window_end
    )
    if not raw_rows:
        logger.warning("No data for monthly pipeline")
        return _empty_summary(0, window_end, start_time)

    # Separate asset prices from indicator values
    asset_rows = []
    indicator_rows = []
    for row in raw_rows:
        if row["entity_type"] == "asset":
            asset_rows.append(row)
        elif row["entity_id"] in NON_DAILY_INDICATORS:
            indicator_rows.append(row)

    if not asset_rows or not indicator_rows:
        logger.warning("Missing asset or indicator data for monthly pipeline")
        return _empty_summary(0, window_end, start_time)

    # Build monthly stock returns
    monthly_returns = _build_monthly_returns(asset_rows)
    logger.info(
        "Built monthly returns: %d assets, %d months",
        len(monthly_returns.columns) - 1,
        len(monthly_returns),
    )

    # Build monthly indicator changes
    monthly_indicators = _build_monthly_indicator_changes(indicator_rows)
    logger.info(
        "Built monthly indicator changes: %d indicators, %d months",
        len(monthly_indicators.columns) - 1,
        len(monthly_indicators),
    )

    # Correlate each indicator against each asset using Spearman
    joined = monthly_returns.join(monthly_indicators, on="month", how="inner")
    results = _correlate_monthly(
        monthly_returns, monthly_indicators, joined
    )
    logger.info(
        "Found %d monthly correlations above threshold", len(results)
    )

    if not results:
        total = (len(monthly_returns.columns) - 1) * (
            len(monthly_indicators.columns) - 1
        )
        return _empty_summary(total, window_end, start_time)

    # FDR correction
    surviving = _apply_fdr_correction(results)
    logger.info(
        "%d monthly correlations survived FDR", len(surviving)
    )

    # Monthly stability filter (bootstrap, sign, tercile, magnitude)
    surviving = apply_monthly_stability_filter(surviving, joined)
    stable_count = sum(
        1 for r in surviving if r.get("stability_status") == "stable"
    )
    flagged_count = len(surviving) - stable_count
    logger.info(
        "%d stable, %d flagged as under_review",
        stable_count, flagged_count,
    )

    # Only store stable ones — flagged get lower confidence
    to_store = []
    for r in surviving:
        if r.get("stability_status") == "under_review":
            # Store with reduced confidence by marking in the result
            r["confidence_penalty"] = 0.5
        to_store.append(r)

    # Build candidates (window_days tags these as monthly frequency)
    candidates = _build_candidates(to_store, MONTHLY_WINDOW_DAYS, window_end)
    logger.info("Built %d monthly candidates", len(candidates))

    cleared = _clear_old_candidates(graph, MONTHLY_WINDOW_DAYS)
    logger.info("Cleared %d old monthly candidates", cleared)

    stored = _store_candidates_batch(graph, candidates)
    logger.info("Stored %d monthly candidates in Neo4j", stored)

    duration = time.monotonic() - start_time
    total = (len(monthly_returns.columns) - 1) * (
        len(monthly_indicators.columns) - 1
    )

    summary = CorrelationRunSummary(
        total_pairs_tested=total,
        pairs_above_threshold=len(results),
        pairs_surviving_fdr=len(surviving),
        relationships_stored=stored,
        window_days=MONTHLY_WINDOW_DAYS,
        window_end=window_end,
        duration_seconds=round(duration, 2),
    )

    logger.info(
        "Monthly pipeline complete in %.1fs: %d pairs tested, %d stored",
        duration, total, stored,
    )
    return summary


# ─── Helpers ─────────────────────────────────────────────────────────


def _empty_summary(
    total: int,
    window_end: pendulum.DateTime,
    start_time: float,
) -> CorrelationRunSummary:
    """Return an empty summary for early exits."""
    return CorrelationRunSummary(
        total_pairs_tested=total,
        pairs_above_threshold=0,
        pairs_surviving_fdr=0,
        relationships_stored=0,
        window_days=MONTHLY_WINDOW_DAYS,
        window_end=window_end,
        duration_seconds=time.monotonic() - start_time,
    )


def _build_monthly_returns(asset_rows: list[dict]) -> pl.DataFrame:
    """Downsample daily asset prices to monthly returns.

    Takes the last price in each month and computes the return:
    (this month's last price / last month's last price) - 1
    """
    records = []
    for row in asset_rows:
        ts = row["timestamp"]
        records.append({
            "date": ts.date() if hasattr(ts, "date") else ts,
            "entity_key": f"asset:{row['entity_id']}",
            "value": float(row["value"]),
        })

    df = pl.DataFrame(records)

    # Take last price per entity per month
    df = df.with_columns(
        pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month")
    )
    monthly = (
        df.group_by(["month", "entity_key"])
        .agg(pl.col("value").last())
        .sort(["entity_key", "month"])
    )

    # Pivot to wide: months as rows, entities as columns
    wide = monthly.pivot(
        on="entity_key", index="month", values="value"
    ).sort("month")

    # Compute monthly returns: (this month / last month) - 1
    entity_cols = [c for c in wide.columns if c != "month"]
    return_cols = {"month": wide["month"]}
    for col in entity_cols:
        shifted = wide[col].shift(1)
        returns = (wide[col] - shifted) / shifted.replace(0, None)
        return_cols[col] = returns

    result = pl.DataFrame(return_cols)
    return result.slice(1)  # Drop first row (NaN from shift)


def _build_monthly_indicator_changes(
    indicator_rows: list[dict],
) -> pl.DataFrame:
    """Build monthly first-differences for FRED indicators.

    For each indicator, takes the value at each month and computes
    the change from the prior month. This captures whether the
    indicator is getting better or worse.
    """
    records = []
    for row in indicator_rows:
        ts = row["timestamp"]
        records.append({
            "date": ts.date() if hasattr(ts, "date") else ts,
            "entity_key": f"indicator:{row['entity_id']}",
            "value": float(row["value"]),
        })

    df = pl.DataFrame(records)

    # Take last value per indicator per month
    df = df.with_columns(
        pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month")
    )
    monthly = (
        df.group_by(["month", "entity_key"])
        .agg(pl.col("value").last())
        .sort(["entity_key", "month"])
    )

    # Pivot to wide
    wide = monthly.pivot(
        on="entity_key", index="month", values="value"
    ).sort("month")

    # First differences: this month - last month
    entity_cols = [c for c in wide.columns if c != "month"]
    diff_cols = {"month": wide["month"]}
    for col in entity_cols:
        diff_cols[col] = wide[col].diff()

    result = pl.DataFrame(diff_cols)
    return result.slice(1)


def _correlate_monthly(
    returns_df: pl.DataFrame,
    indicators_df: pl.DataFrame,
    joined: pl.DataFrame,
) -> list[dict]:
    """Correlate monthly stock returns against monthly indicator changes.

    Uses Spearman rank correlation — more robust than Pearson for
    small sample sizes and non-normal distributions (which monthly
    data always has).
    """
    min_strength = settings.CORRELATION_TIER_STORE_MONTHLY

    if len(joined) < MIN_MONTHLY_OBSERVATIONS:
        logger.warning(
            "Only %d overlapping months (need %d)",
            len(joined), MIN_MONTHLY_OBSERVATIONS,
        )
        return []

    asset_cols = [c for c in returns_df.columns if c != "month"]
    indicator_cols = [c for c in indicators_df.columns if c != "month"]

    results = []

    for asset_col in asset_cols:
        for ind_col in indicator_cols:
            pair = joined.select([asset_col, ind_col]).drop_nulls()

            if pair.height < MIN_MONTHLY_OBSERVATIONS:
                continue

            a = pair[asset_col].to_numpy()
            b = pair[ind_col].to_numpy()

            if np.std(a) < 1e-10 or np.std(b) < 1e-10:  # near-zero variance
                continue

            # Spearman rank correlation
            r, p = stats.spearmanr(a, b)

            if math.isnan(r) or abs(r) < min_strength:
                continue

            results.append({
                "entity_a": asset_col,
                "entity_b": ind_col,
                "lag_days": 0,
                "pearson_r": float(r),  # Actually Spearman, but same field
                "pearson_p": float(p),
                "observation_count": pair.height,
                "method": "spearman",
            })

    return results
