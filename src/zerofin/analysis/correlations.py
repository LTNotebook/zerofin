"""
Correlation engine — discovers long-term statistical relationships.

This is the math brain of Phase 2. It pulls historical price and economic
data from PostgreSQL, calculates correlations between every pair of entities
at multiple time lags, filters out the noise, and stores the surviving
candidates in Neo4j as CORRELATES_WITH relationships.

The pipeline:
    1. Fetch all market data for the window (one bulk query)
    2. Build a wide Polars DataFrame (dates as rows, entities as columns)
    3. Transform: natural returns per asset type (log returns, first diff, pct change)
    4. Z-score all returns to a common scale
    5. Winsorize to cap extreme moves (now in standard deviation units)
    6. Remove market + sector beta (joint OLS residualization)
    7. Calculate pairwise correlations (Pearson + Spearman) via numpy matrix
    8. Apply FDR correction to filter false positives
    9. Apply stability filter (correlation must hold in both halves of window)
    10. Validate through CorrelationCandidate model
    11. Batch-store candidates in Neo4j

Run standalone:
    python scripts/run_correlations.py

Or as part of the weekly pipeline (added later).
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import pendulum
import polars as pl
from scipy import stats

from zerofin.config import settings
from zerofin.data.tickers import REDUNDANCY_LOOKUP, STOCK_SECTOR_MAP
from zerofin.models.correlations import CorrelationCandidate, CorrelationRunSummary
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


# ─── Main entry point ────────────────────────────────────────────────


def run_correlation_pipeline(
    db: PostgresStorage,
    graph: GraphStorage,
    *,
    window_days: int | None = None,
) -> CorrelationRunSummary:
    """Run the full correlation discovery pipeline for a single window size.

    This is the top-level function. Call it with open database connections
    and it does everything: fetch, transform, correlate, filter, store.

    Args:
        db: Open PostgresStorage connection.
        graph: Open GraphStorage connection.
        window_days: Override the window size (defaults to each window
                     in settings.CORRELATION_WINDOWS).

    Returns:
        Summary of what was found and stored.
    """
    start_time = time.monotonic()

    # Use the provided window or default to the longest one
    window = window_days or max(settings.CORRELATION_WINDOWS)
    window_end = pendulum.now("UTC")
    window_start = window_end.subtract(days=window)

    logger.info(
        "Starting correlation pipeline: %d-day window (%s to %s)",
        window,
        window_start.to_date_string(),
        window_end.to_date_string(),
    )

    # Step 1: Fetch all data in one bulk query
    raw_rows = db.get_all_market_data_range(start=window_start, end=window_end)
    if not raw_rows:
        logger.warning("No data found in the date range — nothing to correlate")
        return CorrelationRunSummary(
            total_pairs_tested=0,
            pairs_above_threshold=0,
            pairs_surviving_fdr=0,
            relationships_stored=0,
            window_days=window,
            window_end=window_end,
            duration_seconds=time.monotonic() - start_time,
        )

    # Calculate min observations based on window size
    min_obs = max(15, int(window * settings.CORRELATION_MIN_OBSERVATIONS_RATIO))
    logger.info("Min observations for %d-day window: %d", window, min_obs)

    # Step 2: Build wide DataFrame
    wide_df = _build_wide_dataframe(raw_rows, min_obs=min_obs)
    entity_count = len(wide_df.columns) - 1  # minus the date column
    logger.info("Built wide DataFrame: %d entities, %d days", entity_count, len(wide_df))

    if entity_count < 2:
        logger.warning("Need at least 2 entities to correlate — only found %d", entity_count)
        return CorrelationRunSummary(
            total_pairs_tested=0,
            pairs_above_threshold=0,
            pairs_surviving_fdr=0,
            relationships_stored=0,
            window_days=window,
            window_end=window_end,
            duration_seconds=time.monotonic() - start_time,
        )

    # Step 3: Transform — natural returns per asset type
    returns_df = _compute_transforms(wide_df)
    logger.info("Computed transforms for %d entities", len(returns_df.columns) - 1)

    # Step 4: Z-score all returns to a common scale
    returns_df = _z_score_all(returns_df)
    logger.info("Z-scored all returns to common scale")

    # Step 5: Winsorize in standard deviation units (clip at ±3σ)
    returns_df = _winsorize(returns_df)

    # Step 6: Remove market + sector beta from asset returns
    returns_df = _remove_market_sector_beta(returns_df)
    logger.info("Removed market + sector beta from asset returns")

    # Step 6: Calculate pairwise correlations at each lag
    entity_columns = [c for c in returns_df.columns if c != "date"]
    raw_results = _compute_pairwise_correlations(returns_df, entity_columns, min_obs=min_obs)
    logger.info(
        "Computed %d raw correlations above min strength threshold",
        len(raw_results),
    )

    if not raw_results:
        logger.info("No correlations above minimum strength — done")
        total_pairs = len(entity_columns) * (len(entity_columns) - 1) // 2
        return CorrelationRunSummary(
            total_pairs_tested=total_pairs,
            pairs_above_threshold=0,
            pairs_surviving_fdr=0,
            relationships_stored=0,
            window_days=window,
            window_end=window_end,
            duration_seconds=time.monotonic() - start_time,
        )

    # Step 6: FDR correction
    surviving = _apply_fdr_correction(raw_results)
    logger.info("%d correlations survived FDR correction", len(surviving))

    # Step 7: Stability filter
    if settings.CORRELATION_STABILITY_FILTER and window >= 42:
        surviving = _apply_stability_filter(returns_df, surviving)
        logger.info("%d correlations survived stability filter", len(surviving))

    # Step 8: Build validated candidates
    candidates = _build_candidates(surviving, window, window_end)
    logger.info("Built %d validated candidates", len(candidates))

    # Step 9: Clear old statistical candidates and store new ones
    cleared = _clear_old_candidates(graph, window)
    logger.info("Cleared %d old statistical candidates", cleared)

    stored = _store_candidates_batch(graph, candidates)
    logger.info("Stored %d new candidates in Neo4j", stored)

    duration = time.monotonic() - start_time
    total_pairs = len(entity_columns) * (len(entity_columns) - 1) // 2

    summary = CorrelationRunSummary(
        total_pairs_tested=total_pairs,
        pairs_above_threshold=len(raw_results),
        pairs_surviving_fdr=len(surviving),
        relationships_stored=stored,
        window_days=window,
        window_end=window_end,
        duration_seconds=round(duration, 2),
    )

    logger.info(
        "Correlation pipeline complete in %.1fs: %d pairs tested, %d stored",
        duration,
        total_pairs,
        stored,
    )

    return summary


# ─── Step 2: Build wide DataFrame ────────────────────────────────────


def _build_wide_dataframe(raw_rows: list[dict], *, min_obs: int = 200) -> pl.DataFrame:
    """Convert raw database rows into a wide Polars DataFrame.

    Input: list of dicts with entity_type, entity_id, metric, value, timestamp
    Output: DataFrame with 'date' column + one column per entity (e.g. 'asset:NVDA')

    Forward-fills missing days so daily and monthly data align.
    Drops entities with fewer than MIN_OBSERVATIONS data points.
    """
    # Build a long-format DataFrame from the raw rows
    records = []
    for row in raw_rows:
        # Create a unique key for each entity: "asset:NVDA" or "indicator:DGS10"
        entity_key = f"{row['entity_type']}:{row['entity_id']}"
        records.append({
            "date": (
                row["timestamp"].date()
                if hasattr(row["timestamp"], "date")
                else row["timestamp"]
            ),
            "entity_key": entity_key,
            "value": float(row["value"]),
        })

    long_df = pl.DataFrame(records)

    # If multiple values for same entity on same day, take the last one
    long_df = long_df.group_by(["date", "entity_key"]).agg(pl.col("value").last())

    # Pivot to wide: dates as rows, one column per entity
    wide_df = long_df.pivot(on="entity_key", index="date", values="value")

    # Sort by date so time series is in order
    wide_df = wide_df.sort("date")

    # Forward-fill to handle gaps (weekends, holidays, monthly data)
    entity_cols = [c for c in wide_df.columns if c != "date"]
    wide_df = wide_df.with_columns([pl.col(c).forward_fill() for c in entity_cols])

    # Drop entities with too few observations
    keep_cols = ["date"]
    for col in entity_cols:
        non_null = wide_df[col].drop_nulls().len()
        if non_null >= min_obs:
            keep_cols.append(col)
        else:
            logger.debug("Dropping %s — only %d observations (need %d)", col, non_null, min_obs)

    wide_df = wide_df.select(keep_cols)

    return wide_df


# ─── Step 3: Transform data ──────────────────────────────────────────


def _compute_transforms(wide_df: pl.DataFrame) -> pl.DataFrame:
    """Stage 1: Transform each series into its natural return.

    This is the first half of two-stage normalization. Each asset type
    gets the transformation that makes economic sense for it:

    - Stocks, ETFs, commodity futures, crypto: log returns
    - VIX (volatility index): percentage change (not log returns)
    - Interest rates and spreads: first differences (absolute change)
    - Economic levels (CPI, industrial production): first differences
    - Survey/count indicators: first differences

    After this step, every series is stationary (no trend) but still
    on different scales. Stage 2 (z-scoring) fixes the scale issue.
    """
    # Tickers that need percentage change instead of log returns
    # VIX can spike but never goes to zero, so pct change works
    VOLATILITY_TICKERS = {"^VIX"}

    result_cols = {"date": wide_df["date"]}

    for col in wide_df.columns:
        if col == "date":
            continue

        series = wide_df[col]
        entity_type, entity_id = col.split(":", 1)

        if entity_type == "asset":
            if entity_id in VOLATILITY_TICKERS:
                # Percentage change for volatility indices
                shifted = series.shift(1)
                transformed = (series - shifted) / shifted.replace(0, None)
            else:
                # Log returns for everything else: stocks, ETFs, commodities, crypto
                transformed = _log_returns(series)

        elif entity_type == "indicator":
            # All indicators get first differences — this captures
            # the CHANGE in the indicator, which is what moves markets.
            # Rates change by basis points, CPI changes month-to-month, etc.
            transformed = series.diff()

        else:
            transformed = _log_returns(series)

        result_cols[col] = transformed

    result = pl.DataFrame(result_cols)
    result = result.slice(1)  # Drop first row — always NaN after transforms

    return result


def _log_returns(series: pl.Series) -> pl.Series:
    """Calculate log returns: ln(today / yesterday).

    Returns are NaN where the previous value was zero or null.
    """
    shifted = series.shift(1)
    ratio = series / shifted.replace(0, None)
    return ratio.log()


# ─── Step 4: Z-score all returns ─────────────────────────────────────


def _z_score_all(df: pl.DataFrame) -> pl.DataFrame:
    """Stage 2: Z-score every return series to a common scale.

    After Stage 1, a stock log return of 0.012 and a rate first difference
    of 0.0006 are both "returns" but on wildly different scales. Z-scoring
    converts both to "how many standard deviations was this move?"

    A 2σ move in rates is now directly comparable to a 2σ move in stocks.
    This is what makes cross-asset correlation meaningful.

    Uses full-window z-score (not rolling) since we're computing correlation
    over the same window anyway.
    """
    result_cols = {"date": df["date"]}

    for col in df.columns:
        if col == "date":
            continue

        series = df[col]
        mean = series.mean()
        std = series.std()

        if std is None or std < 1e-10:
            # Near-zero variance — can't z-score, fill with nulls
            result_cols[col] = pl.Series(col, [None] * series.len())
        else:
            result_cols[col] = ((series - mean) / std).alias(col)

    return pl.DataFrame(result_cols)


# ─── Step 4: Winsorize ───────────────────────────────────────────────


def _winsorize(df: pl.DataFrame) -> pl.DataFrame:
    """Cap extreme values after z-scoring.

    Since the data is now z-scored (mean ~0, std ~1), we clip at the
    configured percentiles. On z-scored data, the 1st/99th percentiles
    are roughly ±2.3σ, which removes extreme outlier days while keeping
    the bulk of the distribution intact.
    """
    low_pct = settings.CORRELATION_WINSORIZE_LOW
    high_pct = settings.CORRELATION_WINSORIZE_HIGH

    entity_cols = [c for c in df.columns if c != "date"]

    capped_cols = []
    for col in entity_cols:
        series = df[col].drop_nulls()
        if series.len() == 0:
            capped_cols.append(df[col])
            continue

        low_val = series.quantile(low_pct)
        high_val = series.quantile(high_pct)
        capped = df[col].clip(low_val, high_val)
        capped_cols.append(capped.alias(col))

    return df.select([pl.col("date")] + capped_cols)


# ─── Step 5: Remove market + sector beta ─────────────────────────────


def _remove_market_sector_beta(df: pl.DataFrame) -> pl.DataFrame:
    """Remove market and sector influence from asset returns using joint OLS.

    For each stock, we regress its returns against the S&P 500 (^GSPC) and
    its sector ETF simultaneously. The residuals — what's left after removing
    both influences — are what we correlate. This way, Apple and Costco
    won't look correlated just because they both follow the market.

    Joint OLS (not sequential) is used because SPY and sector ETFs are
    themselves correlated. Doing it in one step avoids order-dependence.

    Indicators are left alone — they're already macro-scale data and don't
    have market beta to remove.
    """
    market_key = "asset:^GSPC"

    if market_key not in df.columns:
        logger.warning("^GSPC not found in data — skipping beta removal")
        return df

    market = df[market_key].to_numpy().astype(float)
    result_cols = {"date": df["date"]}

    for col in df.columns:
        if col == "date":
            continue

        # Only remove beta from individual stocks, not from indices/ETFs/indicators
        entity_type, entity_id = col.split(":", 1)
        if entity_type != "asset" or entity_id not in STOCK_SECTOR_MAP:
            result_cols[col] = df[col]
            continue

        series = df[col].to_numpy().astype(float)

        # Build the factor matrix: [constant, market, sector (if available)]
        sector_etf = STOCK_SECTOR_MAP[entity_id].get("sector")
        sector_key = f"asset:{sector_etf}" if sector_etf else None

        # Find valid rows where all factors have data
        mask = ~np.isnan(series) & ~np.isnan(market)
        factors = [np.ones(mask.sum()), market[mask]]

        if sector_key and sector_key in df.columns:
            sector = df[sector_key].to_numpy().astype(float)
            mask = mask & ~np.isnan(sector)
            factors = [np.ones(mask.sum()), market[mask], sector[mask]]

        if mask.sum() < 30:
            # Not enough data for regression — keep original
            result_cols[col] = df[col]
            continue

        # Joint OLS: regress stock on [constant, market, sector]
        x_matrix = np.column_stack(factors)
        betas = np.linalg.lstsq(x_matrix, series[mask], rcond=None)[0]

        # Calculate residuals for ALL rows (not just the masked ones)
        full_factors = [np.ones(len(series)), market]
        if sector_key and sector_key in df.columns:
            sector_full = df[sector_key].to_numpy().astype(float)
            full_factors.append(sector_full)
        full_x = np.column_stack(full_factors)
        predicted = full_x @ betas
        residuals = series - predicted

        # Restore NaNs where original data was missing
        residuals[np.isnan(series)] = np.nan

        result_cols[col] = pl.Series(col, residuals)

    return pl.DataFrame(result_cols)


# ─── Step 6: Pairwise correlations ───────────────────────────────────


def _compute_pairwise_correlations(
    df: pl.DataFrame,
    entity_columns: list[str],
    *,
    min_obs: int = 200,
) -> list[dict]:
    """Calculate pairwise correlations using numpy matrix approach.

    Instead of looping through ~100,000 pairs one at a time, we build a
    matrix with all entities (including lagged versions) and compute
    every correlation in one numpy call. This is 100-300x faster because
    numpy uses optimized C/BLAS code under the hood.

    For each lag offset, we add shifted copies of the entity columns to
    the matrix. One corrcoef call covers everything. P-values are
    calculated from the correlation matrix using the t-distribution formula.

    Spearman check runs as a targeted loop only on candidates that pass
    the Pearson filter — much smaller set than checking everything.
    """
    min_strength = settings.CORRELATION_TIER_STORE
    do_spearman = settings.CORRELATION_SPEARMAN_CHECK

    # Determine all unique lags we need to test
    equity_lags = settings.CORRELATION_LAGS_EQUITY
    macro_lags = settings.CORRELATION_LAGS_MACRO
    all_lags = sorted(set(equity_lags + macro_lags))

    n_pairs = len(entity_columns) * (len(entity_columns) - 1) // 2
    logger.info("Testing %d entity pairs across %d lags", n_pairs, len(all_lags))

    # Build the mega matrix: one row per entity per lag
    # Track what each row represents: (column_name, lag_days)
    row_labels = []
    row_data = []

    for col in entity_columns:
        for lag in all_lags:
            if lag == 0:
                series = df[col].to_numpy().astype(float)
            else:
                series = df[col].shift(lag).to_numpy().astype(float)

            # Skip if too many NaNs or near-zero variance
            valid = ~np.isnan(series)
            if valid.sum() < min_obs:
                continue
            if np.nanstd(series) < 1e-10:
                continue

            row_labels.append((col, lag))
            row_data.append(series)

    if len(row_data) < 2:
        return []

    # Stack into matrix: shape (n_rows, n_days)
    matrix = np.array(row_data)

    # Find columns (days) where ALL rows have valid data
    valid_mask = ~np.isnan(matrix).any(axis=0)
    n_obs = valid_mask.sum()

    # Need a reasonable minimum — at least 30 observations for any statistical meaning
    if n_obs < 30:
        logger.warning(
            "Only %d complete observations — need at least 30", n_obs
        )
        return []

    logger.info("Matrix: %d rows x %d valid days (after NaN strip)", len(row_labels), n_obs)

    # Strip NaN columns — corrcoef requires no NaNs
    clean_matrix = matrix[:, valid_mask]

    # One call to get all correlations
    corr_matrix = np.corrcoef(clean_matrix)

    # Calculate p-values from the correlation matrix using t-distribution
    # Clip to avoid division by zero when r = exactly 1.0
    r_clipped = np.clip(corr_matrix, -0.99999, 0.99999)
    deg_freedom = n_obs - 2
    t_stat = r_clipped * np.sqrt(deg_freedom) / np.sqrt(1 - r_clipped ** 2)
    p_matrix = 2 * stats.t.sf(np.abs(t_stat), deg_freedom)

    # Build lookup: for each entity, which row indices correspond to which lag
    # Format: {entity_name: {lag: row_index}}
    entity_lag_index: dict[str, dict[int, int]] = {}
    for idx, (col, lag) in enumerate(row_labels):
        if col not in entity_lag_index:
            entity_lag_index[col] = {}
        entity_lag_index[col][lag] = idx

    # Extract results by iterating over entity pairs (not matrix rows)
    # This is much smaller: ~18,000 pairs vs ~800,000 matrix cells
    results = []
    entity_names = list(entity_lag_index.keys())

    for i_idx, col_a in enumerate(entity_names):
        type_a = col_a.split(":")[0]
        id_a = col_a.split(":", 1)[1]

        for col_b in entity_names[i_idx + 1:]:
            type_b = col_b.split(":")[0]
            id_b = col_b.split(":", 1)[1]

            # Skip redundant pairs (e.g., IEMG vs EEM — same underlying)
            group_a = REDUNDANCY_LOOKUP.get(id_a)
            group_b = REDUNDANCY_LOOKUP.get(id_b)
            if group_a and group_b and group_a == group_b:
                continue

            # Determine which lags to check for this pair
            is_macro = (type_a == "indicator") or (type_b == "indicator")
            valid_lags = macro_lags if is_macro else equity_lags

            for lag in valid_lags:
                # We need A at lag 0 and B at lag N
                row_a = entity_lag_index[col_a].get(0)
                row_b = entity_lag_index[col_b].get(lag)

                if row_a is None or row_b is None:
                    continue

                r = corr_matrix[row_a, row_b]
                p = p_matrix[row_a, row_b]

                if math.isnan(r) or abs(r) < min_strength:
                    continue

                results.append({
                    "entity_a": col_a,
                    "entity_b": col_b,
                    "lag_days": lag,
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "observation_count": int(n_obs),
                    "method": "pearson",
                })

    # Spearman sanity check — only on candidates that passed Pearson
    if do_spearman and results:
        spearman_verified = []
        for result in results:
            col_a = result["entity_a"]
            col_b = result["entity_b"]
            lag = result["lag_days"]

            series_a = df[col_a]
            series_b = df[col_b].shift(lag) if lag > 0 else df[col_b]
            pair_df = pl.DataFrame({"a": series_a, "b": series_b}).drop_nulls()

            if pair_df.height < 20:
                spearman_verified.append(result)
                continue

            a_arr = pair_df["a"].to_numpy()
            b_arr = pair_df["b"].to_numpy()

            if np.std(a_arr) < 1e-10 or np.std(b_arr) < 1e-10:
                spearman_verified.append(result)
                continue

            spearman_r, spearman_p = stats.spearmanr(a_arr, b_arr)
            result["spearman_r"] = spearman_r
            result["spearman_p"] = spearman_p

            # If Pearson and Spearman strongly disagree, skip
            if abs(result["pearson_r"]) >= min_strength and abs(spearman_r) < min_strength:
                logger.debug(
                    "Pearson/Spearman disagree for %s vs %s (lag %d): "
                    "Pearson=%.3f, Spearman=%.3f — skipping",
                    col_a, col_b, lag, result["pearson_r"], spearman_r,
                )
                continue

            spearman_verified.append(result)

        results = spearman_verified

    return results


# ─── Step 6: FDR correction ──────────────────────────────────────────


def _apply_fdr_correction(results: list[dict]) -> list[dict]:
    """Apply Benjamini-Hochberg False Discovery Rate correction.

    When you test 20,000 pairs, ~1,000 will look significant by pure chance
    at p < 0.05. FDR correction adjusts for this so we only keep the ones
    that are likely real.

    The algorithm:
      1. Sort all p-values from smallest to largest
      2. For each p-value at rank i out of m total tests:
         adjusted_p = p * m / i
      3. Keep only results where adjusted_p <= alpha
    """
    alpha = settings.CORRELATION_FDR_ALPHA

    if not results:
        return []

    # Sort by p-value (smallest first)
    sorted_results = sorted(results, key=lambda r: r["pearson_p"])
    m = len(sorted_results)

    # Benjamini-Hochberg: adjusted_p = p * m / rank
    # Walk backwards to enforce monotonicity (each adjusted_p <= the one after it)
    adjusted_p_values = [0.0] * m
    adjusted_p_values[m - 1] = sorted_results[m - 1]["pearson_p"]

    for i in range(m - 2, -1, -1):
        rank = i + 1
        raw_adjusted = sorted_results[i]["pearson_p"] * m / rank
        # Enforce monotonicity: can't be larger than the next one
        adjusted_p_values[i] = min(raw_adjusted, adjusted_p_values[i + 1])

    # Keep results where adjusted p-value is below alpha
    surviving = []
    for i, result in enumerate(sorted_results):
        result["adjusted_p"] = adjusted_p_values[i]
        if adjusted_p_values[i] <= alpha:
            surviving.append(result)

    return surviving


# ─── Step 7: Stability filter ────────────────────────────────────────


def _apply_stability_filter(
    df: pl.DataFrame,
    results: list[dict],
) -> list[dict]:
    """Split the window in half and require correlation in both halves.

    A correlation that only exists in one half of the data is probably
    a temporary fluke or regime-dependent. If it holds in both halves,
    it's more likely to be real.
    """
    min_strength = settings.CORRELATION_TIER_STORE
    midpoint = df.height // 2
    first_half = df.slice(0, midpoint)
    second_half = df.slice(midpoint)

    stable = []
    for result in results:
        col_a = result["entity_a"]
        col_b = result["entity_b"]
        lag = result["lag_days"]

        # Check first half
        r1 = _quick_correlation(first_half, col_a, col_b, lag)
        # Check second half
        r2 = _quick_correlation(second_half, col_a, col_b, lag)

        if r1 is not None and r2 is not None:
            if abs(r1) >= min_strength and abs(r2) >= min_strength:
                result["stability_score"] = min(abs(r1), abs(r2))
                stable.append(result)
            else:
                logger.debug(
                    "Stability filter removed %s vs %s (lag %d): "
                    "half1=%.3f, half2=%.3f",
                    col_a, col_b, lag, r1, r2,
                )
        else:
            logger.debug(
                "Stability filter removed %s vs %s (lag %d): "
                "not enough data in one half",
                col_a, col_b, lag,
            )

    return stable


def _quick_correlation(
    df: pl.DataFrame,
    col_a: str,
    col_b: str,
    lag: int,
) -> float | None:
    """Calculate a quick Pearson r for a pair in a DataFrame subset.

    Returns None if not enough data or zero variance.
    """
    series_a = df[col_a]
    series_b = df[col_b].shift(lag) if lag > 0 else df[col_b]

    pair_df = pl.DataFrame({"a": series_a, "b": series_b}).drop_nulls()

    # Need at least 20 points in each half to be meaningful
    if pair_df.height < 20:
        return None

    a_arr = pair_df["a"].to_numpy()
    b_arr = pair_df["b"].to_numpy()

    if np.std(a_arr) < 1e-10 or np.std(b_arr) < 1e-10:
        return None

    r, _ = stats.pearsonr(a_arr, b_arr)
    return r if not math.isnan(r) else None


# ─── Step 8: Build candidates ────────────────────────────────────────


def _build_candidates(
    results: list[dict],
    window_days: int,
    window_end: pendulum.DateTime,
) -> list[CorrelationCandidate]:
    """Convert raw result dicts into validated CorrelationCandidate models.

    Parses the entity_key back into type + id, and lets the Pydantic
    model handle tier assignment and derived fields.
    """
    candidates = []

    for result in results:
        # Parse "asset:NVDA" back into type and id
        a_type, a_id = result["entity_a"].split(":", 1)
        b_type, b_id = result["entity_b"].split(":", 1)

        try:
            candidate = CorrelationCandidate(
                entity_a_type=a_type,
                entity_a_id=a_id,
                entity_b_type=b_type,
                entity_b_id=b_id,
                correlation=round(result["pearson_r"], 6),
                p_value=round(result.get("adjusted_p", result["pearson_p"]), 6),
                method="pearson",
                lag_days=result["lag_days"],
                observation_count=result["observation_count"],
                window_days=window_days,
                window_end=window_end,
            )
            candidates.append(candidate)
        except Exception as error:
            logger.warning(
                "Failed to validate candidate %s vs %s: %s",
                result["entity_a"],
                result["entity_b"],
                error,
            )

    return candidates


# ─── Step 9: Store in Neo4j ──────────────────────────────────────────


def _clear_old_candidates(graph: GraphStorage, window_days: int) -> int:
    """Delete CORRELATES_WITH candidates from a specific window size.

    Only clears candidates from the same window size so different
    windows (21, 63, 252 day) can coexist in the graph. Active or
    promoted relationships are NOT touched.
    """
    query = """
        MATCH ()-[r:CORRELATES_WITH {
            source: 'statistical',
            status: 'candidate',
            window_days: $window_days
        }]->()
        DELETE r RETURN count(r) AS deleted
    """
    result = graph.run_query(query, {"window_days": window_days})
    return result[0]["deleted"] if result else 0


def _store_candidates_batch(
    graph: GraphStorage,
    candidates: list[CorrelationCandidate],
) -> int:
    """Batch-store all candidates in Neo4j using UNWIND.

    Each candidate becomes a CORRELATES_WITH edge between two entity nodes.
    We match nodes by id without specifying the label — the uniqueness
    constraints ensure there's only one node with each id.

    The MERGE includes lag_days so the same pair can have separate
    relationships at different lags (e.g. same-day + 5-day lead).
    """
    if not candidates:
        return 0

    # Build the batch data — list of dicts for UNWIND
    batch = []
    for c in candidates:
        props = c.to_neo4j_properties()
        batch.append({
            "a_id": c.entity_a_id,
            "b_id": c.entity_b_id,
            "lag": c.lag_days,
            "props": props,
        })

    # UNWIND sends everything in one Cypher call instead of N separate ones.
    # We match nodes by id across all labels — works because ids are unique.
    query = """
        UNWIND $batch AS item
        MATCH (a {id: item.a_id})
        MATCH (b {id: item.b_id})
        MERGE (a)-[r:CORRELATES_WITH {lag_days: item.lag}]->(b)
        SET r += item.props
        RETURN count(r) AS stored
    """

    result = graph.run_query(query, {"batch": batch})
    return result[0]["stored"] if result else 0
