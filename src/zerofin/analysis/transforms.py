"""
Data transforms for the correlation engine.

These functions prepare raw price/indicator data for correlation analysis:
    1. Compute natural returns per asset type (log returns, first diff, pct change)
    2. Z-score all returns to a common scale
    3. Winsorize to cap extreme moves
    4. Remove market + sector beta (joint OLS residualization)
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from zerofin.config import settings
from zerofin.data.tickers import STOCK_SECTOR_MAP

logger = logging.getLogger(__name__)

# Tickers that need percentage change instead of log returns.
# VIX can spike but never goes to zero, so pct change works.
VOLATILITY_TICKERS = {"^VIX"}

# Minimum variance threshold — series with std below this are treated
# as constant (zero information) and excluded from analysis.
MIN_VARIANCE = 1e-10

# Minimum data points needed for OLS regression in beta removal
MIN_REGRESSION_OBS = 30


# ─── Stage 1: Natural returns per asset type ─────────────────────────


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


# ─── Stage 2: Z-score to common scale ───────────────────────────────


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

        if std is None or std < MIN_VARIANCE:
            # Near-zero variance — can't z-score, fill with nulls
            result_cols[col] = pl.Series(col, [None] * series.len())
        else:
            result_cols[col] = ((series - mean) / std).alias(col)

    return pl.DataFrame(result_cols)


# ─── Winsorize ───────────────────────────────────────────────────────


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


# ─── Remove market + sector beta ─────────────────────────────────────


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

        if sector_key and sector_key in df.columns:
            sector = df[sector_key].to_numpy().astype(float)
            mask = mask & ~np.isnan(sector)
            factors = [np.ones(mask.sum()), market[mask], sector[mask]]
        else:
            factors = [np.ones(mask.sum()), market[mask]]

        if mask.sum() < MIN_REGRESSION_OBS:
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
