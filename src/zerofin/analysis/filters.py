"""
Statistical filters for the correlation engine.

These functions remove false positives from raw correlation results:
    - FDR correction (Benjamini-Hochberg) to control false discovery rate
    - Stability filter to ensure correlations hold across both halves of the window
    - Monthly stability filter (bootstrap CI, sign consistency, magnitude check)
"""

from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl
from scipy import stats

from zerofin.config import settings

logger = logging.getLogger(__name__)

# Minimum observations needed in each stability-check half-window
MIN_HALF_WINDOW_OBS = 20

# Series with std below this are treated as zero-variance (no signal)
MIN_VARIANCE = 1e-10


# ─── FDR correction ──────────────────────────────────────────────────


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
    # Copy each dict to avoid mutating the caller's input list
    surviving = []
    for i, result in enumerate(sorted_results):
        annotated = {**result, "adjusted_p": adjusted_p_values[i]}
        if adjusted_p_values[i] <= alpha:
            surviving.append(annotated)

    return surviving


# ─── Stability filter ────────────────────────────────────────────────


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

    if pair_df.height < MIN_HALF_WINDOW_OBS:
        return None

    a_arr = pair_df["a"].to_numpy()
    b_arr = pair_df["b"].to_numpy()

    if np.std(a_arr) < MIN_VARIANCE or np.std(b_arr) < MIN_VARIANCE:
        return None

    r, _ = stats.pearsonr(a_arr, b_arr)
    return r if not math.isnan(r) else None


# ─── Monthly stability filter ────────────────────────────────────────


def apply_monthly_stability_filter(
    results: list[dict],
    joined_df: pl.DataFrame,
) -> list[dict]:
    """Multi-criteria stability filter for monthly correlations.

    Monthly data has few observations (~36 months), so correlations
    are noisy. This filter checks four things to make sure a
    correlation is real and not a fluke from one weird period:

    1. Bootstrap CI excludes zero (is it statistically real?)
    2. Sign consistency in halves (same direction in both halves?)
    3. Tercile consistency (same direction in all thirds?)
    4. Magnitude stability (similar strength across halves?)

    Results that fail get flagged as 'under_review' instead of deleted.
    """
    stable = []
    flagged = []

    for result in results:
        col_a = result["entity_a"]
        col_b = result["entity_b"]

        pair = joined_df.select([col_a, col_b]).drop_nulls()
        if pair.height < 10:
            continue

        a = pair[col_a].to_numpy()
        b = pair[col_b].to_numpy()

        # Check 1: Bootstrap CI excludes zero
        ci_pass = _bootstrap_ci_excludes_zero(a, b)

        # Check 2: Sign consistency in halves
        sign_pass = _sign_consistent(a, b)

        # Check 3: Tercile consistency (if enough data)
        tercile_pass = _tercile_consistent(a, b) if len(a) >= 18 else True

        # Check 4: Magnitude stability
        mag_pass = _magnitude_stable(a, b)

        if ci_pass and sign_pass and tercile_pass and mag_pass:
            result["stability_status"] = "stable"
            stable.append(result)
        else:
            # Flag why it failed
            reasons = []
            if not ci_pass:
                reasons.append("bootstrap_ci_includes_zero")
            if not sign_pass:
                reasons.append("sign_inconsistent")
            if not tercile_pass:
                reasons.append("tercile_inconsistent")
            if not mag_pass:
                reasons.append("magnitude_unstable")
            result["stability_status"] = "under_review"
            result["stability_flags"] = reasons
            flagged.append(result)

    logger.info(
        "Monthly stability: %d stable, %d flagged",
        len(stable), len(flagged),
    )

    # Return both — stable ones go to Neo4j, flagged ones get stored
    # with lower confidence
    return stable + flagged


def _bootstrap_ci_excludes_zero(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
) -> bool:
    """Run Spearman 1,000 times on random samples.

    If the 95% confidence interval includes zero, the correlation
    might not be real — it could just be noise.
    """
    n = len(a)
    boot_corrs = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(a[idx], b[idx])
        if not math.isnan(r):
            boot_corrs.append(r)

    if len(boot_corrs) < 100:
        return False

    lower = np.percentile(boot_corrs, 2.5)
    upper = np.percentile(boot_corrs, 97.5)

    # CI must not include zero
    return (lower > 0 and upper > 0) or (lower < 0 and upper < 0)


def _sign_consistent(a: np.ndarray, b: np.ndarray) -> bool:
    """Split in half. Same sign in both halves?

    If it's positive in one half and negative in the other,
    something weird happened during one period.
    """
    mid = len(a) // 2
    if mid < 6:
        return True  # Not enough data to split

    r1, _ = stats.spearmanr(a[:mid], b[:mid])
    r2, _ = stats.spearmanr(a[mid:], b[mid:])

    if math.isnan(r1) or math.isnan(r2):
        return False

    return (r1 > 0) == (r2 > 0)


def _tercile_consistent(a: np.ndarray, b: np.ndarray) -> bool:
    """Split into thirds. Same sign in all three?

    Even harder for a fluke to survive all three chunks.
    Like asking three witnesses instead of two.
    """
    n = len(a)
    t = n // 3
    if t < 6:
        return True  # Not enough data

    signs = []
    for i in range(3):
        start = i * t
        end = start + t if i < 2 else n  # Last chunk gets remainder
        r, _ = stats.spearmanr(a[start:end], b[start:end])
        if math.isnan(r):
            return False
        signs.append(r > 0)

    return len(set(signs)) == 1


def _magnitude_stable(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = 0.4,
) -> bool:
    """Is the strength roughly similar across halves?

    If the overall is 0.80 but one half is 0.05 and the other
    is 0.95, the relationship only exists in one time period.
    The weakest half must be at least 40% of the full strength.
    """
    r_full, _ = stats.spearmanr(a, b)
    if math.isnan(r_full) or abs(r_full) < 0.01:
        return False

    mid = len(a) // 2
    if mid < 6:
        return True

    r1, _ = stats.spearmanr(a[:mid], b[:mid])
    r2, _ = stats.spearmanr(a[mid:], b[mid:])

    if math.isnan(r1) or math.isnan(r2):
        return False

    min_half = min(abs(r1), abs(r2))
    return min_half >= threshold * abs(r_full)
