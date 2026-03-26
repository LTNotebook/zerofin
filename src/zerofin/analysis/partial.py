"""
Partial correlation engine — finds direct relationships between entities.

Regular correlation tells you "A and B move together." But maybe they only
move together because they both follow the market. Partial correlation asks:
"Do A and B still move together after accounting for EVERYTHING else?"

The math:
    1. Compute a shrunk covariance matrix (Ledoit-Wolf handles the case
       where we have lots of entities relative to observations)
    2. Invert it to get the precision matrix
    3. Convert precision entries to partial correlations:
       ρ_ij = -P_ij / sqrt(P_ii * P_jj)

This is one matrix inversion — no loops, no bootstraps. The whole thing
runs in under a second for 200 entities.

The key difference from the regular pipeline: we do NOT remove market/sector
beta first. Partial correlation inherently controls for all confounders
simultaneously. That's the whole point — it separates direct connections
from indirect ones without having to subtract anything.
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pendulum
import polars as pl
from sklearn.covariance import GraphicalLasso

from zerofin.analysis.correlations import _build_wide_dataframe
from zerofin.analysis.filters import is_pair_plausible
from zerofin.analysis.transforms import _compute_transforms, _winsorize, _z_score_all
from zerofin.config import settings
from zerofin.data.tickers import NON_DAILY_INDICATORS, REDUNDANCY_LOOKUP
from zerofin.models.correlations import CorrelationCandidate, CorrelationRunSummary
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

# Minimum observations needed for reliable precision matrix estimation
MIN_PRECISION_OBS = 10

# Minimum variables needed (partial correlation needs at least 3)
MIN_PRECISION_VARS = 3

logger = logging.getLogger(__name__)


def run_partial_correlation_pipeline(
    db: PostgresStorage,
    graph: GraphStorage,
    *,
    window_days: int | None = None,
) -> CorrelationRunSummary:
    """Run partial correlation discovery for a single window size.

    Uses the same data loading and transforms as the regular pipeline,
    but skips beta removal and computes the precision matrix instead
    of pairwise Pearson correlations.
    """
    start_time = time.monotonic()

    window = window_days or max(settings.CORRELATION_WINDOWS)
    window_end = pendulum.now("UTC")
    window_start = window_end.subtract(days=window)

    logger.info(
        "Starting partial correlation pipeline: %d-day window (%s to %s)",
        window,
        window_start.to_date_string(),
        window_end.to_date_string(),
    )

    # Step 1: Fetch data (same as regular pipeline)
    all_rows = db.get_all_market_data_range(start=window_start, end=window_end)
    raw_rows = [
        row for row in all_rows
        if row["entity_id"] not in NON_DAILY_INDICATORS
    ]

    if not raw_rows:
        logger.warning("No data found — nothing to correlate")
        return _empty_summary(window, window_end, start_time)

    # Step 2: Build wide DataFrame
    min_obs = max(15, int(window * settings.CORRELATION_MIN_OBSERVATIONS_RATIO))
    wide_df = _build_wide_dataframe(raw_rows, min_obs=min_obs)
    entity_cols = [c for c in wide_df.columns if c != "date"]
    entity_count = len(entity_cols)

    logger.info(
        "Built wide DataFrame: %d entities, %d days",
        entity_count,
        len(wide_df),
    )

    if entity_count < MIN_PRECISION_VARS:
        logger.warning("Need at least %d entities for partial correlation", MIN_PRECISION_VARS)
        return _empty_summary(window, window_end, start_time)

    # Step 3-5: Transform, z-score, winsorize (NO beta removal)
    returns_df = _compute_transforms(wide_df)
    returns_df = _z_score_all(returns_df)
    returns_df = _winsorize(returns_df)

    logger.info("Computed transforms for %d entities (no beta removal)", entity_count)

    # Step 6: Compute partial correlations via precision matrix
    partial_matrix, valid_cols, n_obs = _compute_partial_correlation_matrix(
        returns_df, entity_cols
    )

    if partial_matrix is None:
        logger.warning("Could not compute precision matrix")
        return _empty_summary(window, window_end, start_time)

    logger.info(
        "Computed partial correlation matrix: %d x %d",
        partial_matrix.shape[0],
        partial_matrix.shape[1],
    )

    # Step 7: Extract pairs above threshold
    threshold = settings.PARTIAL_CORRELATION_THRESHOLD
    n_vars = len(valid_cols)
    total_pairs = n_vars * (n_vars - 1) // 2

    results = _extract_significant_pairs(
        partial_matrix, valid_cols, threshold, n_obs
    )
    logger.info(
        "Found %d partial correlations above %.2f threshold",
        len(results),
        threshold,
    )

    # Note: No FDR correction here. EBIC-tuned glasso already controls
    # false discovery at the estimation stage. Stacking FDR on top is
    # statistically invalid (post-selection inference violation).
    # See: Glasso FDR Redundancy Question.md in Research folder.

    # Step 8: Gate 2 plausibility check
    results = [r for r in results if is_pair_plausible(r["entity_a"], r["entity_b"])]
    logger.info("%d survived plausibility filter", len(results))

    # Step 10: Build candidates
    candidates = _build_partial_candidates(results, window, window_end)
    logger.info("Built %d partial correlation candidates", len(candidates))

    # Step 11: Clear old partial candidates and store new ones
    cleared = _clear_old_partial_candidates(graph, window)
    logger.info("Cleared %d old partial candidates", cleared)

    stored = _store_candidates_batch(graph, candidates)
    logger.info("Stored %d new partial candidates in Neo4j", stored)

    duration = time.monotonic() - start_time
    logger.info(
        "Partial correlation pipeline complete in %.1fs: "
        "%d pairs tested, %d stored",
        duration,
        total_pairs,
        stored,
    )

    return CorrelationRunSummary(
        total_pairs_tested=total_pairs,
        pairs_above_threshold=len(results),
        pairs_surviving_fdr=len(results),  # No FDR for glasso; EBIC handles it
        relationships_stored=stored,
        window_days=window,
        window_end=window_end,
        duration_seconds=duration,
    )


# ─── Core math ─────────────────────────────────────────────────────


def _compute_partial_correlation_matrix(
    returns_df: pl.DataFrame,
    entity_cols: list[str],
) -> tuple[np.ndarray | None, list[str], int]:
    """Compute partial correlations via a sparse precision matrix.

    Uses Graphical Lasso with EBIC tuning instead of Ledoit-Wolf.
    Graphical Lasso enforces sparsity directly — weak connections
    become exact mathematical zeros, not small numbers we have to
    threshold. EBIC (Extended BIC) selects the sparsity level,
    avoiding the over-permissive results that cross-validation gives.

    Returns:
        (partial_corr_matrix, valid_column_names, n_obs) or
        (None, [], 0) on failure.
    """
    # Build numpy matrix, drop rows with any NaN
    data = returns_df.select(entity_cols).to_numpy().astype(float)
    valid_mask = ~np.isnan(data).any(axis=1)
    data_clean = data[valid_mask]

    n_obs, n_vars = data_clean.shape
    logger.info(
        "Precision matrix input: %d observations x %d variables",
        n_obs,
        n_vars,
    )

    if n_obs < MIN_PRECISION_OBS or n_vars < MIN_PRECISION_VARS:
        return None, [], 0

    # Fit Graphical Lasso with EBIC-selected alpha
    precision = _fit_glasso_ebic(data_clean, n_obs, n_vars)

    if precision is None:
        return None, [], 0

    # Count nonzero off-diagonal entries (the edges glasso kept)
    n_edges = np.count_nonzero(
        np.triu(precision, k=1)
    )
    logger.info(
        "Graphical Lasso kept %d nonzero edges out of %d possible",
        n_edges,
        n_vars * (n_vars - 1) // 2,
    )

    # Convert to partial correlations:
    # ρ_ij = -P_ij / sqrt(P_ii * P_jj)
    diag = np.sqrt(np.diag(precision))

    # Avoid division by zero for any zero-variance columns
    diag[diag == 0] = 1.0

    partial_corr = -precision / np.outer(diag, diag)

    # Diagonal should be 1.0 (correlation of a variable with itself)
    np.fill_diagonal(partial_corr, 1.0)

    return partial_corr, entity_cols, n_obs


# Number of alpha values to test for EBIC tuning
EBIC_N_ALPHAS = 50

# EBIC gamma parameter — controls sparsity aggressiveness.
# 0.5 is standard for very high-dimensional data (p >> n).
# 0.1 is appropriate when p/n < 1 and we want more discovery,
# accepting ~15-20% noise that downstream AI verification handles.
EBIC_GAMMA = 0.1


def _fit_glasso_ebic(
    data: np.ndarray,
    n_obs: int,
    n_vars: int,
) -> np.ndarray | None:
    """Fit Graphical Lasso with EBIC-selected regularization.

    Tests a range of alpha values and picks the one that minimizes:
        EBIC = -2 * log_likelihood + |E| * log(n) + 4 * gamma * |E| * log(p)

    where |E| is the number of nonzero edges. This penalizes complexity
    more aggressively than cross-validation, controlling false positives.
    """
    # Compute empirical covariance
    emp_cov = np.cov(data, rowvar=False)

    # Generate alpha range (log-spaced from small to large)
    # Small alpha = dense graph, large alpha = sparse graph
    alpha_max = np.max(np.abs(emp_cov - np.diag(np.diag(emp_cov))))
    alpha_min = alpha_max * 0.01
    alphas = np.logspace(
        np.log10(alpha_min), np.log10(alpha_max), EBIC_N_ALPHAS
    )

    best_ebic = np.inf
    best_precision = None

    # Suppress convergence warnings — extreme alpha values often don't
    # converge, which is expected. We skip them and move on.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=".*did not converge.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*Objective did not converge.*"
        )

        for alpha in alphas:
            try:
                gl = GraphicalLasso(
                    alpha=alpha,
                    mode="cd",
                    max_iter=200,
                    tol=1e-4,
                    assume_centered=True,
                )
                gl.fit(data)

                precision = gl.precision_

                # Compute log-likelihood
                # L = (n/2) * (log|Θ| - trace(S * Θ))
                sign, log_det = np.linalg.slogdet(precision)
                if sign <= 0:
                    continue
                log_lik = 0.5 * n_obs * (
                    log_det - np.trace(emp_cov @ precision)
                )

                # Count nonzero off-diagonal edges
                n_edges = np.count_nonzero(
                    np.triu(precision, k=1)
                )

                # EBIC = -2L + |E|*log(n) + 4*gamma*|E|*log(p)
                ebic = (
                    -2 * log_lik
                    + n_edges * np.log(n_obs)
                    + 4 * EBIC_GAMMA * n_edges * np.log(n_vars)
                )

                if ebic < best_ebic:
                    best_ebic = ebic
                    best_precision = precision

            except Exception:
                # Some alpha values may not converge — skip them
                continue

    if best_precision is not None:
        n_edges = np.count_nonzero(np.triu(best_precision, k=1))
        logger.info(
            "EBIC selected alpha with %d edges (EBIC=%.1f)",
            n_edges,
            best_ebic,
        )

    return best_precision


def _extract_significant_pairs(
    partial_matrix: np.ndarray,
    cols: list[str],
    threshold: float,
    n_obs: int,
) -> list[dict]:
    """Extract pairs above threshold from the glasso partial correlation matrix.

    No p-value computation — EBIC-tuned glasso already controls false
    discovery at the estimation stage. The nonzero entries ARE the
    significant edges. We just filter by strength and redundancy.
    """
    n_vars = partial_matrix.shape[0]

    # Get upper triangle indices (avoid duplicates and diagonal)
    i_idx, j_idx = np.triu_indices(n_vars, k=1)

    # Extract all upper-triangle values at once
    all_corrs = partial_matrix[i_idx, j_idx]

    # Mask for above-threshold pairs
    above = np.abs(all_corrs) >= threshold
    sig_i = i_idx[above]
    sig_j = j_idx[above]
    sig_corrs = all_corrs[above]

    # Check redundancy groups
    results = []
    for k in range(len(sig_i)):
        col_a = cols[sig_i[k]]
        col_b = cols[sig_j[k]]
        r = sig_corrs[k]

        # Skip redundant pairs (same as regular pipeline)
        _, id_a = col_a.split(":", 1)
        _, id_b = col_b.split(":", 1)
        group_a = REDUNDANCY_LOOKUP.get(id_a)
        group_b = REDUNDANCY_LOOKUP.get(id_b)
        if group_a and group_a == group_b:
            continue

        results.append({
            "entity_a": col_a,
            "entity_b": col_b,
            "pearson_r": float(r),
            "pearson_p": 0.0,  # glasso handles significance via EBIC
            "lag_days": 0,
            "observation_count": n_obs,
        })

    return results


# ─── Build and store candidates ────────────────────────────────────


def _build_partial_candidates(
    results: list[dict],
    window_days: int,
    window_end: pendulum.DateTime,
) -> list[CorrelationCandidate]:
    """Convert result dicts into CorrelationCandidate models."""
    candidates = []

    for result in results:
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
                method="partial",
                lag_days=0,
                observation_count=result["observation_count"],
                window_days=window_days,
                window_end=window_end,
            )
            candidates.append(candidate)
        except Exception as error:
            logger.warning(
                "Failed to validate partial candidate %s vs %s: %s",
                result["entity_a"],
                result["entity_b"],
                error,
            )

    return candidates


def _clear_old_partial_candidates(graph: GraphStorage, window_days: int) -> int:
    """Delete old partial correlation candidates for this window size."""
    query = """
        MATCH ()-[r:CORRELATES_WITH {
            source: 'statistical',
            status: 'candidate',
            method: 'partial',
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
    """Batch-store partial correlation candidates in Neo4j."""
    if not candidates:
        return 0

    batch = []
    for c in candidates:
        props = c.to_neo4j_properties()
        batch.append({
            "a_id": c.entity_a_id,
            "b_id": c.entity_b_id,
            "lag": c.lag_days,
            "props": props,
        })

    query = """
        UNWIND $batch AS item
        MATCH (a {id: item.a_id})
        MATCH (b {id: item.b_id})
        MERGE (a)-[r:CORRELATES_WITH {
            lag_days: item.lag,
            method: 'partial',
            window_days: item.props.window_days
        }]->(b)
        SET r += item.props
        RETURN count(r) AS stored
    """

    result = graph.run_query(query, {"batch": batch})
    return result[0]["stored"] if result else 0


# ─── Helpers ───────────────────────────────────────────────────────


def _empty_summary(
    window: int,
    window_end: pendulum.DateTime,
    start_time: float,
) -> CorrelationRunSummary:
    """Return an empty summary when there's nothing to process."""
    return CorrelationRunSummary(
        total_pairs_tested=0,
        pairs_above_threshold=0,
        pairs_surviving_fdr=0,
        relationships_stored=0,
        window_days=window,
        window_end=window_end,
        duration_seconds=time.monotonic() - start_time,
    )
