"""
Partial correlation engine — finds direct relationships between entities.

Regular correlation tells you "A and B move together." But maybe they only
move together because they both follow the market. Partial correlation asks:
"Do A and B still move together after accounting for EVERYTHING else?"

The math:
    1. Fit Graphical Lasso at multiple alpha values (sparsity levels)
    2. Select the best alpha using EBIC (Extended BIC) — controls false
       discovery by penalizing model complexity
    3. The resulting sparse precision matrix has exact structural zeros
       for pairs with no direct connection
    4. Convert nonzero precision entries to partial correlations:
       ρ_ij = -P_ij / sqrt(P_ii * P_jj)

No post-hoc p-value testing or FDR correction — EBIC handles false
discovery at the estimation stage. See Research/Glasso FDR Redundancy
Question.md for why this is correct.

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
from zerofin.data.tickers import (
    COMMODITIES,
    KEY_STOCKS,
    NON_DAILY_INDICATORS,
    REDUNDANCY_LOOKUP,
    STOCK_SECTOR_MAP,
    US_INDICES,
)
from zerofin.models.correlations import CorrelationCandidate, CorrelationRunSummary
from zerofin.storage.graph import GraphStorage
from zerofin.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)

# Minimum observations needed for reliable precision matrix estimation.
# At least 30 — standard threshold for reasonable covariance estimates.
MIN_PRECISION_OBS = 30

# Minimum variables needed (partial correlation needs at least 3)
MIN_PRECISION_VARS = 3

# ─── Two-Pass Entity Classification ───────────────────────────────
#
# Pass 1 (micro): individual stocks + non-overlapping assets
# Pass 2 (macro): sector ETFs + broad indices + international ETFs
#
# ETFs that contain stocks in our matrix are excluded from Pass 1
# to prevent edge absorption (the ETF soaks up stock-to-stock edges).

def _build_pass1_excludes() -> set[str]:
    """Build the set of entity IDs to exclude from Pass 1 (micro).

    These are ETFs/indices that contain individual stocks or futures
    in our matrix. Including them causes the glasso to route signals
    through the ETF intermediary, zeroing out direct connections.

    Derived from tickers.py data — not hardcoded. When tickers.py
    changes, these exclusions update automatically.
    """
    # Sector ETFs that our stocks map to (from STOCK_SECTOR_MAP)
    sector_etfs = set()
    for mapping in STOCK_SECTOR_MAP.values():
        sector_etfs.add(mapping["sector"])
        if "sub_sector" in mapping:
            sector_etfs.add(mapping["sub_sector"])

    # Broad market indices — derived from US_INDICES, excluding
    # volatility indices which don't contain our stocks
    volatility_indices = {"^VIX", "^VVIX", "^MOVE", "^GVZ", "^OVX"}
    broad_indices = set(US_INDICES) - volatility_indices

    # Factor/broad ETFs that hold stocks from across all sectors
    # These are in SECTOR_ETFS but not mapped via STOCK_SECTOR_MAP
    factor_etfs = {"RSP", "MTUM", "QUAL", "USMV"}

    # International ETFs that contain stocks we track as individuals
    # Check KEY_STOCKS against known ETF holdings
    tracked_stocks = set(KEY_STOCKS)
    intl_with_overlap = set()
    _intl_holdings = {
        "FXI": {"BABA", "PDD"},
        "KWEB": {"BABA", "PDD"},
        "EWT": {"TSM"},
        "EWG": {"ASML"},
    }
    for etf, holdings in _intl_holdings.items():
        if holdings & tracked_stocks:
            intl_with_overlap.add(etf)

    # Commodity ETFs that contain futures we track
    tracked_futures = set(COMMODITIES)
    commodity_with_overlap = set()
    _commodity_holdings = {
        "GLD": {"GC=F"},
        "DBA": {"ZC=F", "ZW=F", "ZS=F"},
        "DBC": {"CL=F", "GC=F", "SI=F", "HG=F"},
    }
    for etf, holdings in _commodity_holdings.items():
        if holdings & tracked_futures:
            commodity_with_overlap.add(etf)

    return (
        sector_etfs
        | broad_indices
        | factor_etfs
        | intl_with_overlap
        | commodity_with_overlap
    )


def _build_pass2_excludes() -> set[str]:
    """Build the set of entity IDs to exclude from Pass 2 (macro).

    Individual stocks are excluded — Pass 2 focuses on sector-level
    and cross-asset macro connections.
    """
    return set(KEY_STOCKS)


# Cache the exclusion sets (they don't change during a run)
PASS1_EXCLUDES = _build_pass1_excludes()
PASS2_EXCLUDES = _build_pass2_excludes()


def run_partial_correlation_pipeline(
    db: PostgresStorage,
    graph: GraphStorage,
    *,
    window_days: int | None = None,
) -> CorrelationRunSummary:
    """Run two-pass partial correlation discovery for a single window.

    Pass 1 (micro): Individual stocks + non-overlapping assets.
    Finds direct stock-to-stock, stock-to-commodity, stock-to-macro
    connections without ETF intermediaries absorbing edges.

    Pass 2 (macro): Sector ETFs + indices + international ETFs.
    Finds sector-to-macro, sector-to-commodity, cross-sector
    connections without individual stocks cluttering the matrix.

    Both passes use Graphical Lasso with EBIC tuning.
    """
    start_time = time.monotonic()

    window = window_days or max(settings.CORRELATION_WINDOWS)
    window_end = pendulum.now("UTC")
    window_start = window_end.subtract(days=window)

    logger.info(
        "Starting two-pass partial correlation: %d-day window (%s to %s)",
        window,
        window_start.to_date_string(),
        window_end.to_date_string(),
    )

    # Step 1: Fetch and transform data (shared by both passes)
    all_rows = db.get_all_market_data_range(start=window_start, end=window_end)
    raw_rows = [
        row for row in all_rows
        if row["entity_id"] not in NON_DAILY_INDICATORS
    ]

    if not raw_rows:
        logger.warning("No data found — nothing to correlate")
        return _empty_summary(window, window_end, start_time)

    min_obs = max(15, int(window * settings.CORRELATION_MIN_OBSERVATIONS_RATIO))
    wide_df = _build_wide_dataframe(raw_rows, min_obs=min_obs)

    # Transform once — both passes use the same transformed data
    returns_df = _compute_transforms(wide_df)
    returns_df = _z_score_all(returns_df)
    returns_df = _winsorize(returns_df)

    all_cols = [c for c in returns_df.columns if c != "date"]
    logger.info("Transformed %d entities total", len(all_cols))

    # ── Pass 1: Micro (stocks + non-overlapping assets) ──────────
    pass1_cols = [
        c for c in all_cols
        if c.split(":", 1)[1] not in PASS1_EXCLUDES
    ]
    logger.info(
        "Pass 1 (micro): %d entities (%d excluded)",
        len(pass1_cols),
        len(all_cols) - len(pass1_cols),
    )

    pass1_results = _run_single_pass(
        returns_df, pass1_cols, "micro"
    )

    # ── Pass 2: Macro (sectors + indices + international) ────────
    pass2_cols = [
        c for c in all_cols
        if c.split(":", 1)[1] not in PASS2_EXCLUDES
    ]
    logger.info(
        "Pass 2 (macro): %d entities (%d excluded)",
        len(pass2_cols),
        len(all_cols) - len(pass2_cols),
    )

    pass2_results = _run_single_pass(
        returns_df, pass2_cols, "macro"
    )

    # ── Merge results from both passes ───────────────────────────
    all_results = pass1_results + pass2_results

    # Deduplicate: if the same pair appears in both passes, keep
    # the one with higher strength
    seen_pairs: dict[tuple[str, str], dict] = {}
    for r in all_results:
        pair = (
            min(r["entity_a"], r["entity_b"]),
            max(r["entity_a"], r["entity_b"]),
        )
        if pair not in seen_pairs or abs(r["correlation_r"]) > abs(
            seen_pairs[pair]["correlation_r"]
        ):
            seen_pairs[pair] = r

    results = list(seen_pairs.values())
    logger.info(
        "Merged: %d from micro + %d from macro = %d unique pairs",
        len(pass1_results),
        len(pass2_results),
        len(results),
    )

    # Gate 2 plausibility check
    results = [
        r for r in results
        if is_pair_plausible(r["entity_a"], r["entity_b"])
    ]
    logger.info("%d survived plausibility filter", len(results))

    # Split into tiers
    tier1_threshold = settings.PARTIAL_CORRELATION_TIER1
    tier1_results = [
        r for r in results if abs(r["correlation_r"]) >= tier1_threshold
    ]
    tier2_results = [
        r for r in results if abs(r["correlation_r"]) < tier1_threshold
    ]

    logger.info(
        "Tiered: %d active (>=%.2f), %d pending verification (%.2f-%.2f)",
        len(tier1_results),
        tier1_threshold,
        len(tier2_results),
        settings.PARTIAL_CORRELATION_TIER2_FLOOR,
        tier1_threshold,
    )

    # Note: total_pairs counts both passes separately. Entities in
    # both passes (commodities, FRED, bonds) are counted twice. This
    # overstates the denominator but is acceptable for monitoring.
    total_pairs = (
        len(pass1_cols) * (len(pass1_cols) - 1) // 2
        + len(pass2_cols) * (len(pass2_cols) - 1) // 2
    )

    # Tier 1: active candidates
    tier1_candidates = _build_partial_candidates(
        tier1_results, window, window_end,
    )
    # Tier 2: pending verification
    tier2_candidates = _build_partial_candidates(
        tier2_results, window, window_end,
        status="pending_verification",
    )

    cleared = _clear_old_partial_candidates(graph, window)
    logger.info("Cleared %d old partial candidates", cleared)

    stored_active = _store_candidates_batch(graph, tier1_candidates)
    stored_pending = _store_candidates_batch(graph, tier2_candidates)
    stored = stored_active + stored_pending

    logger.info(
        "Stored %d active + %d pending = %d total in Neo4j",
        stored_active,
        stored_pending,
        stored,
    )

    duration = time.monotonic() - start_time
    logger.info(
        "Two-pass partial correlation complete in %.1fs: "
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


def _run_single_pass(
    returns_df: pl.DataFrame,
    entity_cols: list[str],
    pass_name: str,
) -> list[dict]:
    """Run glasso on a subset of entities and return significant pairs.

    This is the core of each pass — compute the precision matrix,
    extract pairs above threshold. No storage, no plausibility check
    (those happen after merging both passes).
    """
    if len(entity_cols) < MIN_PRECISION_VARS:
        logger.warning(
            "Pass %s: only %d entities, skipping",
            pass_name,
            len(entity_cols),
        )
        return []

    partial_matrix, valid_cols, n_obs = _compute_partial_correlation_matrix(
        returns_df, entity_cols
    )

    if partial_matrix is None:
        logger.warning("Pass %s: could not compute precision matrix", pass_name)
        return []

    # Use Tier 2 floor as threshold — captures both tiers.
    # The pipeline function splits into tiers after merging.
    threshold = settings.PARTIAL_CORRELATION_TIER2_FLOOR
    results = _extract_significant_pairs(
        partial_matrix, valid_cols, threshold, n_obs
    )

    logger.info(
        "Pass %s: %d pairs above %.2f threshold",
        pass_name,
        len(results),
        threshold,
    )

    return results


# ─── Core math ─────────────────────────────────────────────────────


def _compute_partial_correlation_matrix(
    returns_df: pl.DataFrame,
    entity_cols: list[str],
) -> tuple[np.ndarray | None, list[str], int]:
    """Compute partial correlations via Graphical Lasso with EBIC.

    Graphical Lasso enforces sparsity directly — weak connections
    become exact mathematical zeros. EBIC selects the sparsity level,
    controlling false discovery at the estimation stage.

    No PCA factor removal — empirically proven to destroy thematic
    signal for mega-cap stocks in our mixed-asset universe.
    See Research/Partial Correlation — Final Recommendation.md.

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
    # Center data explicitly — GraphicalLasso with assume_centered=False
    # would do this internally, but we need the centered covariance for
    # EBIC scoring to be consistent with the precision matrix
    data_centered = data - data.mean(axis=0)
    emp_cov = np.cov(data_centered, rowvar=False, ddof=0)

    # Generate alpha range (log-spaced from small to large)
    # Small alpha = dense graph, large alpha = sparse graph
    alpha_max = np.max(np.abs(emp_cov - np.diag(np.diag(emp_cov))))
    alpha_min = alpha_max * settings.GLASSO_ALPHA_RANGE_RATIO
    alphas = np.logspace(
        np.log10(alpha_min),
        np.log10(alpha_max),
        settings.EBIC_N_ALPHAS,
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
                    max_iter=settings.GLASSO_MAX_ITER,
                    tol=settings.GLASSO_CONVERGENCE_TOL,
                    assume_centered=True,
                )
                gl.fit(data_centered)

                precision = gl.precision_

                # Compute log-likelihood using consistent covariance
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
                    + 4 * settings.EBIC_GAMMA
                    * n_edges * np.log(n_vars)
                )

                if ebic < best_ebic:
                    best_ebic = ebic
                    best_precision = precision

            except (
                FloatingPointError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                # Numerical failures at extreme alpha values are expected —
                # singular matrices, convergence issues, etc. Skip and try next.
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
            "correlation_r": float(r),
            "correlation_p": 0.0,  # glasso handles significance via EBIC
            "lag_days": 0,
            "observation_count": n_obs,
        })

    return results


# ─── Build and store candidates ────────────────────────────────────


def _build_partial_candidates(
    results: list[dict],
    window_days: int,
    window_end: pendulum.DateTime,
    *,
    status: str = "candidate",
) -> list[CorrelationCandidate]:
    """Convert result dicts into CorrelationCandidate models.

    Args:
        status: "candidate" for Tier 1 (active), "pending_verification"
                for Tier 2 (waiting for DeepSeek confirmation).
    """
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
                correlation=round(result["correlation_r"], 6),
                # p_value=0.0 — EBIC handles significance, not p-tests
                p_value=0.0,
                method="partial",
                lag_days=0,
                observation_count=result["observation_count"],
                window_days=window_days,
                window_end=window_end,
                status=status,
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
    """Delete old partial correlation candidates for this window size.

    Clears both active (candidate) and pending_verification edges.
    """
    query = """
        MATCH ()-[r:CORRELATES_WITH {
            source: 'statistical',
            method: 'partial',
            window_days: $window_days
        }]->()
        WHERE r.status IN ['candidate', 'pending_verification']
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
        pairs_surviving_fdr=0,  # No FDR for glasso; EBIC handles it
        relationships_stored=0,
        window_days=window,
        window_end=window_end,
        duration_seconds=time.monotonic() - start_time,
    )
