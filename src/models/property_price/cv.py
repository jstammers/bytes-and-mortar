"""Time-series cross-validation splitters for UK property price data.

Three strategies are supported, all accessible via ``make_cv_splits``:

``sliding_window``  (default)
    Fixed-size training window that advances one fold at a time.
    Produces approximately equal-sized train sets — good default for
    comparing models on a fair footing.

``expanding_window``
    Training set grows with each fold (all history up to the cutoff).
    Rewards models that benefit from more data.

``year_based``
    One fold per calendar year.  Training set is everything before the
    held-out year.  Useful for year-over-year performance reporting.

All strategies honour a ``gap_months`` buffer between the end of the
training window and the start of validation to avoid target leakage from
properties that take weeks to register with the Land Registry.

Usage::

    from src.models.property_price.cv import make_cv_splits, CVConfig, CVStrategy

    splits = make_cv_splits(df, CVConfig(strategy=CVStrategy.sliding_window))
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.property_price.config import CVConfig, CVStrategy

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def make_cv_splits(
    df: pd.DataFrame,
    config: CVConfig,
    date_col: str = "date_of_transfer",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_indices, val_indices) arrays.

    Dispatches to the appropriate strategy based on ``config.strategy``.
    All indices are integer positional (``iloc``-compatible).

    Args:
        df: Full training DataFrame (test-year rows should already be excluded).
        config: CV configuration.
        date_col: Column name containing transaction dates.

    Returns:
        List of ``(train_idx, val_idx)`` integer-index tuples.

    Raises:
        ValueError: If ``date_col`` is missing or no valid folds can be formed.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. Available columns: {df.columns.tolist()}"
        )

    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    if config.strategy == CVStrategy.sliding_window:
        splits = list(_sliding_window(df_sorted, config, date_col))
    elif config.strategy == CVStrategy.expanding_window:
        splits = list(_expanding_window(df_sorted, config, date_col))
    elif config.strategy == CVStrategy.year_based:
        splits = list(_year_based(df_sorted, config, date_col))
    else:
        raise ValueError(f"Unknown CV strategy: {config.strategy}")

    if not splits:
        raise ValueError(
            f"CV strategy '{config.strategy}' produced 0 folds. "
            "Check that the DataFrame spans enough time for the requested n_splits / window_months."
        )

    logger.info(
        "CV strategy=%s produced %d folds (gap=%d months)",
        config.strategy,
        len(splits),
        config.gap_months,
    )
    return splits


def _month_offset(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    """Add ``months`` calendar months to a Timestamp."""
    return dt + pd.DateOffset(months=months)


def _sliding_window(
    df: pd.DataFrame,
    config: CVConfig,
    date_col: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Fixed-size sliding training window, advancing in equal steps."""
    dates: pd.Series = pd.to_datetime(df[date_col])
    min_date, max_date = dates.min(), dates.max()

    window_months = config.window_months or 24
    # Total span available for validation
    total_span_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month
    val_months = max(
        1,
        (total_span_months - window_months - config.gap_months) // config.n_splits,
    )

    for fold in range(config.n_splits):
        # Validation window starts after initial training window + gap
        val_start = _month_offset(min_date, window_months + config.gap_months + fold * val_months)
        val_end = _month_offset(val_start, val_months)

        train_start = _month_offset(val_start, -window_months - config.gap_months)
        train_end = _month_offset(val_start, -config.gap_months)

        if val_end > max_date:
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        val_mask = (dates >= val_start) & (dates < val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.debug("Fold %d skipped: empty train or val set", fold)
            continue

        logger.debug(
            "Fold %d: train %s→%s (%d rows), val %s→%s (%d rows)",
            fold,
            train_start.date(),
            train_end.date(),
            len(train_idx),
            val_start.date(),
            val_end.date(),
            len(val_idx),
        )
        yield train_idx, val_idx


def _expanding_window(
    df: pd.DataFrame,
    config: CVConfig,
    date_col: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Growing training window — all history up to the cutoff date."""
    dates: pd.Series = pd.to_datetime(df[date_col])
    min_date, max_date = dates.min(), dates.max()

    total_span_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month
    # Reserve the last portion for validation folds
    initial_train_months = max(
        6,
        total_span_months - (config.n_splits * 2 + config.gap_months),
    )
    val_months = max(
        1,
        (total_span_months - initial_train_months - config.gap_months) // config.n_splits,
    )

    for fold in range(config.n_splits):
        train_end = _month_offset(min_date, initial_train_months + fold * val_months)
        val_start = _month_offset(train_end, config.gap_months)
        val_end = _month_offset(val_start, val_months)

        if val_end > max_date:
            break

        train_mask = dates < train_end
        val_mask = (dates >= val_start) & (dates < val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.debug("Fold %d skipped: empty train or val set", fold)
            continue

        logger.debug(
            "Fold %d: train up to %s (%d rows), val %s→%s (%d rows)",
            fold,
            train_end.date(),
            len(train_idx),
            val_start.date(),
            val_end.date(),
            len(val_idx),
        )
        yield train_idx, val_idx


def _year_based(
    df: pd.DataFrame,
    config: CVConfig,
    date_col: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """One held-out year per fold; training set is all prior years."""
    dates: pd.Series = pd.to_datetime(df[date_col])

    if config.holdout_years:
        holdout_years = sorted(config.holdout_years)
    else:
        all_years = sorted(dates.dt.year.unique())
        # Hold out the last n_splits years
        holdout_years = all_years[-config.n_splits :]

    for year in holdout_years:
        gap_cutoff = pd.Timestamp(year=year, month=1, day=1) - pd.DateOffset(
            months=config.gap_months
        )
        train_mask = dates < gap_cutoff
        val_mask = dates.dt.year == year

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.debug("Year %d fold skipped: empty train or val set", year)
            continue

        logger.debug(
            "Year fold %d: train up to %s (%d rows), val year %d (%d rows)",
            year,
            gap_cutoff.date(),
            len(train_idx),
            year,
            len(val_idx),
        )
        yield train_idx, val_idx


def temporal_train_test_split(
    df: pd.DataFrame,
    test_years: list[int],
    date_col: str = "date_of_transfer",
    stratify_col: str | None = "property_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets by calendar year.

    Rows in ``test_years`` form the test set; all prior rows form the train set.
    Within the train set, ``stratify_col`` distribution is logged for reference
    (no resampling is performed — the temporal ordering is preserved).

    Args:
        df: Full DataFrame.
        test_years: Calendar years to hold out as the test set.
        date_col: Date column for the temporal split.
        stratify_col: Column whose class distribution is logged for inspection.

    Returns:
        ``(train_df, test_df)`` DataFrames.

    Raises:
        ValueError: If ``date_col`` is not present.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    dates = pd.to_datetime(df[date_col])
    test_mask = dates.dt.year.isin(test_years)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        "Temporal split: %d train rows, %d test rows (test years: %s)",
        len(train_df),
        len(test_df),
        test_years,
    )

    if stratify_col and stratify_col in df.columns:
        train_dist = train_df[stratify_col].value_counts(normalize=True).round(3).to_dict()
        test_dist = test_df[stratify_col].value_counts(normalize=True).round(3).to_dict()
        logger.info("Train %s distribution: %s", stratify_col, train_dist)
        logger.info("Test  %s distribution: %s", stratify_col, test_dist)

    return train_df, test_df


def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    stratify_col: str = "property_type",
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified subsample of ``n`` rows from ``df``.

    Useful during HPO to keep trial time short on large datasets.

    Args:
        df: Source DataFrame.
        n: Target sample size. If ``n >= len(df)`` the original DataFrame is
           returned unchanged.
        stratify_col: Column to stratify on.
        random_state: Random seed for reproducibility.

    Returns:
        Subsampled DataFrame with reset index.
    """
    if n >= len(df):
        return df

    if stratify_col in df.columns:
        # Pass integer n so sklearn returns exactly n rows (fraction-based
        # splitting rounds per stratum and can give n±k rows).
        _, sampled = train_test_split(
            df,
            test_size=n,
            stratify=df[stratify_col],
            random_state=random_state,
        )
    else:
        sampled = df.sample(n=n, random_state=random_state)

    return sampled.reset_index(drop=True)
