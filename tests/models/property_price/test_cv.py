"""Tests for src/models/property_price/cv.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.property_price.config import CVConfig, CVStrategy
from src.models.property_price.cv import (
    make_cv_splits,
    stratified_subsample,
    temporal_train_test_split,
)


@pytest.fixture
def multi_year_df() -> pd.DataFrame:
    """Three years of monthly data — enough for all CV strategies."""
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="MS")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date_of_transfer": dates.repeat(10),  # 10 rows per month
            "price": rng.integers(100_000, 500_000, len(dates) * 10).astype(float),
            "property_type": rng.choice(["D", "S", "T", "F"], len(dates) * 10),
        }
    )


# ---------------------------------------------------------------------------
# make_cv_splits
# ---------------------------------------------------------------------------


class TestMakeCVSplits:
    def test_sliding_window_returns_correct_n_splits(self, multi_year_df):
        config = CVConfig(strategy=CVStrategy.sliding_window, n_splits=4, window_months=12)
        splits = make_cv_splits(multi_year_df, config)
        assert 1 <= len(splits) <= 4  # may be fewer if data runs out

    def test_expanding_window_train_grows(self, multi_year_df):
        config = CVConfig(strategy=CVStrategy.expanding_window, n_splits=3, gap_months=0)
        splits = make_cv_splits(multi_year_df, config)
        assert len(splits) >= 1
        train_sizes = [len(t) for t, _ in splits]
        # Each fold's training set should be at least as large as the previous
        assert train_sizes == sorted(train_sizes)

    def test_year_based_folds(self, multi_year_df):
        config = CVConfig(strategy=CVStrategy.year_based, n_splits=2, gap_months=0)
        splits = make_cv_splits(multi_year_df, config)
        assert len(splits) >= 1

    def test_no_overlap_between_train_and_val(self, multi_year_df):
        config = CVConfig(
            strategy=CVStrategy.sliding_window, n_splits=3, window_months=12, gap_months=1
        )
        splits = make_cv_splits(multi_year_df, config)
        for train_idx, val_idx in splits:
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, "Train and val indices must not overlap"

    def test_train_always_precedes_val_in_time(self, multi_year_df):
        config = CVConfig(
            strategy=CVStrategy.sliding_window, n_splits=3, window_months=12, gap_months=0
        )
        splits = make_cv_splits(multi_year_df, config)
        df_sorted = multi_year_df.sort_values("date_of_transfer").reset_index(drop=True)
        for train_idx, val_idx in splits:
            max_train_date = pd.to_datetime(df_sorted.iloc[train_idx]["date_of_transfer"]).max()
            min_val_date = pd.to_datetime(df_sorted.iloc[val_idx]["date_of_transfer"]).min()
            assert max_train_date <= min_val_date

    def test_missing_date_col_raises(self, multi_year_df):
        config = CVConfig(strategy=CVStrategy.sliding_window)
        with pytest.raises(ValueError, match="Date column"):
            make_cv_splits(multi_year_df, config, date_col="nonexistent")

    def test_year_based_explicit_holdout_years(self, multi_year_df):
        config = CVConfig(
            strategy=CVStrategy.year_based,
            gap_months=0,
            holdout_years=[2022, 2023],
        )
        splits = make_cv_splits(multi_year_df, config)
        assert len(splits) == 2


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------


class TestTemporalTrainTestSplit:
    def test_splits_correctly_by_year(self, multi_year_df):
        train, test = temporal_train_test_split(multi_year_df, test_years=[2023])
        train_years = pd.to_datetime(train["date_of_transfer"]).dt.year.unique()
        test_years = pd.to_datetime(test["date_of_transfer"]).dt.year.unique()
        assert 2023 not in train_years
        assert list(test_years) == [2023]

    def test_no_rows_lost(self, multi_year_df):
        train, test = temporal_train_test_split(multi_year_df, test_years=[2023])
        assert len(train) + len(test) == len(multi_year_df)

    def test_missing_date_col_raises(self, multi_year_df):
        with pytest.raises(ValueError, match="Date column"):
            temporal_train_test_split(multi_year_df, test_years=[2023], date_col="bad_col")


# ---------------------------------------------------------------------------
# stratified_subsample
# ---------------------------------------------------------------------------


class TestStratifiedSubsample:
    def test_returns_n_rows(self, multi_year_df):
        result = stratified_subsample(multi_year_df, n=100)
        assert len(result) == 100

    def test_returns_original_when_n_exceeds_length(self, multi_year_df):
        result = stratified_subsample(multi_year_df, n=99999)
        assert len(result) == len(multi_year_df)

    def test_preserves_approximate_class_proportions(self, multi_year_df):
        result = stratified_subsample(multi_year_df, n=200, stratify_col="property_type")
        original_dist = multi_year_df["property_type"].value_counts(normalize=True)
        sample_dist = result["property_type"].value_counts(normalize=True)
        for pt in original_dist.index:
            if pt in sample_dist.index:
                assert abs(original_dist[pt] - sample_dist[pt]) < 0.15
