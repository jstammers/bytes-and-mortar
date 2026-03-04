"""Tests for src/features/build_features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_TARGET_ENCODE_FEATURES,
    FeatureConfig,
    MedianByGroupBaseline,
    MissingStrategy,
    build_feature_pipeline,
    validate_features,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame covering all default feature columns."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "price": rng.integers(100_000, 900_000, n).astype(float),
            "property_type": rng.choice(["D", "S", "T", "F"], n),
            "old_new": rng.choice(["Y", "N"], n),
            "duration": rng.choice(["F", "L"], n),
            "county": rng.choice(["Greater London", "Surrey", "Kent"], n),
            "postcode_outward": rng.choice(["SW1", "E1", "N1", "SE1"], n),
            "current_energy_rating": rng.choice(["A", "B", "C", "D", "E"], n),
            "construction_age_band": rng.choice(["pre-1900", "1950s", "2000s"], n),
            "district": rng.choice(["Westminster", "Lambeth", "Islington", "Hackney"], n),
            "year": rng.integers(2018, 2024, n),
            "month": rng.integers(1, 13, n),
            "current_energy_efficiency": rng.integers(20, 100, n).astype(float),
            "total_floor_area": rng.uniform(30, 300, n),
            "number_habitable_rooms": rng.integers(1, 8, n).astype(float),
            "averageprice": rng.uniform(200_000, 1_000_000, n),
            "index": rng.uniform(100, 200, n),
        }
    )


@pytest.fixture
def config_impute() -> FeatureConfig:
    return FeatureConfig(missing_strategy=MissingStrategy.impute)


@pytest.fixture
def config_drop() -> FeatureConfig:
    return FeatureConfig(missing_strategy=MissingStrategy.drop)


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


class TestFeatureConfig:
    def test_default_all_features_union(self):
        cfg = FeatureConfig()
        expected = (
            DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES + DEFAULT_TARGET_ENCODE_FEATURES
        )
        assert cfg.all_features == expected

    def test_to_dict_keys(self):
        cfg = FeatureConfig()
        d = cfg.to_dict()
        assert "numeric_features" in d
        assert "missing_strategy" in d
        assert "log_transform_target" in d

    def test_to_dict_values_are_strings(self):
        d = FeatureConfig().to_dict()
        assert all(isinstance(v, str) for v in d.values())

    def test_custom_feature_lists(self):
        cfg = FeatureConfig(
            numeric_features=["year", "total_floor_area"],
            categorical_features=["property_type"],
            target_encode_features=[],
        )
        assert cfg.all_features == ["year", "total_floor_area", "property_type"]


# ---------------------------------------------------------------------------
# validate_features
# ---------------------------------------------------------------------------


class TestValidateFeatures:
    def test_passes_when_all_columns_present(self, sample_df):
        cfg = FeatureConfig()
        validate_features(sample_df, cfg)  # should not raise

    def test_raises_for_missing_column(self, sample_df):
        cfg = FeatureConfig(numeric_features=["nonexistent_col"])
        with pytest.raises(ValueError, match="nonexistent_col"):
            validate_features(sample_df, cfg)

    def test_error_message_lists_all_missing(self, sample_df):
        cfg = FeatureConfig(numeric_features=["col_a", "col_b"])
        with pytest.raises(ValueError) as exc_info:
            validate_features(sample_df, cfg)
        assert "col_a" in str(exc_info.value)
        assert "col_b" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_feature_pipeline
# ---------------------------------------------------------------------------


class TestBuildFeaturePipeline:
    def test_returns_sklearn_pipeline(self, config_impute):
        from sklearn.pipeline import Pipeline

        pipeline = build_feature_pipeline(config_impute)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_fits_and_transforms(self, sample_df, config_impute):
        cfg = FeatureConfig(
            numeric_features=["year", "total_floor_area"],
            categorical_features=["property_type"],
            target_encode_features=["district"],
            missing_strategy=MissingStrategy.impute,
        )
        pipeline = build_feature_pipeline(cfg)
        X = sample_df[cfg.all_features]  # noqa: N806
        y = sample_df["price"]
        result = pipeline.fit_transform(X, y)
        assert result.shape[0] == len(sample_df)
        assert result.shape[1] >= 3  # at least one col per transformer

    def test_pipeline_numeric_only(self, sample_df):
        cfg = FeatureConfig(
            numeric_features=["year", "total_floor_area"],
            categorical_features=[],
            target_encode_features=[],
        )
        pipeline = build_feature_pipeline(cfg)
        X = sample_df[cfg.all_features]  # noqa: N806
        result = pipeline.fit_transform(X, sample_df["price"])
        assert result.shape == (len(sample_df), 2)

    def test_output_is_finite(self, sample_df, config_impute):
        cfg = FeatureConfig(
            numeric_features=["year", "total_floor_area"],
            categorical_features=["property_type"],
            target_encode_features=[],
            missing_strategy=MissingStrategy.impute,
        )
        pipeline = build_feature_pipeline(cfg)
        X = sample_df[cfg.all_features]  # noqa: N806
        result = pipeline.fit_transform(X, sample_df["price"])
        assert np.isfinite(result).all()

    def test_with_missing_values_impute(self, sample_df):
        """Imputation strategy should produce no NaNs."""
        cfg = FeatureConfig(
            numeric_features=["total_floor_area"],
            categorical_features=["property_type"],
            target_encode_features=[],
            missing_strategy=MissingStrategy.impute,
        )
        df_with_nans = sample_df.copy()
        df_with_nans.loc[0:5, "total_floor_area"] = np.nan
        df_with_nans.loc[6:10, "property_type"] = np.nan

        pipeline = build_feature_pipeline(cfg)
        result = pipeline.fit_transform(df_with_nans[cfg.all_features], df_with_nans["price"])
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# MedianByGroupBaseline
# ---------------------------------------------------------------------------


class TestMedianByGroupBaseline:
    def test_fit_predict_basic(self, sample_df):
        baseline = MedianByGroupBaseline()
        X = sample_df[["property_type", "district"]]  # noqa: N806
        y = sample_df["price"]
        baseline.fit(X, y)
        preds = baseline.predict(X)
        assert preds.shape == (len(sample_df),)
        assert np.all(np.isfinite(preds))

    def test_predictions_are_positive(self, sample_df):
        baseline = MedianByGroupBaseline()
        X = sample_df[["property_type", "district"]]  # noqa: N806
        y = sample_df["price"]
        baseline.fit(X, y)
        preds = baseline.predict(X)
        assert np.all(preds > 0)

    def test_fallback_to_global_median_for_unseen_group(self, sample_df):
        """Predictions for rows with unseen groups should use global median fallback."""
        baseline = MedianByGroupBaseline()
        X_train = sample_df[["property_type", "district"]]  # noqa: N806
        y_train = sample_df["price"]
        baseline.fit(X_train, y_train)

        unseen = pd.DataFrame(
            {"property_type": ["X"], "district": ["Atlantis"]}  # both unseen
        )
        preds = baseline.predict(unseen)
        assert len(preds) == 1
        assert np.isfinite(preds[0])
        assert preds[0] == pytest.approx(baseline.global_median_, rel=1e-6)

    def test_fallback_to_property_type_median_for_unseen_district(self, sample_df):
        baseline = MedianByGroupBaseline()
        X_train = sample_df[["property_type", "district"]]  # noqa: N806
        y_train = sample_df["price"]
        baseline.fit(X_train, y_train)

        # Known property_type, unknown district
        unseen_district = pd.DataFrame({"property_type": ["D"], "district": ["NowhereVille"]})
        preds = baseline.predict(unseen_district)
        assert np.isfinite(preds[0])

    def test_check_is_fitted_raises_before_fit(self):
        from sklearn.exceptions import NotFittedError

        baseline = MedianByGroupBaseline()
        with pytest.raises(NotFittedError):
            baseline.predict(pd.DataFrame({"property_type": ["D"], "district": ["X"]}))
