"""Tests for src/features/derived.py — DerivedFeatureTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.derived import (
    DERIVED_NUMERIC_FEATURES,
    ENERGY_RATING_MAP,
    DerivedFeatureTransformer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small DataFrame with the columns DerivedFeatureTransformer reads."""
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame(
        {
            "total_floor_area": rng.uniform(30, 250, n),
            "number_habitable_rooms": rng.integers(1, 8, n).astype(float),
            "current_energy_rating": rng.choice(list(ENERGY_RATING_MAP), n),
            "price": rng.integers(100_000, 800_000, n).astype(float),
        }
    )


@pytest.fixture
def transformer() -> DerivedFeatureTransformer:
    return DerivedFeatureTransformer()


# ---------------------------------------------------------------------------
# ENERGY_RATING_MAP
# ---------------------------------------------------------------------------


class TestEnergyRatingMap:
    def test_all_grades_present(self):
        assert set(ENERGY_RATING_MAP) == {"A", "B", "C", "D", "E", "F", "G"}

    def test_a_is_best(self):
        assert ENERGY_RATING_MAP["A"] > ENERGY_RATING_MAP["G"]

    def test_strictly_decreasing(self):
        grades = ["A", "B", "C", "D", "E", "F", "G"]
        values = [ENERGY_RATING_MAP[g] for g in grades]
        assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# DerivedFeatureTransformer
# ---------------------------------------------------------------------------


class TestDerivedFeatureTransformer:
    def test_fit_returns_self(self, transformer, sample_df):
        result = transformer.fit(sample_df)
        assert result is transformer

    def test_transform_adds_log_floor_area(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        assert "log_floor_area" in out.columns

    def test_transform_adds_floor_area_per_room(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        assert "floor_area_per_room" in out.columns

    def test_transform_adds_energy_rating_numeric(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        assert "energy_rating_numeric" in out.columns

    def test_log_floor_area_is_log1p(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        expected = np.log1p(sample_df["total_floor_area"].clip(lower=0))
        np.testing.assert_allclose(out["log_floor_area"].values, expected.values, rtol=1e-6)

    def test_floor_area_per_room_formula(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        expected = sample_df["total_floor_area"] / sample_df["number_habitable_rooms"].clip(lower=1)
        np.testing.assert_allclose(out["floor_area_per_room"].values, expected.values, rtol=1e-6)

    def test_energy_rating_numeric_values(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        for letter, numeric in ENERGY_RATING_MAP.items():
            mask = sample_df["current_energy_rating"] == letter
            if mask.any():
                assert (out.loc[mask, "energy_rating_numeric"] == numeric).all()

    def test_does_not_mutate_input(self, transformer, sample_df):
        original_cols = set(sample_df.columns)
        transformer.fit_transform(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_passthrough_original_columns(self, transformer, sample_df):
        out = transformer.fit_transform(sample_df)
        for col in sample_df.columns:
            assert col in out.columns

    def test_missing_source_column_skips_gracefully(self, transformer):
        """When source columns are absent the derived columns are simply omitted."""
        df = pd.DataFrame({"price": [200_000.0, 300_000.0]})
        out = transformer.fit_transform(df)
        assert "log_floor_area" not in out.columns
        assert "floor_area_per_room" not in out.columns
        assert "energy_rating_numeric" not in out.columns

    def test_zero_floor_area_produces_log1p_of_zero(self, transformer):
        df = pd.DataFrame(
            {
                "total_floor_area": [0.0, 50.0],
                "number_habitable_rooms": [3.0, 3.0],
            }
        )
        out = transformer.fit_transform(df)
        assert out["log_floor_area"].iloc[0] == pytest.approx(0.0)  # log1p(0) == 0

    def test_zero_rooms_clipped_to_one(self, transformer):
        df = pd.DataFrame(
            {
                "total_floor_area": [100.0],
                "number_habitable_rooms": [0.0],
            }
        )
        out = transformer.fit_transform(df)
        assert out["floor_area_per_room"].iloc[0] == pytest.approx(100.0)  # 100 / 1

    def test_unknown_energy_rating_produces_nan(self, transformer):
        df = pd.DataFrame(
            {
                "current_energy_rating": ["X", "A"],
            }
        )
        out = transformer.fit_transform(df)
        assert pd.isna(out["energy_rating_numeric"].iloc[0])
        assert out["energy_rating_numeric"].iloc[1] == ENERGY_RATING_MAP["A"]

    def test_all_derived_numeric_features_in_constant(self):
        assert DERIVED_NUMERIC_FEATURES == [
            "log_floor_area",
            "floor_area_per_room",
            "energy_rating_numeric",
        ]


# ---------------------------------------------------------------------------
# Integration: derived features in FeatureConfig pipeline
# ---------------------------------------------------------------------------


class TestDerivedFeaturesInPipeline:
    def test_build_feature_pipeline_with_derived(self):
        """Pipeline with derived_features=True should include a 'derive' step."""
        from sklearn.pipeline import Pipeline

        from src.features.build_features import (
            FeatureConfig,
            MissingStrategy,
            build_feature_pipeline,
        )

        config = FeatureConfig(
            numeric_features=["total_floor_area", "number_habitable_rooms"],
            categorical_features=["property_type"],
            target_encode_features=["district"],
            missing_strategy=MissingStrategy.impute,
            derived_features=True,
        )
        pipeline = build_feature_pipeline(config)
        assert isinstance(pipeline, Pipeline)
        assert "derive" in pipeline.named_steps

    def test_derived_pipeline_fit_transform(self):
        """Pipeline with derived=True should transform without errors."""
        import numpy as np

        from src.features.build_features import (
            FeatureConfig,
            MissingStrategy,
            build_feature_pipeline,
        )

        config = FeatureConfig(
            numeric_features=["total_floor_area", "number_habitable_rooms"],
            categorical_features=["property_type"],
            target_encode_features=["district"],
            missing_strategy=MissingStrategy.impute,
            derived_features=True,
        )
        pipeline = build_feature_pipeline(config)
        rng = np.random.default_rng(1)
        n = 50
        df = pd.DataFrame(
            {
                "total_floor_area": rng.uniform(30, 250, n),
                "number_habitable_rooms": rng.integers(1, 6, n).astype(float),
                "property_type": rng.choice(["D", "S", "T"], n),
                "district": rng.choice(["Westminster", "Lambeth"], n),
                "current_energy_rating": rng.choice(["A", "B", "C"], n),
            }
        )
        y = rng.uniform(100_000, 800_000, n)
        out = pipeline.fit_transform(df, y)
        assert out.shape[0] == n

    def test_effective_numeric_features_includes_derived(self):
        from src.features.build_features import FeatureConfig
        from src.features.derived import DERIVED_NUMERIC_FEATURES

        config = FeatureConfig(derived_features=True)
        assert all(f in config.effective_numeric_features for f in DERIVED_NUMERIC_FEATURES)

    def test_effective_numeric_features_without_derived(self):
        from src.features.build_features import DEFAULT_NUMERIC_FEATURES, FeatureConfig

        config = FeatureConfig(derived_features=False)
        assert config.effective_numeric_features == DEFAULT_NUMERIC_FEATURES

    def test_all_features_does_not_include_derived(self):
        """all_features returns only the *raw* columns for validation / loading."""
        from src.features.build_features import FeatureConfig
        from src.features.derived import DERIVED_NUMERIC_FEATURES

        config = FeatureConfig(derived_features=True)
        for feat in DERIVED_NUMERIC_FEATURES:
            assert feat not in config.all_features
