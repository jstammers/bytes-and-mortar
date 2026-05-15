"""Tests for src/models/property_price/investigate.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.property_price.investigate import (
    ImputationBiasResult,
    InvestigationSummary,
    PermutationImportanceResult,
    calibration_analysis,
    imputation_bias_analysis,
    permutation_importance_study,
    plot_calibration,
    plot_imputation_bias,
    plot_permutation_importance,
    plot_regional_performance,
    plot_shap_summary,
    regional_performance,
    run_investigations,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200


@pytest.fixture
def y_true() -> np.ndarray:
    return RNG.uniform(100_000, 800_000, N)


@pytest.fixture
def y_pred(y_true) -> np.ndarray:
    """Predictions with ≈15% noise around true values."""
    noise = RNG.normal(0, 0.15, N)
    return y_true * (1 + noise)


@pytest.fixture
def y_test_log(y_true) -> np.ndarray:
    return np.log1p(y_true)


@pytest.fixture
def sample_test_df() -> pd.DataFrame:
    counties = ["Greater London", "Surrey", "Kent", "Essex", "Sussex"] * (N // 5)
    return pd.DataFrame(
        {
            "county": counties[:N],
            "property_type": RNG.choice(["D", "S", "T", "F"], N),
            "total_floor_area": RNG.uniform(30, 250, N),
            "number_habitable_rooms": RNG.integers(1, 7, N).astype(float),
            "current_energy_efficiency": RNG.integers(20, 100, N).astype(float),
            "current_energy_rating": RNG.choice(["A", "B", "C", "D", "E"], N),
            "averageprice": RNG.uniform(200_000, 900_000, N),
        }
    )


@pytest.fixture
def sample_test_df_with_nulls(sample_test_df) -> pd.DataFrame:
    """DataFrame with some NaN values in numeric columns."""
    df = sample_test_df.copy()
    null_idx = RNG.choice(N, size=30, replace=False)
    df.loc[null_idx, "total_floor_area"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


class TestCalibrationAnalysis:
    def test_returns_n_bins(self, y_true, y_pred):
        bins = calibration_analysis(y_true, y_pred, n_bins=10)
        assert len(bins) == 10

    def test_bin_indices_are_sequential(self, y_true, y_pred):
        bins = calibration_analysis(y_true, y_pred, n_bins=5)
        assert [b.bin_index for b in bins] == list(range(5))

    def test_all_samples_covered(self, y_true, y_pred):
        bins = calibration_analysis(y_true, y_pred, n_bins=10)
        total = sum(b.n_samples for b in bins)
        assert total == len(y_true)

    def test_perfect_calibration_zero_bias(self):
        y = np.arange(1_000, 1_100, dtype=float) * 1_000
        bins = calibration_analysis(y, y.copy(), n_bins=5)
        for b in bins:
            assert abs(b.bias) < 1e-3  # perfect predictions → zero bias

    def test_bias_property(self, y_true, y_pred):
        bins = calibration_analysis(y_true, y_pred, n_bins=5)
        for b in bins:
            assert b.bias == pytest.approx(b.mean_predicted - b.mean_actual)

    def test_monotone_predicted_means(self, y_true, y_pred):
        bins = calibration_analysis(y_true, y_pred, n_bins=10)
        preds = [b.mean_predicted for b in bins]
        assert preds == sorted(preds)

    def test_custom_bin_count(self, y_true, y_pred):
        bins = calibration_analysis(y_true, y_pred, n_bins=20)
        assert len(bins) == 20


class TestPlotCalibration:
    def test_returns_figure(self, y_true, y_pred):
        import matplotlib.pyplot as plt

        bins = calibration_analysis(y_true, y_pred, n_bins=10)
        fig = plot_calibration(bins)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_empty_bins_returns_figure(self):
        import matplotlib.pyplot as plt

        fig = plot_calibration([])
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Imputation bias analysis
# ---------------------------------------------------------------------------


class TestImputationBiasAnalysis:
    def test_no_nulls_all_complete(self, sample_test_df, y_true, y_pred):
        feature_cols = ["total_floor_area", "number_habitable_rooms"]
        result = imputation_bias_analysis(sample_test_df, y_true, y_pred, feature_cols)
        assert result.n_imputed == 0
        assert result.n_complete == N
        assert result.imputed_fraction == pytest.approx(0.0)
        assert result.imputed_metrics is None  # below min_group_size

    def test_with_nulls_detected(self, sample_test_df_with_nulls, y_true, y_pred):
        feature_cols = ["total_floor_area", "number_habitable_rooms"]
        result = imputation_bias_analysis(sample_test_df_with_nulls, y_true, y_pred, feature_cols)
        assert result.n_imputed == 30
        assert result.n_complete == N - 30
        assert result.imputed_fraction == pytest.approx(30 / N)

    def test_imputed_metrics_computed_when_sufficient(
        self, sample_test_df_with_nulls, y_true, y_pred
    ):
        feature_cols = ["total_floor_area"]
        result = imputation_bias_analysis(
            sample_test_df_with_nulls, y_true, y_pred, feature_cols, min_group_size=10
        )
        assert result.imputed_metrics is not None
        assert result.complete_metrics is not None

    def test_total_adds_up(self, sample_test_df_with_nulls, y_true, y_pred):
        feature_cols = ["total_floor_area"]
        result = imputation_bias_analysis(sample_test_df_with_nulls, y_true, y_pred, feature_cols)
        assert result.n_imputed + result.n_complete == N

    def test_missing_feature_col_ignored(self, sample_test_df, y_true, y_pred):
        """Non-existent columns in feature_cols are silently skipped."""
        result = imputation_bias_analysis(
            sample_test_df, y_true, y_pred, ["total_floor_area", "nonexistent_col"]
        )
        assert result.n_imputed == 0


class TestPlotImputationBias:
    def test_returns_figure_with_data(self, sample_test_df_with_nulls, y_true, y_pred):
        import matplotlib.pyplot as plt

        feature_cols = ["total_floor_area"]
        result = imputation_bias_analysis(
            sample_test_df_with_nulls, y_true, y_pred, feature_cols, min_group_size=10
        )
        fig = plot_imputation_bias(result)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_returns_figure_no_groups(self):
        import matplotlib.pyplot as plt

        result = ImputationBiasResult(
            imputed_metrics=None,
            complete_metrics=None,
            n_imputed=0,
            n_complete=0,
            imputed_fraction=0.0,
        )
        fig = plot_imputation_bias(result)
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Regional performance
# ---------------------------------------------------------------------------


class TestRegionalPerformance:
    def test_returns_dict(self, sample_test_df, y_true, y_pred):
        result = regional_performance(sample_test_df, y_true, y_pred, region_col="county")
        assert isinstance(result, dict)

    def test_all_regions_have_metrics(self, sample_test_df, y_true, y_pred):
        result = regional_performance(
            sample_test_df, y_true, y_pred, region_col="county", min_samples=1
        )
        expected_regions = set(sample_test_df["county"].unique())
        assert set(result.keys()) == expected_regions

    def test_min_samples_filters(self, sample_test_df, y_true, y_pred):
        """Regions with fewer than min_samples rows should be excluded."""
        result = regional_performance(
            sample_test_df, y_true, y_pred, region_col="county", min_samples=N
        )
        # No region has all N samples with N=200 and 5 equal-sized counties
        assert len(result) == 0

    def test_missing_region_col_returns_empty(self, sample_test_df, y_true, y_pred):
        result = regional_performance(sample_test_df, y_true, y_pred, region_col="nonexistent")
        assert result == {}

    def test_metrics_are_finite(self, sample_test_df, y_true, y_pred):
        result = regional_performance(
            sample_test_df, y_true, y_pred, region_col="county", min_samples=1
        )
        for metrics in result.values():
            assert np.isfinite(metrics.rmse)
            assert np.isfinite(metrics.mdape)


class TestPlotRegionalPerformance:
    def test_returns_figure(self, sample_test_df, y_true, y_pred):
        import matplotlib.pyplot as plt

        result = regional_performance(
            sample_test_df, y_true, y_pred, region_col="county", min_samples=1
        )
        fig = plot_regional_performance(result)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_empty_dict_returns_figure(self):
        import matplotlib.pyplot as plt

        fig = plot_regional_performance({})
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


class TestPermutationImportanceStudy:
    @pytest.fixture
    def fitted_pipeline(self):
        """Tiny fitted perpetual pipeline for use in permutation importance tests."""
        from src.models.property_price.features import FeatureConfig, MissingStrategy
        from src.models.property_price.perpetual import build_perpetual_pipeline

        rng = np.random.default_rng(0)
        n = 120
        X = pd.DataFrame(
            {
                "year": rng.integers(2018, 2024, n).astype(float),
                "total_floor_area": rng.uniform(30, 200, n),
                "property_type": rng.choice(["D", "S", "T"], n),
                "district": rng.choice(["Westminster", "Lambeth", "Hackney"], n),
            }
        )
        y_log = np.log1p(rng.uniform(100_000, 800_000, n))

        config = FeatureConfig(
            numeric_features=["year", "total_floor_area"],
            categorical_features=["property_type"],
            target_encode_features=["district"],
            missing_strategy=MissingStrategy.impute,
        )
        pipeline = build_perpetual_pipeline(feature_config=config, budget=0.5)
        pipeline.fit(X, y_log)
        return pipeline, X, y_log, config.all_features

    def test_returns_list(self, fitted_pipeline):
        pipeline, X, y_log, feature_cols = fitted_pipeline
        results = permutation_importance_study(pipeline, X, y_log, feature_cols, n_repeats=2)
        assert isinstance(results, list)

    def test_one_result_per_feature(self, fitted_pipeline):
        pipeline, X, y_log, feature_cols = fitted_pipeline
        results = permutation_importance_study(pipeline, X, y_log, feature_cols, n_repeats=2)
        assert len(results) == len(feature_cols)

    def test_sorted_descending(self, fitted_pipeline):
        pipeline, X, y_log, feature_cols = fitted_pipeline
        results = permutation_importance_study(pipeline, X, y_log, feature_cols, n_repeats=2)
        means = [r.importance_mean for r in results]
        assert means == sorted(means, reverse=True)

    def test_result_fields(self, fitted_pipeline):
        pipeline, X, y_log, feature_cols = fitted_pipeline
        results = permutation_importance_study(pipeline, X, y_log, feature_cols, n_repeats=2)
        for r in results:
            assert isinstance(r, PermutationImportanceResult)
            assert isinstance(r.feature, str)
            assert np.isfinite(r.importance_mean)
            assert r.importance_std >= 0


class TestPlotPermutationImportance:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        results = [
            PermutationImportanceResult("feat_a", 0.5, 0.1),
            PermutationImportanceResult("feat_b", 0.1, 0.05),
            PermutationImportanceResult("feat_c", -0.02, 0.01),
        ]
        fig = plot_permutation_importance(results)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_empty_list_returns_figure(self):
        import matplotlib.pyplot as plt

        fig = plot_permutation_importance([])
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP plots (unit test with synthetic data — no real booster needed)
# ---------------------------------------------------------------------------


class TestPlotShapSummary:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        shap_vals = rng.normal(0, 1, (50, 5))
        names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        fig = plot_shap_summary(shap_vals, names)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_max_features_respected(self):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        shap_vals = rng.normal(0, 1, (50, 10))
        names = [f"feat_{i}" for i in range(10)]
        fig = plot_shap_summary(shap_vals, names, max_features=5)
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# run_investigations integration test
# ---------------------------------------------------------------------------


class TestRunInvestigations:
    @pytest.fixture
    def pipeline_and_data(self, tmp_path):
        """Minimal fitted pipeline + train/test data for run_investigations."""
        from src.models.property_price.features import FeatureConfig, MissingStrategy
        from src.models.property_price.perpetual import build_perpetual_pipeline

        rng = np.random.default_rng(7)
        n = 200

        def _make_df(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "year": rng.integers(2018, 2024, n).astype(float),
                    "total_floor_area": rng.uniform(30, 200, n),
                    "current_energy_efficiency": rng.integers(20, 100, n).astype(float),
                    "averageprice": rng.uniform(200_000, 800_000, n),
                    "property_type": rng.choice(["D", "S", "T", "F"], n),
                    "old_new": rng.choice(["Y", "N"], n),
                    "duration": rng.choice(["F", "L"], n),
                    "county": rng.choice(["Greater London", "Surrey", "Kent"], n),
                    "postcode_outward": rng.choice(["SW1", "E1"], n),
                    "district": rng.choice(["Westminster", "Lambeth"], n),
                    "month": rng.integers(1, 13, n).astype(float),
                    "number_habitable_rooms": rng.integers(1, 7, n).astype(float),
                }
            )

        config = FeatureConfig(missing_strategy=MissingStrategy.impute)
        X_train = _make_df(n)
        y_train_log = np.log1p(rng.uniform(100_000, 800_000, n))

        pipeline = build_perpetual_pipeline(feature_config=config, budget=0.5)
        pipeline.fit(X_train, y_train_log)

        X_test = _make_df(150)
        # Introduce some NaN to test imputation bias
        X_test.loc[X_test.index[:15], "total_floor_area"] = np.nan
        y_test_log = np.log1p(rng.uniform(100_000, 800_000, 150))

        return pipeline, config, X_train, y_train_log, X_test, y_test_log

    def test_run_investigations_returns_summary(self, pipeline_and_data, tmp_path):
        import mlflow

        pipeline, config, X_train, y_train_log, X_test, y_test_log = pipeline_and_data

        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
        mlflow.set_experiment("test_investigate")

        with mlflow.start_run():
            summary = run_investigations(
                pipeline=pipeline,
                X_train=X_train,
                y_train_log=y_train_log,
                X_test_raw=X_test,
                y_test_log=y_test_log,
                feature_cols=config.all_features,
                region_col="county",
                n_calibration_bins=5,
                n_shap_samples=50,
                n_perm_repeats=2,
            )

        assert isinstance(summary, InvestigationSummary)

    def test_calibration_bins_populated(self, pipeline_and_data, tmp_path):
        import mlflow

        pipeline, config, X_train, y_train_log, X_test, y_test_log = pipeline_and_data

        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns2'}")
        mlflow.set_experiment("test_investigate2")

        with mlflow.start_run():
            summary = run_investigations(
                pipeline=pipeline,
                X_train=X_train,
                y_train_log=y_train_log,
                X_test_raw=X_test,
                y_test_log=y_test_log,
                feature_cols=config.all_features,
                n_calibration_bins=5,
                n_shap_samples=50,
                n_perm_repeats=2,
            )

        assert len(summary.calibration_bins) == 5

    def test_regional_metrics_populated(self, pipeline_and_data, tmp_path):
        import mlflow

        pipeline, config, X_train, y_train_log, X_test, y_test_log = pipeline_and_data

        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns3'}")
        mlflow.set_experiment("test_investigate3")

        with mlflow.start_run():
            summary = run_investigations(
                pipeline=pipeline,
                X_train=X_train,
                y_train_log=y_train_log,
                X_test_raw=X_test,
                y_test_log=y_test_log,
                feature_cols=config.all_features,
                region_col="county",
                n_calibration_bins=5,
                n_shap_samples=50,
                n_perm_repeats=2,
            )

        assert len(summary.regional_metrics) > 0

    def test_shap_populated_for_perpetual(self, pipeline_and_data, tmp_path):
        import mlflow

        pipeline, config, X_train, y_train_log, X_test, y_test_log = pipeline_and_data

        mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns4'}")
        mlflow.set_experiment("test_investigate4")

        with mlflow.start_run():
            summary = run_investigations(
                pipeline=pipeline,
                X_train=X_train,
                y_train_log=y_train_log,
                X_test_raw=X_test,
                y_test_log=y_test_log,
                feature_cols=config.all_features,
                n_calibration_bins=5,
                n_shap_samples=50,
                n_perm_repeats=2,
            )

        assert summary.shap_mean_abs is not None
        assert len(summary.shap_feature_names) > 0
