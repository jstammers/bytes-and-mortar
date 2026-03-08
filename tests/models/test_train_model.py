"""Integration tests for the train/predict orchestrator.

These tests use a small synthetic dataset so they run quickly without
requiring real processed data or a GPU.  MLflow is pointed at a temporary
directory to avoid polluting the project's mlruns/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from src.features.build_features import FeatureConfig, MissingStrategy
from src.models.config import Experiment, ModelType, PerpetualConfig

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df(tmp_path: Path) -> tuple[Path, pl.DataFrame]:
    """500-row synthetic property dataset saved as parquet."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pl.date_range(
        start=pl.date(2020, 1, 1),
        end=pl.date(2020, 1, 1) + pl.duration(days=n * 3 - 1),
        interval="3d",
        eager=True,
    )

    df = pl.DataFrame(
        {
            "date_of_transfer": dates,
            "price": rng.integers(150_000, 800_000, n).astype(float),
            "property_type": rng.choice(["D", "S", "T", "F"], n),
            "old_new": rng.choice(["Y", "N"], n),
            "duration": rng.choice(["F", "L"], n),
            "county": rng.choice(["Greater London", "Surrey"], n),
            "postcode_outward": rng.choice(["SW1", "E1", "N1"], n),
            "current_energy_rating": rng.choice(["A", "B", "C", "D", "E"], n),
            "construction_age_band": rng.choice(["pre-1900", "1950s", "2000s"], n),
            "district": rng.choice(["Westminster", "Lambeth", "Islington"], n),
            "year": dates.dt.year(),
            "month": dates.dt.month(),
            "current_energy_efficiency": rng.integers(20, 100, n).astype(float),
            "total_floor_area": rng.uniform(30, 250, n),
            "number_habitable_rooms": rng.integers(1, 7, n).astype(float),
            "averageprice": rng.uniform(200_000, 900_000, n),
            "index": rng.uniform(100, 200, n),
        }
    )

    parquet_path = tmp_path / "uk_property_sales.parquet"
    df.write_parquet(parquet_path)
    return parquet_path, df


@pytest.fixture
def base_experiment(synthetic_df, tmp_path: Path) -> Experiment:
    """Minimal Experiment config pointing at the synthetic dataset."""
    parquet_path, _ = synthetic_df
    return Experiment(
        model_type=ModelType.perpetual,
        feature_config=FeatureConfig(
            numeric_features=[
                "year",
                "month",
                "total_floor_area",
                "current_energy_efficiency",
            ],
            categorical_features=["property_type", "old_new"],
            target_encode_features=["district"],
            missing_strategy=MissingStrategy.impute,
        ),
        perpetual_config=PerpetualConfig(budget=0.5),
        data_path=parquet_path,
        test_years=[2023],
        register_model=False,
        mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
        models_dir=tmp_path / "models",
    )


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_perpetual_returns_pipeline_and_metrics(self, base_experiment):
        from sklearn.pipeline import Pipeline

        from src.models.evaluate import RegressionMetrics
        from src.models.train_model import train

        pipeline, metrics = train(base_experiment)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(metrics, RegressionMetrics)

    def test_train_linear_returns_pipeline(self, base_experiment, tmp_path):
        from sklearn.pipeline import Pipeline

        from src.models.train_model import train

        experiment = Experiment(
            model_type=ModelType.linear,
            feature_config=FeatureConfig(
                numeric_features=["year", "month", "total_floor_area"],
                categorical_features=["property_type"],
                target_encode_features=["district"],
                missing_strategy=MissingStrategy.impute,
            ),
            data_path=base_experiment.data_path,
            test_years=[2023],
            register_model=False,
            mlflow_tracking_uri=base_experiment.mlflow_tracking_uri,
            models_dir=tmp_path / "models_linear",
        )
        pipeline, _ = train(experiment)
        assert isinstance(pipeline, Pipeline)

    def test_metrics_have_finite_values(self, base_experiment):
        from src.models.train_model import train

        _, metrics = train(base_experiment)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.r2)

    def test_model_saved_locally(self, base_experiment):
        from src.models.train_model import train

        train(base_experiment)
        local_model = base_experiment.models_dir / f"{base_experiment.model_type}_best.joblib"
        assert local_model.exists()

    def test_mlflow_run_created(self, base_experiment):
        import mlflow

        from src.models.train_model import train

        train(base_experiment)

        mlflow.set_tracking_uri(base_experiment.mlflow_tracking_uri)
        experiment_name = f"{base_experiment.name}/{base_experiment.model_type}"
        mlflow_exp = mlflow.get_experiment_by_name(experiment_name)
        assert mlflow_exp is not None

        runs = mlflow.search_runs(experiment_ids=[mlflow_exp.experiment_id])
        assert len(runs) >= 1

    def test_missing_data_file_raises(self, base_experiment, tmp_path):
        from src.models.train_model import train

        bad_experiment = Experiment(
            model_type=base_experiment.model_type,
            feature_config=base_experiment.feature_config,
            perpetual_config=base_experiment.perpetual_config,
            data_path=tmp_path / "does_not_exist.parquet",
            test_years=base_experiment.test_years,
            register_model=False,
            mlflow_tracking_uri=base_experiment.mlflow_tracking_uri,
            models_dir=base_experiment.models_dir,
        )
        with pytest.raises(FileNotFoundError):
            train(bad_experiment)

    @pytest.mark.parametrize("strategy", list(MissingStrategy))
    def test_all_missing_strategies(self, synthetic_df, tmp_path, strategy):
        """All three missing strategies should complete without error."""
        from sklearn.pipeline import Pipeline

        from src.models.train_model import train

        parquet_path, _ = synthetic_df
        experiment = Experiment(
            model_type=ModelType.perpetual,
            feature_config=FeatureConfig(
                numeric_features=["year", "total_floor_area"],
                categorical_features=["property_type"],
                target_encode_features=["district"],
                missing_strategy=strategy,
            ),
            perpetual_config=PerpetualConfig(budget=0.5),
            data_path=parquet_path,
            test_years=[2023],
            register_model=False,
            mlflow_tracking_uri=f"file://{tmp_path / f'mlruns_missing_{strategy}'}",
            models_dir=tmp_path / f"models_missing_{strategy}",
        )
        pipeline, _ = train(experiment)
        assert isinstance(pipeline, Pipeline)

    def test_xgboost_emits_deprecation_warning(self, base_experiment, tmp_path):
        """Using ModelType.xgboost should emit a DeprecationWarning."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        pytest.importorskip("optuna", reason="optuna not installed")

        from src.models.train_model import train

        experiment = Experiment(
            model_type=ModelType.xgboost,
            feature_config=FeatureConfig(
                numeric_features=["year", "month", "total_floor_area"],
                categorical_features=["property_type"],
                target_encode_features=["district"],
                missing_strategy=MissingStrategy.impute,
            ),
            data_path=base_experiment.data_path,
            test_years=[2023],
            register_model=False,
            mlflow_tracking_uri=f"file://{tmp_path / 'mlruns_xgb'}",
            models_dir=tmp_path / "models_xgb",
        )
        with pytest.warns(DeprecationWarning, match="xgboost"):
            train(experiment)
