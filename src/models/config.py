"""Configuration dataclasses for UK property price model training.

The ``Experiment`` dataclass is the single source of truth for a training run.
It is constructed by the CLI and threaded through to every component, making
each MLflow run fully reproducible from its logged params.

Usage::

    from src.models.config import Experiment, CVConfig, HPOConfig, ModelType

    experiment = Experiment(
        model_type=ModelType.xgboost,
        cv_config=CVConfig(strategy=CVStrategy.sliding_window, n_splits=5),
        hpo_config=HPOConfig(n_trials=50),
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path  # noqa: TC003

from src.data.config import MLRUNS_DIR, MODELS_DIR, PROCESSED_DATA_PATH
from src.features.build_features import FeatureConfig


class CVStrategy(StrEnum):
    """Time-series cross-validation fold strategy."""

    sliding_window = "sliding_window"
    expanding_window = "expanding_window"
    year_based = "year_based"


class ModelType(StrEnum):
    """Supported model architectures."""

    linear = "linear"
    xgboost = "xgboost"


@dataclass
class CVConfig:
    """Cross-validation configuration.

    Attributes:
        strategy: Split strategy (sliding, expanding, or year-based).
        n_splits: Number of folds. For ``year_based`` this is overridden by the
            number of available years minus one.
        gap_months: Months to skip between train end and validation start.
            Prevents target leakage from slow-to-register transactions.
        window_months: Training window size in months for ``sliding_window``.
            ``None`` uses all available history (equivalent to expanding window).
        holdout_years: Explicit years to hold out for ``year_based`` splits.
            ``None`` auto-selects the last ``n_splits`` years.
    """

    strategy: CVStrategy = CVStrategy.sliding_window
    n_splits: int = 5
    gap_months: int = 1
    window_months: int | None = 24
    holdout_years: list[int] | None = None


@dataclass
class HPOConfig:
    """Optuna hyperparameter optimisation configuration.

    Attributes:
        n_trials: Number of Optuna trials to run.
        timeout_seconds: Hard time limit across all trials. ``None`` = no limit.
        metric: CV metric that Optuna minimises (always RMSE in log-price space).
        sampler: Optuna sampler name. ``"tpe"`` (default) is the Tree-structured
            Parzen Estimator; ``"random"`` is useful for debugging.
    """

    n_trials: int = 50
    timeout_seconds: int | None = None
    metric: str = "rmse"
    sampler: str = "tpe"


@dataclass
class Experiment:
    """Full configuration for a single training run.

    Construct via the CLI or directly in code. Pass to ``train()`` as-is.
    All fields have defaults so the dataclass is usable with zero arguments.

    Attributes:
        name: Human-readable experiment name (used as MLflow experiment name).
        model_type: Which model architecture to train.
        feature_config: Feature engineering options.
        cv_config: Cross-validation fold strategy and parameters.
        hpo_config: Hyperparameter optimisation settings.
        data_path: Path to the processed parquet file.
        nrows: Limit rows loaded. ``None`` loads everything. Useful for dev.
        test_years: Calendar years held out as the final test set. These rows
            are never seen during HPO or CV.
        stratify_by: Column used to check class distribution balance after the
            temporal train/test split (informational — not enforced).
        mlflow_tracking_uri: MLflow tracking URI. Defaults to
            ``MLFLOW_TRACKING_URI`` env var, then ``mlruns/`` directory.
        register_model: Whether to push the best model to the MLflow registry.
        model_name: Registered model name in the MLflow registry.
        models_dir: Local directory for serialised model artefacts.
    """

    name: str = "uk_property_price"
    model_type: ModelType = ModelType.xgboost
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    cv_config: CVConfig = field(default_factory=CVConfig)
    hpo_config: HPOConfig = field(default_factory=HPOConfig)

    # Data
    data_path: Path = field(default_factory=lambda: PROCESSED_DATA_PATH)
    nrows: int | None = None

    # Split
    test_years: list[int] = field(default_factory=lambda: [2024])
    stratify_by: str | None = "property_type"

    # MLflow
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}")
    )
    register_model: bool = False
    model_name: str = "uk_property_price"

    # Output
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)

    def to_mlflow_params(self) -> dict[str, str]:
        """Flatten all config into a ``dict[str, str]`` for ``mlflow.log_params()``.

        Returns:
            Flat string mapping of all experiment parameters.
        """
        params: dict[str, str] = {
            "experiment_name": self.name,
            "model_type": self.model_type.value,
            "data_path": str(self.data_path),
            "nrows": str(self.nrows),
            "test_years": ",".join(str(y) for y in self.test_years),
            "stratify_by": str(self.stratify_by),
            "register_model": str(self.register_model),
            # CV config
            "cv_strategy": self.cv_config.strategy.value,
            "cv_n_splits": str(self.cv_config.n_splits),
            "cv_gap_months": str(self.cv_config.gap_months),
            "cv_window_months": str(self.cv_config.window_months),
            # HPO config
            "hpo_n_trials": str(self.hpo_config.n_trials),
            "hpo_timeout_seconds": str(self.hpo_config.timeout_seconds),
            "hpo_sampler": self.hpo_config.sampler,
        }
        # Feature config (already returns str values)
        for k, v in self.feature_config.to_dict().items():
            params[f"feat_{k}"] = str(v)
        return params
