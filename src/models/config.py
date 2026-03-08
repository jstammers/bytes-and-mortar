"""Configuration dataclasses for UK property price model training.

The ``Experiment`` dataclass is the single source of truth for a training run.
It is constructed by the CLI and threaded through to every component, making
each MLflow run fully reproducible from its logged params.

Usage::

    from src.models.config import Experiment, PerpetualConfig, ModelType

    experiment = Experiment(
        model_type=ModelType.perpetual,
        perpetual_config=PerpetualConfig(budget=1.0),
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
    """Time-series cross-validation fold strategy.

    .. deprecated::
        CV is no longer needed in the default Perpetual training path.
        Retained for backward compatibility and direct use of
        :func:`src.models.cv.make_cv_splits`.
    """

    sliding_window = "sliding_window"
    expanding_window = "expanding_window"
    year_based = "year_based"


class ModelType(StrEnum):
    """Supported model architectures.

    The recommended model is ``perpetual`` — it matches XGBoost+Optuna
    accuracy with a single ``budget`` parameter and no HPO loop.
    """

    perpetual = "perpetual"
    linear = "linear"
    xgboost = "xgboost"  # deprecated — use perpetual


@dataclass
class PerpetualConfig:
    """Configuration for Perpetual GBM — no HPO required.

    Perpetual automatically determines the number of trees from ``budget``.
    Increase ``budget`` for better accuracy at the cost of training time.

    Budget reference (10% of full UK property dataset, 2024 test set):

        ======  ======  ========  =======
        Budget  MdAPE   RMSE (£)  Time
        ======  ======  ========  =======
        0.5     16.7%   £290,172  16s
        0.7     16.2%   £287,186  20s
        1.0     16.1%   £288,002  33s
        ======  ======  ========  =======

    XGBoost+Optuna (25 trials) achieved MdAPE 16.0% / RMSE £284,588 in 6.4min.

    Attributes:
        budget: Complexity control. ``1.0`` is the recommended default.
        objective: Loss function. ``"SquaredLoss"`` for regression.
    """

    budget: float = 1.0
    objective: str = "SquaredLoss"


@dataclass
class CVConfig:
    """Cross-validation configuration.

    .. deprecated::
        CV is no longer used by the Perpetual or Ridge training paths.
        Retained for backward compatibility.

    Attributes:
        strategy: Split strategy (sliding, expanding, or year-based).
        n_splits: Number of folds.
        gap_months: Months to skip between train end and validation start.
        window_months: Training window size in months for ``sliding_window``.
        holdout_years: Explicit years to hold out for ``year_based`` splits.
    """

    strategy: CVStrategy = CVStrategy.sliding_window
    n_splits: int = 5
    gap_months: int = 1
    window_months: int | None = 24
    holdout_years: list[int] | None = None


@dataclass
class HPOConfig:
    """Optuna hyperparameter optimisation configuration.

    .. deprecated::
        HPO is no longer part of the default training pipeline.
        Use :class:`PerpetualConfig` with a ``budget`` value instead.

    Attributes:
        n_trials: Number of Optuna trials to run.
        timeout_seconds: Hard time limit across all trials.
        metric: CV metric minimised by Optuna.
        sampler: Optuna sampler name.
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
        model_type: Which model architecture to train. Defaults to
            ``ModelType.perpetual`` — a self-tuning GBM requiring no HPO.
        feature_config: Feature engineering options.
        perpetual_config: Perpetual GBM settings. Ignored for other model types.
        cv_config: Deprecated. Ignored by the Perpetual training path.
        hpo_config: Deprecated. Ignored by the Perpetual training path.
        data_path: Path to the processed parquet file.
        nrows: Limit rows loaded. ``None`` loads everything.
        test_years: Calendar years held out as the final test set.
        stratify_by: Column used to log class distribution after the split.
        mlflow_tracking_uri: MLflow tracking URI.
        register_model: Whether to push the best model to the MLflow registry.
        model_name: Registered model name in the MLflow registry.
        models_dir: Local directory for serialised model artefacts.
    """

    name: str = "uk_property_price"
    model_type: ModelType = ModelType.perpetual
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    perpetual_config: PerpetualConfig = field(default_factory=PerpetualConfig)

    # Deprecated: retained for backward compatibility
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
        }
        if self.model_type == ModelType.perpetual:
            params["perpetual_budget"] = str(self.perpetual_config.budget)
            params["perpetual_objective"] = self.perpetual_config.objective
        for k, v in self.feature_config.to_dict().items():
            params[f"feat_{k}"] = str(v)
        return params
