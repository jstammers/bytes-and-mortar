"""Linear regression model for UK property price prediction.

Uses Ridge regression (L2 regularisation) on log1p-transformed prices.
The log transformation maps the right-skewed price distribution to
approximate normality, making linear assumptions more defensible.

Sklearn pipeline structure::

    FeaturePipeline | Ridge(alpha=...)

Hyperparameter search space (Optuna):
    alpha: Log-uniform [1e-3, 1e3]

Usage::

    from src.models.linear import build_linear_pipeline, linear_objective

    pipeline = build_linear_pipeline(alpha=10.0)
    pipeline.fit(X_train, y_train_log)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features.build_features import FeatureConfig, build_feature_pipeline

if TYPE_CHECKING:
    import optuna
    import pandas as pd

logger = logging.getLogger(__name__)


def build_linear_pipeline(
    feature_config: FeatureConfig | None = None,
    alpha: float = 1.0,
) -> Pipeline:
    """Return an unfitted Ridge regression pipeline.

    The pipeline is:
        1. ``FeaturePipeline`` (ColumnTransformer with imputation + encoding + scaling)
        2. ``Ridge`` with the given ``alpha``

    Ridge is preferred over plain OLS because: (a) property data has many
    correlated features, (b) regularisation prevents overfitting on smaller
    regional datasets.

    The target must be log1p-transformed *before* calling ``fit``.  The pipeline
    itself does not transform y.  Callers should apply ``np.expm1`` to undo the
    log when converting predictions back to GBP.

    Args:
        feature_config: Feature engineering config.  Defaults to
            ``FeatureConfig(missing_strategy=MissingStrategy.impute)``
            since Ridge cannot handle NaN inputs.
        alpha: Ridge regularisation strength.  Higher values = more shrinkage.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline``.
    """
    from src.features.build_features import MissingStrategy

    if feature_config is None:
        # Linear models require imputation — passthrough will error on NaN
        feature_config = FeatureConfig(missing_strategy=MissingStrategy.impute)
    elif feature_config.missing_strategy == MissingStrategy.passthrough:
        logger.warning(
            "Linear model received missing_strategy=passthrough. "
            "Ridge cannot handle NaN inputs — switching to impute."
        )
        feature_config = FeatureConfig(
            numeric_features=feature_config.numeric_features,
            categorical_features=feature_config.categorical_features,
            target_encode_features=feature_config.target_encode_features,
            missing_strategy=MissingStrategy.impute,
            log_transform_target=feature_config.log_transform_target,
        )

    feature_pipeline = build_feature_pipeline(feature_config)
    return Pipeline(
        [
            ("feature_pipeline", feature_pipeline),
            ("model", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )


def linear_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,  # noqa: N803
    y_train: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_config: FeatureConfig,
) -> float:
    """Optuna objective function for Ridge hyperparameter optimisation.

    Performs k-fold cross-validation using pre-computed ``cv_splits`` and
    returns mean CV RMSE (in log-price space — Optuna minimises this).

    Args:
        trial: Optuna trial object used to sample hyperparameters.
        X_train: Training features (pre-split, full training set).
        y_train: Log1p-transformed prices for training rows.
        cv_splits: List of ``(train_idx, val_idx)`` from ``make_cv_splits``.
        feature_config: Feature engineering configuration.

    Returns:
        Mean CV RMSE across folds (log-price space, lower is better).
    """
    import optuna

    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)

    fold_rmses: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_fold_train = X_train.iloc[train_idx]  # noqa: N806
        X_fold_val = X_train.iloc[val_idx]  # noqa: N806
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        pipeline = build_linear_pipeline(feature_config=feature_config, alpha=alpha)

        try:
            pipeline.fit(X_fold_train, y_fold_train)
            y_pred = pipeline.predict(X_fold_val)
            rmse = float(np.sqrt(np.mean((y_fold_val - y_pred) ** 2)))
            fold_rmses.append(rmse)
        except Exception as exc:
            logger.warning("Fold %d failed: %s", fold, exc)
            raise optuna.exceptions.TrialPruned() from exc

        # Report intermediate value for Optuna pruner
        trial.report(float(np.mean(fold_rmses)), step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_rmse = float(np.mean(fold_rmses))
    logger.debug("Trial alpha=%.4f -> CV RMSE=%.4f", alpha, mean_rmse)
    return mean_rmse
