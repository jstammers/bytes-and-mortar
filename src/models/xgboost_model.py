"""XGBoost model for UK property price prediction.

Uses the XGBoost sklearn wrapper (``xgb.XGBRegressor``) with the ``hist``
tree method, which handles missing values natively — no imputation required,
though imputation is still applied when ``FeatureConfig.missing_strategy=impute``.

Predicts ``log1p(price)`` so the loss landscape is better behaved across the
wide UK price range (£50k bedsit → £10M penthouse).

Hyperparameter search space (Optuna):
    n_estimators:       int   [100, 1000]
    max_depth:          int   [3, 10]
    learning_rate:      float log-uniform [1e-3, 0.3]
    subsample:          float [0.6, 1.0]
    colsample_bytree:   float [0.6, 1.0]
    min_child_weight:   int   [1, 10]
    reg_alpha (L1):     float log-uniform [1e-8, 10]
    reg_lambda (L2):    float log-uniform [1e-8, 10]

Usage::

    from src.models.xgboost_model import build_xgboost_pipeline

    pipeline = build_xgboost_pipeline(n_estimators=500, max_depth=6)
    pipeline.fit(X_train, y_train_log)
    y_pred_log = pipeline.predict(X_test)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.pipeline import Pipeline

from src.features.build_features import FeatureConfig, build_feature_pipeline

if TYPE_CHECKING:
    import optuna
    import pandas as pd

logger = logging.getLogger(__name__)

# Default XGBoost parameters — reasonable starting point before HPO
_XGBOOST_DEFAULTS: dict[str, int | float | str] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def build_xgboost_pipeline(
    feature_config: FeatureConfig | None = None,
    **xgb_params: int | float | str,
) -> Pipeline:
    """Return an unfitted XGBoost regression pipeline.

    The pipeline is:
        1. ``FeaturePipeline`` (ColumnTransformer encoding + optional imputation)
        2. ``xgb.XGBRegressor`` with the given parameters

    XGBoost handles NaN natively via the ``hist`` tree method, so
    ``missing_strategy=passthrough`` is a valid (and fast) option.

    Target must be log1p-transformed before calling ``fit``.

    Args:
        feature_config: Feature engineering config.  Defaults to
            ``FeatureConfig()`` (impute strategy, all default columns).
        **xgb_params: Override any XGBoost hyperparameter.  Merged with
            ``_XGBOOST_DEFAULTS``.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline``.
    """
    import xgboost as xgb

    if feature_config is None:
        feature_config = FeatureConfig()

    params = {**_XGBOOST_DEFAULTS, **xgb_params}
    feature_pipeline = build_feature_pipeline(feature_config)

    return Pipeline(
        [
            ("feature_pipeline", feature_pipeline),
            ("model", xgb.XGBRegressor(**params)),
        ]
    )


def xgboost_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,  # noqa: N803
    y_train: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_config: FeatureConfig,
) -> float:
    """Optuna objective function for XGBoost hyperparameter optimisation.

    Performs k-fold cross-validation using pre-computed ``cv_splits`` and
    returns mean CV RMSE (in log-price space).  ``n_estimators`` is a fixed
    value sampled by Optuna — the HPO search space upper-bounds it at 1000.

    Args:
        trial: Optuna trial object.
        X_train: Training features (full training set).
        y_train: Log1p-transformed prices.
        cv_splits: ``(train_idx, val_idx)`` index pairs from ``make_cv_splits``.
        feature_config: Feature engineering configuration.

    Returns:
        Mean CV RMSE (log-price space, lower is better).
    """
    import optuna

    params: dict[str, int | float | str] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    fold_rmses: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_fold_train = X_train.iloc[train_idx]  # noqa: N806
        X_fold_val = X_train.iloc[val_idx]  # noqa: N806
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        pipeline = build_xgboost_pipeline(feature_config=feature_config, **params)

        try:
            pipeline.fit(X_fold_train, y_fold_train)
            y_pred = pipeline.predict(X_fold_val)
            rmse = float(np.sqrt(np.mean((y_fold_val - y_pred) ** 2)))
            fold_rmses.append(rmse)
        except Exception as exc:
            logger.warning("Fold %d failed: %s", fold, exc)
            raise optuna.exceptions.TrialPruned() from exc

        trial.report(float(np.mean(fold_rmses)), step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_rmse = float(np.mean(fold_rmses))
    logger.debug(
        "Trial n_est=%d depth=%d lr=%.4f -> CV RMSE=%.4f",
        params["n_estimators"],
        params["max_depth"],
        params["learning_rate"],
        mean_rmse,
    )
    return mean_rmse
