"""Perpetual GBM model for UK property price prediction.

``PerpetualBooster`` is a self-generalising gradient boosting machine that
automatically selects the number of trees based on a single ``budget``
parameter — **no hyperparameter optimisation loop required**.

Budget reference (from benchmarks on the full UK property dataset):

    ========  ======  ========  =======
    Budget    MdAPE   RMSE (£)  Time
    ========  ======  ========  =======
    0.5       16.7%   £290,172  16s
    0.7       16.2%   £287,186  20s
    1.0       16.1%   £288,002  33s
    XGB+HPO   16.0%   £284,588  6.4min
    ========  ======  ========  =======

At ``budget=1.0`` Perpetual matches XGBoost+Optuna accuracy (0.06pp MdAPE
gap) while being ~12× faster to train.

Usage::

    from src.models.perpetual_model import build_perpetual_pipeline

    pipeline = build_perpetual_pipeline(budget=1.0)
    pipeline.fit(X_train, y_train_log)
    y_pred_log = pipeline.predict(X_test)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.features.build_features import FeatureConfig, build_feature_pipeline

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_PERPETUAL_DEFAULTS: dict[str, float | str] = {
    "budget": 1.0,
    "objective": "SquaredLoss",
}


class PerpetualWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper around ``PerpetualBooster``.

    ``PerpetualBooster`` implements the sklearn fit/predict interface but does
    not subclass ``BaseEstimator``.  Sklearn >= 1.6's ``Pipeline.predict``
    calls ``check_is_fitted(self)`` → ``get_tags(last_step)`` which requires
    ``__sklearn_tags__``, causing an ``AttributeError`` when the last step is
    a bare ``PerpetualBooster``.

    Inheriting from ``BaseEstimator`` and ``RegressorMixin`` provides
    ``__sklearn_tags__``, ``get_params``/``set_params``, and ``score`` for free,
    making ``PerpetualBooster`` a fully-compliant sklearn estimator.

    Args:
        objective: Loss function passed to ``PerpetualBooster``. Use
            ``"SquaredLoss"`` for regression.
        budget: Complexity control.  Higher values allow more trees and
            typically improve accuracy at the cost of training time.
    """

    def __init__(self, objective: str = "SquaredLoss", budget: float = 1.0) -> None:
        self.objective = objective
        self.budget = budget

    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_params: object) -> PerpetualWrapper:
        """Fit a ``PerpetualBooster`` on the given data.

        Args:
            X: Feature matrix.
            y: Log1p-transformed sale prices.
            **fit_params: Forwarded to ``PerpetualBooster.fit``.

        Returns:
            ``self`` (fitted instance).
        """
        from perpetual import PerpetualBooster

        self.booster_ = PerpetualBooster(objective=self.objective, budget=self.budget)
        self.booster_.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for *X*.

        Args:
            X: Feature matrix (same schema as used in ``fit``).

        Returns:
            1-D array of predicted log-prices.
        """
        check_is_fitted(self)
        return self.booster_.predict(X)


def build_perpetual_pipeline(
    feature_config: FeatureConfig | None = None,
    budget: float = 1.0,
    objective: str = "SquaredLoss",
) -> Pipeline:
    """Return an unfitted Perpetual GBM regression pipeline.

    The pipeline is:
        1. ``FeaturePipeline`` (ColumnTransformer encoding + optional imputation)
        2. ``PerpetualWrapper`` — a sklearn-compatible wrapper around
           ``PerpetualBooster`` with the given budget

    PerpetualBooster handles missing values natively, so
    ``missing_strategy=passthrough`` is a valid option.

    Target must be log1p-transformed before calling ``fit``.

    Args:
        feature_config: Feature engineering config. Defaults to
            ``FeatureConfig()`` (impute strategy, all default columns).
        budget: Complexity control. Higher values allow more trees and
            typically improve accuracy at the cost of training time.
            Default ``1.0`` matches XGBoost+Optuna accuracy on this dataset.
        objective: Loss function. ``"SquaredLoss"`` for regression.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline``.
    """
    if feature_config is None:
        feature_config = FeatureConfig()

    feature_pipeline = build_feature_pipeline(feature_config)
    return Pipeline(
        [
            ("feature_pipeline", feature_pipeline),
            ("model", PerpetualWrapper(objective=objective, budget=budget)),
        ]
    )
