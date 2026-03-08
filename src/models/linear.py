"""Linear regression model for UK property price prediction.

Uses Ridge regression (L2 regularisation) on log1p-transformed prices.
The log transformation maps the right-skewed price distribution to
approximate normality, making linear assumptions more defensible.

Alpha is selected automatically via ``RidgeCV`` (leave-one-out CV over a
log-spaced grid from 1e-3 to 1e3) — no separate HPO loop required.

Sklearn pipeline structure::

    FeaturePipeline → RidgeCV(alphas=logspace(-3, 3, 13))

Usage::

    from src.models.linear import build_linear_pipeline

    pipeline = build_linear_pipeline()
    pipeline.fit(X_train, y_train_log)
    y_pred_log = pipeline.predict(X_test)
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

from src.features.build_features import FeatureConfig, build_feature_pipeline

logger = logging.getLogger(__name__)

# Log-spaced alpha grid: equivalent search space to the old Optuna [1e-3, 1e3]
_RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def build_linear_pipeline(
    feature_config: FeatureConfig | None = None,
) -> Pipeline:
    """Return an unfitted Ridge regression pipeline with automatic alpha selection.

    The pipeline is:
        1. ``FeaturePipeline`` (ColumnTransformer with imputation + encoding + scaling)
        2. ``RidgeCV`` which internally selects the best L2 regularisation strength

    Ridge is preferred over plain OLS because property data has many correlated
    features and regularisation prevents overfitting on smaller regional datasets.
    ``RidgeCV`` replaces the previous Optuna HPO loop — it achieves the same
    result deterministically in a fraction of the time.

    The target must be log1p-transformed *before* calling ``fit``. Callers
    should apply ``np.expm1`` to undo the log when converting back to GBP.

    Args:
        feature_config: Feature engineering config. Defaults to
            ``FeatureConfig(missing_strategy=MissingStrategy.impute)``
            since Ridge cannot handle NaN inputs.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline``.
    """
    from src.features.build_features import MissingStrategy

    if feature_config is None:
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
            ("model", RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)),
        ]
    )
