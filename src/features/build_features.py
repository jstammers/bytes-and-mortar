"""Feature engineering pipeline for UK property price prediction.

Builds a scikit-learn Pipeline that applies:
- Configurable missing value strategy (drop rows / median imputation / passthrough)
- Target encoding for high-cardinality ``district`` column (~300 categories)
- Ordinal encoding for low-cardinality categoricals (property_type, duration, old_new)
- Standard scaling for numeric features

Also provides ``MedianByGroupBaseline`` — a sklearn-compatible estimator used as a
comparison baseline during model evaluation.

Usage::

    from src.features.build_features import FeatureConfig, build_feature_pipeline

    config = FeatureConfig(missing_strategy=MissingStrategy.impute)
    pipeline = build_feature_pipeline(config)
    X_transformed = pipeline.fit_transform(X_train, y_train)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema and defaults
# ---------------------------------------------------------------------------

#: Canonical schema: column name -> semantic type used to route through the transformer
FEATURE_SCHEMA: dict[str, Literal["numeric", "categorical", "target_encode"]] = {
    "property_type": "categorical",
    "old_new": "categorical",
    "duration": "categorical",
    "county": "categorical",
    "postcode_outward": "categorical",
    "current_energy_rating": "categorical",
    "construction_age_band": "categorical",
    "district": "target_encode",
    "year": "numeric",
    "month": "numeric",
    "current_energy_efficiency": "numeric",
    "total_floor_area": "numeric",
    "number_habitable_rooms": "numeric",
    "averageprice": "numeric",
    "index": "numeric",
}

DEFAULT_NUMERIC_FEATURES: list[str] = [
    "year",
    "month",
    "current_energy_efficiency",
    "total_floor_area",
    "number_habitable_rooms",
    "averageprice",
]

DEFAULT_CATEGORICAL_FEATURES: list[str] = [
    "property_type",
    "old_new",
    "duration",
    "county",
    "postcode_outward",
]

DEFAULT_TARGET_ENCODE_FEATURES: list[str] = ["district"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MissingStrategy(StrEnum):
    """Strategy for handling missing values in feature columns."""

    drop = "drop"
    impute = "impute"
    passthrough = "passthrough"


@dataclass
class FeatureConfig:
    """All feature engineering options. Fully serialisable for MLflow logging.

    Attributes:
        numeric_features: Numeric columns to include. Scaled after imputation.
        categorical_features: Low-cardinality categoricals. Ordinal-encoded.
        target_encode_features: High-cardinality categoricals. Target-encoded.
        missing_strategy: How to handle NaN values before encoding.
        log_transform_target: If True, callers should apply np.log1p to ``price``
            before passing ``y`` to fit/transform. This flag is metadata only —
            the pipeline itself does not touch ``y``.
    """

    numeric_features: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC_FEATURES))
    categorical_features: list[str] = field(
        default_factory=lambda: list(DEFAULT_CATEGORICAL_FEATURES)
    )
    target_encode_features: list[str] = field(
        default_factory=lambda: list(DEFAULT_TARGET_ENCODE_FEATURES)
    )
    missing_strategy: MissingStrategy = MissingStrategy.impute
    log_transform_target: bool = True

    @property
    def all_features(self) -> list[str]:
        """Ordered union of all feature column names."""
        return self.numeric_features + self.categorical_features + self.target_encode_features

    def to_dict(self) -> dict[str, str | bool]:
        """Flat dict suitable for ``mlflow.log_params()``."""
        return {
            "numeric_features": ",".join(self.numeric_features),
            "categorical_features": ",".join(self.categorical_features),
            "target_encode_features": ",".join(self.target_encode_features),
            "missing_strategy": self.missing_strategy.value,
            "log_transform_target": str(self.log_transform_target),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_features(df: pd.DataFrame, config: FeatureConfig) -> None:
    """Raise ``ValueError`` if any configured feature column is absent from ``df``.

    Args:
        df: The DataFrame to validate against.
        config: Feature configuration specifying required columns.

    Raises:
        ValueError: Lists all missing columns in one message.
    """
    missing = [col for col in config.all_features if col not in df.columns]
    if missing:
        raise ValueError(
            f"The following feature columns are missing from the DataFrame: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def build_feature_pipeline(config: FeatureConfig) -> Pipeline:
    """Return an unfitted sklearn Pipeline for feature engineering.

    The pipeline transforms X (a DataFrame of feature columns) into a 2-D
    numeric array. It does NOT touch y — log-transforming the target is the
    caller's responsibility (see ``FeatureConfig.log_transform_target``).

    Pipeline structure::

        ColumnTransformer
          numeric branch:
            SimpleImputer(strategy="median")  [only when missing_strategy=impute]
            StandardScaler()
          categorical branch:
            SimpleImputer(strategy="most_frequent")  [only when missing_strategy=impute]
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
          target_encode branch:
            SimpleImputer(strategy="most_frequent")  [only when missing_strategy=impute]
            sklearn TargetEncoder()

    When ``missing_strategy=passthrough`` all imputers are omitted and NaN
    values are passed through. XGBoost handles these natively; linear models
    will error — callers should set ``missing_strategy=impute`` for linear models.

    When ``missing_strategy=drop`` the pipeline itself does no dropping. The
    caller (``train_model.train``) is responsible for calling
    ``df.dropna(subset=config.all_features)`` before passing data to the pipeline.
    This keeps the sklearn Pipeline stateless with respect to row removal.

    Args:
        config: Feature engineering configuration.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline`` with a single ``"features"`` step
        containing a ``ColumnTransformer``.
    """
    use_imputer = config.missing_strategy == MissingStrategy.impute

    # --- Numeric branch ---
    numeric_steps: list[tuple[str, object]] = []
    if use_imputer:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # --- Categorical branch (low cardinality) ---
    categorical_steps: list[tuple[str, object]] = []
    if use_imputer:
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing"))
        )
    categorical_steps.append(
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            ),
        )
    )
    categorical_pipe = Pipeline(categorical_steps)

    # --- Target-encode branch (high cardinality, e.g. district) ---
    from sklearn.preprocessing import TargetEncoder  # requires scikit-learn >= 1.3

    target_enc_steps: list[tuple[str, object]] = []
    if use_imputer:
        target_enc_steps.append(
            ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing"))
        )
    target_enc_steps.append(
        (
            "encoder",
            TargetEncoder(
                target_type="continuous",
                smooth="auto",
                cv=5,
            ),
        )
    )
    target_enc_pipe = Pipeline(target_enc_steps)

    remainder = "passthrough" if config.missing_strategy == MissingStrategy.passthrough else "drop"

    transformers: list[tuple[str, object, list[str]]] = []
    if config.numeric_features:
        transformers.append(("numeric", numeric_pipe, config.numeric_features))
    if config.categorical_features:
        transformers.append(("categorical", categorical_pipe, config.categorical_features))
    if config.target_encode_features:
        transformers.append(("target_encode", target_enc_pipe, config.target_encode_features))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )

    return Pipeline([("features", column_transformer)])


# ---------------------------------------------------------------------------
# Baseline estimator
# ---------------------------------------------------------------------------


class MedianByGroupBaseline(BaseEstimator, RegressorMixin):
    """Predict the median target value for each (property_type, district) group.

    Sklearn-compatible (implements ``fit`` / ``predict`` / ``score``).

    Fallback hierarchy for unseen combinations at predict time:
    1. Exact ``(property_type, district)`` group median
    2. ``property_type`` median (district unseen)
    3. Global median (both unseen)

    This estimator operates on the *raw* (pre-feature-pipeline) DataFrame so
    it has access to the original categorical strings. It is evaluated
    separately from the main model pipeline.

    Attributes:
        group_cols: Columns to group by. Defaults to ``["property_type", "district"]``.
        fallback_col: Single-column fallback. Defaults to ``"property_type"``.
        group_medians_: Fitted per-group median lookup.
        fallback_medians_: Fitted single-column median lookup.
        global_median_: Fitted global median.
    """

    def __init__(
        self,
        group_cols: list[str] | None = None,
        fallback_col: str = "property_type",
    ) -> None:
        self.group_cols = group_cols
        self.fallback_col = fallback_col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MedianByGroupBaseline:  # noqa: N803
        """Compute group medians from training data.

        Args:
            X: Feature DataFrame containing at least ``group_cols`` columns.
            y: Target series (log-transformed price or raw price).

        Returns:
            Self (fitted).
        """
        cols = self.group_cols or ["property_type", "district"]
        available = [c for c in cols if c in X.columns]

        combined = X[available].copy()
        combined["_y"] = y.values

        if available:
            self.group_medians_: dict[tuple[str, ...], float] = (
                combined.groupby(available)["_y"].median().to_dict()
            )
        else:
            self.group_medians_ = {}

        if self.fallback_col in X.columns:
            self.fallback_medians_: dict[str, float] = (
                combined.groupby(self.fallback_col)["_y"].median().to_dict()
            )
        else:
            self.fallback_medians_ = {}

        self.global_median_: float = float(y.median())
        self._fit_cols = available
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Predict target using group median lookup with fallback chain.

        Args:
            X: Feature DataFrame.

        Returns:
            1-D numpy array of predictions.
        """
        check_is_fitted(self, ["group_medians_", "global_median_"])

        preds: list[float] = []
        for _, row in X.iterrows():
            key = tuple(row[c] for c in self._fit_cols if c in X.columns)
            if key in self.group_medians_:
                preds.append(self.group_medians_[key])
            elif (
                self.fallback_col in X.columns and row[self.fallback_col] in self.fallback_medians_
            ):
                preds.append(self.fallback_medians_[row[self.fallback_col]])
            else:
                preds.append(self.global_median_)

        return np.array(preds)
