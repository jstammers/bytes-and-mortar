"""Derived feature engineering for UK property price prediction.

Computes new features from raw columns before the main ColumnTransformer step:

- ``log_floor_area``: ``log1p(total_floor_area)`` — addresses the right-skew of
  floor-area distribution, improving its linear relationship with log-price.
- ``floor_area_per_room``: ``total_floor_area / number_habitable_rooms`` —
  captures space efficiency (density), a proxy for property quality.
- ``energy_rating_numeric``: EPC letter grade A-G mapped to 7-1 -- lets the
  model treat energy efficiency as a continuous ordinal feature rather than a
  nominal category.

Usage::

    from src.features.derived import DerivedFeatureTransformer

    transformer = DerivedFeatureTransformer()
    X_enriched = transformer.fit_transform(X_raw)  # adds new columns in-place
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EPC letter grade → ordinal integer (A is best = 7, G is worst = 1)
ENERGY_RATING_MAP: dict[str, int] = {
    "A": 7,
    "B": 6,
    "C": 5,
    "D": 4,
    "E": 3,
    "F": 2,
    "G": 1,
}

#: Names of features added by :class:`DerivedFeatureTransformer`
DERIVED_NUMERIC_FEATURES: list[str] = [
    "log_floor_area",
    "floor_area_per_room",
    "energy_rating_numeric",
]


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add derived numeric features to a property DataFrame.

    This transformer is designed to be inserted as the *first* step inside
    ``build_feature_pipeline`` when ``FeatureConfig.derived_features=True``.
    It mutates a copy of the input DataFrame — it never drops columns.

    New columns produced (all numeric, NaN-safe):

    ``log_floor_area``
        ``log1p(clip(total_floor_area, 0, None))`` — linearises the
        relationship between floor area and log-price.

    ``floor_area_per_room``
        ``total_floor_area / max(number_habitable_rooms, 1)`` — space
        efficiency proxy; rooms clipped to ≥1 to avoid division by zero.

    ``energy_rating_numeric``
        EPC letter grade mapped to integers 1-7 via
        :data:`ENERGY_RATING_MAP`.  Rows with unrecognised or missing ratings
        produce NaN (handled downstream by the imputer).

    Args:
        energy_rating_col: Column name for EPC letter grade.
            Defaults to ``"current_energy_rating"``.
        floor_area_col: Column name for total floor area (m²).
            Defaults to ``"total_floor_area"``.
        rooms_col: Column name for habitable room count.
            Defaults to ``"number_habitable_rooms"``.
    """

    def __init__(
        self,
        energy_rating_col: str = "current_energy_rating",
        floor_area_col: str = "total_floor_area",
        rooms_col: str = "number_habitable_rooms",
    ) -> None:
        self.energy_rating_col = energy_rating_col
        self.floor_area_col = floor_area_col
        self.rooms_col = rooms_col

    # fit is a no-op — transformer has no learned state
    def fit(self, X: pd.DataFrame, y: object = None) -> DerivedFeatureTransformer:  # noqa: N803
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Add derived columns to a copy of *X*.

        Args:
            X: Feature DataFrame that must contain the source columns
               (``total_floor_area``, ``number_habitable_rooms``,
               ``current_energy_rating``).  Missing source columns are
               silently skipped -- derived columns will be absent from output.

        Returns:
            A copy of *X* with up to three new numeric columns appended.
        """
        X = X.copy()  # noqa: N806

        # log_floor_area
        if self.floor_area_col in X.columns:
            X["log_floor_area"] = np.log1p(X[self.floor_area_col].clip(lower=0))

        # floor_area_per_room
        if self.floor_area_col in X.columns and self.rooms_col in X.columns:
            rooms_safe = X[self.rooms_col].clip(lower=1)
            X["floor_area_per_room"] = X[self.floor_area_col] / rooms_safe

        # energy_rating_numeric
        if self.energy_rating_col in X.columns:
            X["energy_rating_numeric"] = (
                X[self.energy_rating_col].map(ENERGY_RATING_MAP).astype("Float64")
            )

        return X

    def get_feature_names_out(self, input_features: object = None) -> list[str]:
        """Return the names of output columns (passthrough + derived)."""
        # Sklearn convention: return column names if input_features is supplied
        base = list(input_features) if input_features is not None else []
        return base + DERIVED_NUMERIC_FEATURES
