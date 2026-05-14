"""Inference helpers used by exploratory notebooks.

The training pipeline fits in log-space (callers apply ``np.log1p`` to ``y``
before fit and ``np.expm1`` to predictions when converting back to GBP).
These helpers wrap that convention so the notebook does not need to
remember it, and provide a residual-bootstrap routine for building a
predictive *distribution* over sale price from a point-only model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np

from src.data.config import MODELS_DIR

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from sklearn.pipeline import Pipeline


DEFAULT_MODEL_PATH = MODELS_DIR / "perpetual_best.joblib"


def load_pipeline(path: Path | None = None) -> Pipeline:
    """Load a trained sklearn Pipeline from disk (joblib)."""
    target = path or DEFAULT_MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"No trained pipeline at {target}. Run `just train` first.")
    return joblib.load(target)


def predict_price(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict GBP prices, inverting the training-time log1p transform."""
    y_log = pipeline.predict(X)
    return np.expm1(y_log)


def bootstrap_distribution(
    pipeline: Pipeline,
    X_single: pd.DataFrame,
    residuals_log: np.ndarray,
    *,
    n: int = 5000,
    seed: int | None = 42,
) -> np.ndarray:
    """Return ``n`` samples from a residual-bootstrap predictive distribution.

    Adds resampled log-space residuals to the point prediction (in log-space)
    and then inverts ``log1p`` so the returned array is in GBP. ``X_single``
    should be a single-row dataframe; ``residuals_log`` is the pool of
    holdout residuals (``y_true_log - y_pred_log``) drawn from a comparable
    slice (e.g. same district + property_type).
    """
    if X_single.shape[0] != 1:
        raise ValueError(f"X_single must have exactly one row, got {X_single.shape[0]}")
    if residuals_log.size == 0:
        raise ValueError("residuals_log is empty; cannot bootstrap a distribution")

    rng = np.random.default_rng(seed)
    point_log = float(pipeline.predict(X_single)[0])
    sampled = rng.choice(residuals_log, size=n, replace=True)
    return np.expm1(point_log + sampled)


def holdout_residuals_log(
    pipeline: Pipeline,
    X_holdout: pd.DataFrame,
    y_holdout_log: np.ndarray,
) -> np.ndarray:
    """Compute log-space residuals on a held-out slice.

    Returns ``y_true_log - y_pred_log`` for every row. Callers typically
    cache this once and pass it to ``bootstrap_distribution`` per-property.
    """
    y_pred_log = pipeline.predict(X_holdout)
    return y_holdout_log - y_pred_log
