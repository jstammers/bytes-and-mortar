"""Regression evaluation metrics for UK property price models.

All metrics operate in *original price space* (GBP), not log-space, to make
results interpretable. When a model predicts ``log(price + 1)``, the caller
must exponentiate predictions before passing them here.

Usage::

    from src.models.property_price.evaluate import compute_metrics, RegressionMetrics

    metrics = compute_metrics(y_true=test_prices, y_pred=predictions)
    print(f"RMSE: £{metrics.rmse:,.0f}  MAPE: {metrics.mape:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """Suite of regression evaluation metrics.

    All monetary metrics are in GBP (original price space).

    Attributes:
        rmse: Root Mean Squared Error — penalises large outliers heavily.
        mae: Mean Absolute Error — interpretable average error in £.
        mape: Mean Absolute Percentage Error — scale-independent.
        mdape: Median Absolute Percentage Error — robust to outliers.
        r2: Coefficient of determination.
        n_samples: Number of predictions evaluated.
    """

    rmse: float
    mae: float
    mape: float
    mdape: float
    r2: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a flat dict for ``mlflow.log_metrics()``."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "mdape": self.mdape,
            "r2": self.r2,
            "n_samples": self.n_samples,
        }

    def summary(self, label: str = "") -> str:
        """Human-readable one-line summary."""
        prefix = f"[{label}] " if label else ""
        return (
            f"{prefix}RMSE=£{self.rmse:,.0f}  MAE=£{self.mae:,.0f}  "
            f"MAPE={self.mape:.1%}  MdAPE={self.mdape:.1%}  R²={self.r2:.4f}  "
            f"n={self.n_samples:,}"
        )


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> RegressionMetrics:
    """Compute the full suite of regression metrics.

    Args:
        y_true: Ground-truth prices in GBP (original, not log-transformed).
        y_pred: Model predictions in GBP (original, not log-transformed).

    Returns:
        ``RegressionMetrics`` dataclass.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove rows where either value is NaN or non-positive
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.warning("compute_metrics: dropped %d rows with invalid values", n_dropped)

    y_true, y_pred = y_true[valid], y_pred[valid]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    ape = np.abs(y_true - y_pred) / y_true
    mape = float(np.mean(ape))
    mdape = float(np.median(ape))
    r2 = float(r2_score(y_true, y_pred))

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        mdape=mdape,
        r2=r2,
        n_samples=len(y_true),
    )


def evaluate_on_test(
    model_pred: np.ndarray | pd.Series,
    baseline_pred: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    log_transformed: bool = True,
) -> dict[str, RegressionMetrics]:
    """Evaluate both a trained model and a baseline against the same test set.

    Handles inverse log transformation when models predict ``log(price + 1)``.

    Args:
        model_pred: Predictions from the trained model.
        baseline_pred: Predictions from the baseline estimator.
        y_true: True target values (in the *same space* as the predictions,
            i.e. log-space if ``log_transformed=True``).
        log_transformed: If ``True``, all three arrays are treated as
            ``log1p(price)`` and are exponentiated before metric computation.

    Returns:
        Dict with keys ``"model"`` and ``"baseline"`` mapping to ``RegressionMetrics``.
    """

    def _to_price(arr: np.ndarray) -> np.ndarray:
        return np.expm1(arr) if log_transformed else arr

    y_true_price = _to_price(np.asarray(y_true, dtype=float))
    model_price = _to_price(np.asarray(model_pred, dtype=float))
    baseline_price = _to_price(np.asarray(baseline_pred, dtype=float))

    model_metrics = compute_metrics(y_true_price, model_price)
    baseline_metrics = compute_metrics(y_true_price, baseline_price)

    logger.info("Model    %s", model_metrics.summary("model"))
    logger.info("Baseline %s", baseline_metrics.summary("baseline"))

    # Log relative improvement
    rmse_improvement = (baseline_metrics.rmse - model_metrics.rmse) / baseline_metrics.rmse
    logger.info("RMSE improvement over baseline: %+.1f%%", rmse_improvement * 100)

    return {"model": model_metrics, "baseline": baseline_metrics}


def cv_metrics_summary(fold_metrics: list[RegressionMetrics]) -> dict[str, float]:
    """Aggregate per-fold metrics into mean ± std summary for MLflow logging.

    Args:
        fold_metrics: List of ``RegressionMetrics``, one per CV fold.

    Returns:
        Dict of ``{metric_mean: ..., metric_std: ...}`` for all scalar metrics.
    """
    if not fold_metrics:
        return {}

    metric_names = ["rmse", "mae", "mape", "mdape", "r2"]
    summary: dict[str, float] = {}

    for name in metric_names:
        values = np.array([getattr(m, name) for m in fold_metrics])
        summary[f"cv_{name}_mean"] = float(values.mean())
        summary[f"cv_{name}_std"] = float(values.std())

    logger.info(
        "CV summary (%d folds): RMSE=£%,.0f ± £%,.0f  R²=%.4f ± %.4f",
        len(fold_metrics),
        summary["cv_rmse_mean"],
        summary["cv_rmse_std"],
        summary["cv_r2_mean"],
        summary["cv_r2_std"],
    )
    return summary
