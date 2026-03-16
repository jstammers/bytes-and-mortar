"""Model investigation and explainability tools for UK property price prediction.

Addresses five diagnostic questions about the £240k RMSE error:

1. **Imputation bias** — does the model perform worse on rows where EPC or
   other features were missing and filled by the median imputer?
2. **Noisy features** — does removing a feature improve (or barely hurt)
   performance?  Measured via permutation importance.
3. **Calibration** — are predictions systematically too high or too low at
   different price levels?
4. **Overfitting to extremes** — do errors grow disproportionately at the
   tails of the price distribution?
5. **Regional consistency** — which counties have the largest prediction error?

SHAP analysis (requires a fitted ``PerpetualBooster``) provides feature-level
attribution to supplement permutation importance.

Usage::

    with mlflow.start_run():
        run_investigations(
            pipeline=fitted_pipeline,
            X_train=X_train,
            y_train_log=y_train_log,
            X_test_raw=X_test,   # BEFORE pipeline transform — may contain NaN
            y_test_log=y_test_log,
            feature_cols=feature_config.all_features,
        )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src.models.evaluate import RegressionMetrics, compute_metrics

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImputationBiasResult:
    """Performance split by whether rows had missing feature values.

    Attributes:
        imputed_metrics: Metrics for rows where ≥1 feature was NaN before the
            pipeline imputed it.  ``None`` if fewer than 10 such rows exist.
        complete_metrics: Metrics for fully-observed rows.
        n_imputed: Count of imputed rows in the test set.
        n_complete: Count of fully-observed rows in the test set.
        imputed_fraction: Fraction of test rows that were imputed.
    """

    imputed_metrics: RegressionMetrics | None
    complete_metrics: RegressionMetrics | None
    n_imputed: int
    n_complete: int
    imputed_fraction: float


@dataclass
class CalibrationBin:
    """Single decile bin for the reliability diagram.

    Attributes:
        bin_index: 0-based bin index (ascending by predicted price).
        mean_predicted: Mean predicted price in this bin (£).
        mean_actual: Mean actual price in this bin (£).
        n_samples: Number of test rows in this bin.
    """

    bin_index: int
    mean_predicted: float
    mean_actual: float
    n_samples: int

    @property
    def bias(self) -> float:
        """Signed bias: predicted - actual (GBP).  Positive = over-prediction."""
        return self.mean_predicted - self.mean_actual


@dataclass
class PermutationImportanceResult:
    """Permutation importance for a single feature.

    Attributes:
        feature: Feature column name.
        importance_mean: Mean increase in MSE (log-space) when this feature is
            randomly shuffled across ``n_repeats`` permutations.  Higher values
            indicate the feature contributes more to predictions.
        importance_std: Standard deviation across repeats.
    """

    feature: str
    importance_mean: float
    importance_std: float


@dataclass
class InvestigationSummary:
    """Top-level container returned by :func:`run_investigations`.

    Attributes:
        imputation_bias: Split metrics for imputed vs complete-case rows.
        calibration_bins: Reliability diagram bins (sorted by predicted price).
        permutation_importance: Per-feature permutation importance, sorted
            descending by ``importance_mean``.
        regional_metrics: Mapping of county name → :class:`RegressionMetrics`.
        shap_feature_names: Feature names aligned to ``shap_mean_abs`` columns.
        shap_mean_abs: Mean |SHAP value| per feature (``None`` for non-GBM).
    """

    imputation_bias: ImputationBiasResult | None = None
    calibration_bins: list[CalibrationBin] = field(default_factory=list)
    permutation_importance: list[PermutationImportanceResult] = field(default_factory=list)
    regional_metrics: dict[str, RegressionMetrics] = field(default_factory=dict)
    shap_feature_names: list[str] = field(default_factory=list)
    shap_mean_abs: np.ndarray | None = None


# ---------------------------------------------------------------------------
# SHAP analysis (PerpetualBooster.predict_contributions)
# ---------------------------------------------------------------------------


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,  # noqa: N803
) -> tuple[np.ndarray, list[str]]:
    """Return SHAP contributions via ``PerpetualBooster.predict_contributions()``.

    Perpetual's native ``predict_contributions`` returns an array of shape
    ``(n_samples, n_features + 1)`` where the last column is the bias term.
    We drop the bias and align the remaining columns to the feature names
    reported by the ColumnTransformer.

    Args:
        pipeline: Fitted sklearn Pipeline with a ``"feature_pipeline"`` step
            (containing a ``ColumnTransformer`` named ``"features"``) and a
            ``"model"`` step (a fitted :class:`PerpetualWrapper`).
        X: Feature DataFrame in the *original* schema (pre-transform).

    Returns:
        ``(shap_values, feature_names)`` — SHAP array of shape
        ``(n_samples, n_features)`` and a list of transformed feature names.

    Raises:
        ValueError: If the pipeline does not contain the expected steps.
        AttributeError: If the model step has not been fitted yet.
    """
    feature_step = pipeline.named_steps.get("feature_pipeline")
    if feature_step is None:
        raise ValueError(
            "Pipeline must contain a 'feature_pipeline' step. "
            "Use build_perpetual_pipeline() or build_linear_pipeline()."
        )

    model_step = pipeline.named_steps.get("model")
    if model_step is None or not hasattr(model_step, "booster_"):
        raise ValueError(
            "Pipeline 'model' step must be a fitted PerpetualWrapper "
            "(i.e. have a 'booster_' attribute). SHAP is only supported for "
            "the perpetual model type."
        )

    # Feature names from the ColumnTransformer (after encoding)
    column_transformer = feature_step.named_steps["features"]
    feature_names = list(column_transformer.get_feature_names_out())

    # Transform X through the feature pipeline only (not the model)
    X_transformed = feature_step.transform(X)  # noqa: N806

    # predict_contributions returns (n_samples, n_features + 1)
    # The last column is the bias/intercept — drop it
    contributions = model_step.booster_.predict_contributions(X_transformed)
    shap_values = contributions[:, : len(feature_names)]

    return shap_values, feature_names


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_features: int = 15,
    title: str = "SHAP Feature Importance",
) -> plt.Figure:
    """Horizontal bar chart of mean |SHAP| per feature (descending importance).

    Args:
        shap_values: Array of shape ``(n_samples, n_features)``.
        feature_names: Feature names aligned to columns of *shap_values*.
        max_features: Maximum number of features to display.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    n = min(max_features, len(feature_names))
    # Sort ascending so the most-important feature is at the top of the chart
    top_idx = np.argsort(mean_abs)[-n:]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.45)))
    vals = mean_abs[top_idx]
    names = [feature_names[i] for i in top_idx]
    bars = ax.barh(names, vals, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (log-price contribution)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    feature_names: list[str],
    X_transformed: np.ndarray,  # noqa: N803
    feature_name: str,
) -> plt.Figure:
    """Scatter plot of transformed feature value vs its SHAP value.

    A flat relationship indicates the feature adds little signal; a clear
    monotone trend suggests genuine predictive value.

    Args:
        shap_values: SHAP array of shape ``(n_samples, n_features)``.
        feature_names: Names aligned to *shap_values* columns.
        X_transformed: Transformed feature matrix from the feature pipeline.
        feature_name: Feature to plot on the x-axis.

    Returns:
        Matplotlib figure.

    Raises:
        ValueError: If *feature_name* is not found in *feature_names*.
    """
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not in feature_names")

    idx = feature_names.index(feature_name)
    x = X_transformed[:, idx]
    y = shap_values[:, idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.3, s=8, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", lw=1.5, label="Zero SHAP")
    ax.set_xlabel(f"{feature_name} (transformed value)", fontsize=11)
    ax.set_ylabel("SHAP value (log-price contribution)", fontsize=11)
    ax.set_title(f"SHAP Dependence — {feature_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Imputation bias analysis
# ---------------------------------------------------------------------------


def imputation_bias_analysis(
    X_test_raw: pd.DataFrame,  # noqa: N803
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list[str],
    min_group_size: int = 10,
) -> ImputationBiasResult:
    """Compare model performance on imputed vs complete-case test rows.

    Rows where any feature column is ``NaN`` *before* the pipeline imputes it
    are classified as "imputed".  This detects whether median/mode fill
    systematically degrades prediction quality.

    Args:
        X_test_raw: Test features **before** pipeline transformation (may
            contain NaN).  Must share the same index as *y_true* / *y_pred*.
        y_true: True sale prices in GBP (not log-transformed).
        y_pred: Predicted sale prices in GBP (not log-transformed).
        feature_cols: Feature columns to check for missingness.
        min_group_size: Minimum rows needed to compute metrics for a group.

    Returns:
        :class:`ImputationBiasResult` with split metrics.
    """
    present = [c for c in feature_cols if c in X_test_raw.columns]
    has_nulls = X_test_raw[present].isnull().any(axis=1).values

    n_imputed = int(has_nulls.sum())
    n_complete = int((~has_nulls).sum())
    total = len(has_nulls)
    imputed_fraction = n_imputed / max(total, 1)

    imputed_metrics = (
        compute_metrics(y_true[has_nulls], y_pred[has_nulls])
        if n_imputed >= min_group_size
        else None
    )
    complete_metrics = (
        compute_metrics(y_true[~has_nulls], y_pred[~has_nulls])
        if n_complete >= min_group_size
        else None
    )

    logger.info(
        "Imputation bias: n_imputed=%d (%.1f%%)  n_complete=%d",
        n_imputed,
        100 * imputed_fraction,
        n_complete,
    )
    if imputed_metrics:
        logger.info(
            "  Imputed:  MdAPE=%.1f%%  RMSE=£%,.0f",
            100 * imputed_metrics.mdape,
            imputed_metrics.rmse,
        )
    if complete_metrics:
        logger.info(
            "  Complete: MdAPE=%.1f%%  RMSE=£%,.0f",
            100 * complete_metrics.mdape,
            complete_metrics.rmse,
        )

    return ImputationBiasResult(
        imputed_metrics=imputed_metrics,
        complete_metrics=complete_metrics,
        n_imputed=n_imputed,
        n_complete=n_complete,
        imputed_fraction=imputed_fraction,
    )


def plot_imputation_bias(result: ImputationBiasResult) -> plt.Figure:
    """Side-by-side bar charts comparing MdAPE and RMSE for imputed vs complete rows.

    Args:
        result: Output of :func:`imputation_bias_analysis`.

    Returns:
        Matplotlib figure.
    """
    groups: list[str] = []
    mdapes: list[float] = []
    rmses: list[float] = []

    if result.complete_metrics:
        groups.append(f"Complete\n(n={result.n_complete:,})")
        mdapes.append(result.complete_metrics.mdape * 100)
        rmses.append(result.complete_metrics.rmse / 1000)

    if result.imputed_metrics:
        groups.append(f"Imputed\n(n={result.n_imputed:,})")
        mdapes.append(result.imputed_metrics.mdape * 100)
        rmses.append(result.imputed_metrics.rmse / 1000)

    if not groups:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            f"Insufficient data for imputation bias analysis\n"
            f"(imputed={result.n_imputed}, complete={result.n_complete})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        return fig

    colors = ["#1976D2", "#E53935"][: len(groups)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(groups, mdapes, color=colors)
    axes[0].set_ylabel("MdAPE (%)", fontsize=11)
    axes[0].set_title("Median Absolute % Error", fontsize=12, fontweight="bold")
    for bar, val in zip(axes[0].patches, mdapes, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axes[1].bar(groups, rmses, color=colors)
    axes[1].set_ylabel("RMSE (£000s)", fontsize=11)
    axes[1].set_title("Root Mean Squared Error", fontsize=12, fontweight="bold")
    for bar, val in zip(axes[1].patches, rmses, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"£{val:.0f}k",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.suptitle(
        f"Imputation Bias Analysis  ({result.imputed_fraction:.1%} of test rows imputed)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


def calibration_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Compute reliability diagram bins: mean predicted vs mean actual.

    Bins are formed by sorting predictions into equal-count buckets.  A
    perfectly calibrated model would have ``mean_predicted == mean_actual``
    in every bin.  Systematic deviation reveals price-level biases (e.g.
    underestimating expensive properties).

    Args:
        y_true: True sale prices in GBP.
        y_pred: Predicted sale prices in GBP.
        n_bins: Number of equal-count bins to form.

    Returns:
        List of :class:`CalibrationBin`, one per bin (ascending predicted price).
    """
    sorted_idx = np.argsort(y_pred)
    bin_size = max(1, len(y_pred) // n_bins)

    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_pred)
        idx = sorted_idx[start:end]
        if len(idx) == 0:
            continue
        bins.append(
            CalibrationBin(
                bin_index=i,
                mean_predicted=float(y_pred[idx].mean()),
                mean_actual=float(y_true[idx].mean()),
                n_samples=len(idx),
            )
        )
    return bins


def plot_calibration(bins: list[CalibrationBin]) -> plt.Figure:
    """Two-panel plot: reliability diagram + bias-by-decile bar chart.

    Args:
        bins: Output of :func:`calibration_analysis`.

    Returns:
        Matplotlib figure.
    """
    if not bins:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        return fig

    mean_preds = [b.mean_predicted / 1_000 for b in bins]
    mean_actuals = [b.mean_actual / 1_000 for b in bins]
    biases = [b.bias / 1_000 for b in bins]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    min_v = min(min(mean_preds), min(mean_actuals))
    max_v = max(max(mean_preds), max(mean_actuals))
    axes[0].plot(
        [min_v, max_v], [min_v, max_v], "r--", lw=2, label="Perfect calibration", zorder=3
    )
    axes[0].plot(mean_preds, mean_actuals, "o-", color="steelblue", lw=2, label="Actual mean")
    axes[0].set_xlabel("Mean Predicted Price (£000s)", fontsize=11)
    axes[0].set_ylabel("Mean Actual Price (£000s)", fontsize=11)
    axes[0].set_title("Reliability Diagram", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].ticklabel_format(style="plain", axis="both")

    # Bias by price decile
    colors = ["#d32f2f" if b > 0 else "#1976D2" for b in biases]
    axes[1].bar(range(len(bins)), biases, color=colors, edgecolor="white")
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set_xlabel("Price Decile (1 = cheapest → 10 = most expensive)", fontsize=11)
    axes[1].set_ylabel("Bias (GBP 000s): predicted - actual", fontsize=11)
    axes[1].set_title(
        "Prediction Bias by Price Decile\n(red = over-prediction, blue = under-prediction)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xticks(range(len(bins)))
    axes[1].set_xticklabels([str(b.bin_index + 1) for b in bins])

    plt.suptitle("Calibration Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regional performance
# ---------------------------------------------------------------------------


def regional_performance(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_col: str = "county",
    min_samples: int = 30,
) -> dict[str, RegressionMetrics]:
    """Compute regression metrics grouped by region.

    Regions with fewer than *min_samples* rows are skipped to avoid noisy
    estimates from small samples.

    Args:
        test_df: Test DataFrame containing the *region_col* column.
        y_true: True sale prices in GBP.
        y_pred: Predicted sale prices in GBP.
        region_col: Column to group by (e.g. ``"county"``).
        min_samples: Minimum rows per region.

    Returns:
        Dict mapping region name → :class:`RegressionMetrics`.
    """
    if region_col not in test_df.columns:
        logger.warning("Column '%s' not in test DataFrame — skipping regional analysis", region_col)
        return {}

    results: dict[str, RegressionMetrics] = {}
    for region in test_df[region_col].dropna().unique():
        mask = (test_df[region_col] == region).values
        n = int(mask.sum())
        if n < min_samples:
            continue
        metrics = compute_metrics(y_true[mask], y_pred[mask])
        results[region] = metrics
        logger.debug("Region %-30s n=%4d  MdAPE=%.1f%%", region, n, 100 * metrics.mdape)

    logger.info(
        "Regional performance: %d regions with ≥%d samples", len(results), min_samples
    )
    return results


def plot_regional_performance(
    regional_metrics: dict[str, RegressionMetrics],
    metric: str = "mdape",
    top_n: int = 30,
) -> plt.Figure:
    """Horizontal bar chart of the worst *top_n* regions by *metric*.

    Regions above the cross-region mean are coloured red; below are blue.

    Args:
        regional_metrics: Output of :func:`regional_performance`.
        metric: Metric attribute on :class:`RegressionMetrics` to plot.
        top_n: Number of regions to display (sorted by worst performance).

    Returns:
        Matplotlib figure.
    """
    if not regional_metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No regional data available", ha="center", va="center")
        return fig

    regions = list(regional_metrics.keys())
    raw_values = [float(getattr(regional_metrics[r], metric)) for r in regions]
    scale = 100.0 if metric in ("mdape", "mape") else 1.0
    values = [v * scale for v in raw_values]

    # Sort by worst (highest) metric; take top_n
    sorted_pairs = sorted(zip(values, regions, strict=False), reverse=True)[:top_n]
    sorted_pairs = sorted_pairs[::-1]  # flip to ascending for horizontal bar
    if not sorted_pairs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    vals, regs = zip(*sorted_pairs, strict=False)

    overall_mean = float(np.mean(values))
    colors = ["#d32f2f" if v > overall_mean else "#1976D2" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(6, len(regs) * 0.38)))
    ax.barh(regs, vals, color=colors, edgecolor="white")
    ax.axvline(overall_mean, color="black", linestyle="--", lw=1.5, label=f"Mean={overall_mean:.1f}%")
    ylabel = f"{metric.upper()} (%)" if metric in ("mdape", "mape") else metric.upper()
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(
        f"Regional Performance — {metric.upper()} by County\n"
        f"(top {len(regs)} shown; red = above mean)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Permutation importance (ablation proxy)
# ---------------------------------------------------------------------------


def permutation_importance_study(
    pipeline: Pipeline,
    X_test: pd.DataFrame,  # noqa: N803
    y_test_log: np.ndarray,
    feature_cols: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[PermutationImportanceResult]:
    """Estimate feature importance by measuring the MSE increase when shuffled.

    Uses sklearn's ``permutation_importance`` (scorer: ``neg_mean_squared_error``
    in log-space).  Features that *hurt* performance when shuffled (high
    ``importance_mean``) are valuable; features with ``importance_mean ≈ 0``
    or negative add noise and could be dropped.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features DataFrame.
        y_test_log: Log-transformed test prices.
        feature_cols: Feature column names (must match *X_test* columns).
        n_repeats: Number of shuffle repetitions per feature.
        random_state: Reproducibility seed.

    Returns:
        List of :class:`PermutationImportanceResult`, sorted descending by
        ``importance_mean`` (most important first).
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline,
        X_test,
        y_test_log,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    perm_results = [
        PermutationImportanceResult(
            feature=feature,
            importance_mean=float(result.importances_mean[i]),
            importance_std=float(result.importances_std[i]),
        )
        for i, feature in enumerate(feature_cols)
    ]
    perm_results.sort(key=lambda r: r.importance_mean, reverse=True)

    logger.info(
        "Permutation importance complete (%d features, %d repeats)", len(perm_results), n_repeats
    )
    for r in perm_results[:5]:
        logger.info("  %-35s mean_Δmse=%.6f ± %.6f", r.feature, r.importance_mean, r.importance_std)
    return perm_results


def plot_permutation_importance(
    results: list[PermutationImportanceResult],
    title: str = "Permutation Feature Importance",
) -> plt.Figure:
    """Horizontal bar chart with error bars for permutation importance.

    Features with ``importance_mean < 0`` (shuffling *helps*) are coloured red
    as a signal that they may be adding noise.

    Args:
        results: Output of :func:`permutation_importance_study`.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No importance data available", ha="center", va="center")
        return fig

    # Show all features; most important at top
    ordered = results[::-1]  # ascending for horizontal bar (best at top)
    features = [r.feature for r in ordered]
    means = [r.importance_mean for r in ordered]
    stds = [r.importance_std for r in ordered]

    colors = ["#d32f2f" if m < 0 else "steelblue" for m in means]

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.42)))
    ax.barh(features, means, xerr=stds, color=colors, edgecolor="white", capsize=3)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel(
        "Mean MSE increase when feature shuffled (neg_MSE scorer; higher = more important)",
        fontsize=10,
    )
    ax.set_title(
        f"{title}\n(red bars: shuffling improves score → potential noise feature)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_investigations(
    pipeline: Pipeline,
    X_train: pd.DataFrame,  # noqa: N803
    y_train_log: np.ndarray,
    X_test_raw: pd.DataFrame,  # noqa: N803
    y_test_log: np.ndarray,
    feature_cols: list[str],
    region_col: str = "county",
    n_calibration_bins: int = 10,
    n_shap_samples: int = 2000,
    n_perm_repeats: int = 5,
) -> InvestigationSummary:
    """Run all investigations and log plots + metrics to the active MLflow run.

    Must be called inside an active ``mlflow.start_run()`` context.  All plots
    are saved under an ``investigate/`` artifact subfolder.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_train: Training features (used only if future analyses need it).
        y_train_log: Log-transformed training prices.
        X_test_raw: Test features **before** pipeline transform — NaN values
            present here reveal which rows were imputed.
        y_test_log: Log-transformed test prices.
        feature_cols: All feature column names (from
            ``FeatureConfig.all_features``).
        region_col: Column for regional grouping (default ``"county"``).
        n_calibration_bins: Number of equal-count bins for the reliability
            diagram (default 10).
        n_shap_samples: Maximum rows to use for SHAP computation (subsampled
            for speed; default 2000).
        n_perm_repeats: Shuffle repetitions per feature in permutation
            importance (default 5).

    Returns:
        :class:`InvestigationSummary` with all computed results.
    """
    import mlflow

    summary = InvestigationSummary()

    # Predictions in price space
    y_pred_log = pipeline.predict(X_test_raw)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    plt.style.use("seaborn-v0_8-whitegrid")

    # ------------------------------------------------------------------
    # 1. SHAP analysis (PerpetualBooster only)
    # ------------------------------------------------------------------
    model_step = pipeline.named_steps.get("model")
    if model_step is not None and hasattr(model_step, "booster_"):
        logger.info("Running SHAP analysis (n=%d rows)…", min(n_shap_samples, len(X_test_raw)))
        rng = np.random.default_rng(42)
        n_shap = min(n_shap_samples, len(X_test_raw))
        shap_idx = rng.choice(len(X_test_raw), size=n_shap, replace=False)
        X_shap = X_test_raw.iloc[shap_idx]  # noqa: N806

        try:
            shap_values, shap_names = compute_shap_values(pipeline, X_shap)
            summary.shap_feature_names = shap_names
            summary.shap_mean_abs = np.abs(shap_values).mean(axis=0)

            # Global summary bar chart
            fig = plot_shap_summary(shap_values, shap_names)
            mlflow.log_figure(fig, "investigate/shap_summary.png")
            plt.close(fig)

            # Dependence plots for top-3 most important features
            feature_step = pipeline.named_steps["feature_pipeline"]
            X_shap_transformed = feature_step.transform(X_shap)  # noqa: N806
            top3 = [shap_names[i] for i in np.argsort(summary.shap_mean_abs)[-3:][::-1]]
            for feat in top3:
                try:
                    fig = plot_shap_dependence(shap_values, shap_names, X_shap_transformed, feat)
                    safe_name = feat.replace("/", "_").replace(" ", "_")
                    mlflow.log_figure(fig, f"investigate/shap_dependence_{safe_name}.png")
                    plt.close(fig)
                except Exception as dep_exc:
                    logger.warning("SHAP dependence plot for '%s' failed: %s", feat, dep_exc)

            # Log mean |SHAP| as MLflow metrics
            for name, val in zip(shap_names, summary.shap_mean_abs, strict=False):
                safe = name.replace(" ", "_")[:50]
                mlflow.log_metric(f"shap_{safe}", float(val))

            logger.info("SHAP analysis complete — %d features", len(shap_names))
        except Exception as exc:
            logger.warning("SHAP analysis failed: %s", exc)
    else:
        logger.info("Skipping SHAP — model is not a fitted PerpetualWrapper")

    # ------------------------------------------------------------------
    # 2. Imputation bias analysis
    # ------------------------------------------------------------------
    logger.info("Running imputation bias analysis…")
    try:
        bias_result = imputation_bias_analysis(X_test_raw, y_true, y_pred, feature_cols)
        summary.imputation_bias = bias_result

        fig = plot_imputation_bias(bias_result)
        mlflow.log_figure(fig, "investigate/imputation_bias.png")
        plt.close(fig)

        mlflow.log_metrics(
            {
                "investigate_imputed_fraction": bias_result.imputed_fraction,
                "investigate_n_imputed": float(bias_result.n_imputed),
            }
        )
        if bias_result.imputed_metrics:
            mlflow.log_metrics(
                {f"investigate_imputed_{k}": v for k, v in bias_result.imputed_metrics.to_dict().items()}
            )
        if bias_result.complete_metrics:
            mlflow.log_metrics(
                {
                    f"investigate_complete_{k}": v
                    for k, v in bias_result.complete_metrics.to_dict().items()
                }
            )
    except Exception as exc:
        logger.warning("Imputation bias analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Calibration analysis
    # ------------------------------------------------------------------
    logger.info("Running calibration analysis…")
    try:
        cal_bins = calibration_analysis(y_true, y_pred, n_bins=n_calibration_bins)
        summary.calibration_bins = cal_bins

        fig = plot_calibration(cal_bins)
        mlflow.log_figure(fig, "investigate/calibration.png")
        plt.close(fig)

        if cal_bins:
            biases = [b.bias for b in cal_bins]
            mlflow.log_metrics(
                {
                    "investigate_mean_bias": float(np.mean(biases)),
                    "investigate_max_abs_bin_bias": float(np.max(np.abs(biases))),
                    "investigate_bias_std": float(np.std(biases)),
                }
            )
    except Exception as exc:
        logger.warning("Calibration analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # 4. Regional performance
    # ------------------------------------------------------------------
    if region_col and region_col in X_test_raw.columns:
        logger.info("Running regional performance analysis…")
        try:
            reg_metrics = regional_performance(X_test_raw, y_true, y_pred, region_col=region_col)
            summary.regional_metrics = reg_metrics

            if reg_metrics:
                fig = plot_regional_performance(reg_metrics)
                mlflow.log_figure(fig, "investigate/regional_performance.png")
                plt.close(fig)

                sorted_regs = sorted(reg_metrics.items(), key=lambda x: x[1].mdape)
                best5 = sorted_regs[:5]
                worst5 = sorted_regs[-5:]
                for region, m in best5:
                    safe = region.replace(" ", "_")[:25]
                    mlflow.log_metric(f"investigate_best_{safe}_mdape", m.mdape)
                for region, m in worst5:
                    safe = region.replace(" ", "_")[:25]
                    mlflow.log_metric(f"investigate_worst_{safe}_mdape", m.mdape)

                worst_region, worst_m = worst5[-1]
                best_region, best_m = best5[0]
                logger.info(
                    "Regional spread: best=%s MdAPE=%.1f%%  worst=%s MdAPE=%.1f%%",
                    best_region,
                    100 * best_m.mdape,
                    worst_region,
                    100 * worst_m.mdape,
                )
        except Exception as exc:
            logger.warning("Regional performance analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # 5. Permutation importance
    # ------------------------------------------------------------------
    logger.info("Running permutation importance study (%d repeats)…", n_perm_repeats)
    try:
        perm_results = permutation_importance_study(
            pipeline,
            X_test_raw,
            y_test_log,
            feature_cols,
            n_repeats=n_perm_repeats,
        )
        summary.permutation_importance = perm_results

        fig = plot_permutation_importance(perm_results)
        mlflow.log_figure(fig, "investigate/permutation_importance.png")
        plt.close(fig)

        for r in perm_results:
            safe = r.feature.replace(" ", "_")[:40]
            mlflow.log_metric(f"investigate_perm_{safe}", r.importance_mean)

        # Warn about potential noise features
        noise_features = [r.feature for r in perm_results if r.importance_mean < 0]
        if noise_features:
            logger.warning(
                "Potential noise features (shuffling improves score): %s", noise_features
            )
            mlflow.set_tag("investigate_noise_features", ",".join(noise_features))
    except Exception as exc:
        logger.warning("Permutation importance failed: %s", exc)

    logger.info("All investigations complete — artefacts logged under investigate/")
    return summary
