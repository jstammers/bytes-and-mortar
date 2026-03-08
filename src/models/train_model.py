"""Training orchestrator and Typer CLI for UK property price models.

Top-level entry point::

    bytes-and-mortar train run --model-type perpetual --budget 1.0

The ``train()`` function orchestrates:
    1. Load processed data
    2. Temporal train/test split
    3. Optional row drop / imputation based on missing_strategy
    4. Fit model (no HPO loop — Perpetual self-tunes via ``budget``)
    5. Evaluate model and baseline on held-out test set
    6. Log all params, metrics, and model artefact to MLflow

The Typer sub-application ``train_app`` is attached to the main CLI via
``app.add_typer(train_app, name="train")`` in ``make_dataset.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
import typer

from src.features.build_features import (
    FeatureConfig,
    MedianByGroupBaseline,
    MissingStrategy,
    validate_features,
)
from src.models.config import Experiment, ModelType, PerpetualConfig
from src.models.cv import temporal_train_test_split
from src.models.evaluate import RegressionMetrics, evaluate_on_test
from src.models.registry import (
    log_and_register_model,
    log_params_and_tags,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

train_app = typer.Typer(help="Train and evaluate UK property price models.")


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------


def _load_data(experiment: Experiment) -> pl.DataFrame:
    """Load processed parquet, optionally limiting rows."""
    if not experiment.data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {experiment.data_path}. "
            "Run `bytes-and-mortar run` first to generate the dataset."
        )

    required_cols = [
        *experiment.feature_config.all_features,
        "price",
        "date_of_transfer",
    ]
    if experiment.stratify_by and experiment.stratify_by not in required_cols:
        required_cols.append(experiment.stratify_by)

    lazy_df = pl.scan_parquet(experiment.data_path).select(required_cols)
    if experiment.nrows:
        lazy_df = lazy_df.head(experiment.nrows)

    df: pl.DataFrame = lazy_df.collect()  # type: ignore[assignment]
    logger.info(
        "Loaded %d rows x %d columns from %s",
        len(df),
        len(df.columns),
        experiment.data_path,
    )
    return df


def _apply_missing_strategy(
    df: pl.DataFrame, config: FeatureConfig, target_col: str = "price"
) -> pl.DataFrame:
    """Drop rows with missing feature values when strategy=drop."""
    if config.missing_strategy == MissingStrategy.drop:
        before = len(df)
        df = df.drop_nulls(subset=[*config.all_features, target_col])
        logger.info(
            "missing_strategy=drop: removed %d rows (%d remain)",
            before - len(df),
            len(df),
        )
    return df


def _build_pipeline(experiment: Experiment):
    """Return an unfitted pipeline for the configured model type."""
    if experiment.model_type == ModelType.perpetual:
        from src.models.perpetual_model import build_perpetual_pipeline

        return build_perpetual_pipeline(
            feature_config=experiment.feature_config,
            budget=experiment.perpetual_config.budget,
            objective=experiment.perpetual_config.objective,
        )
    elif experiment.model_type == ModelType.linear:
        from src.models.linear import build_linear_pipeline

        return build_linear_pipeline(feature_config=experiment.feature_config)
    else:
        raise ValueError(f"Unsupported model_type: {experiment.model_type}")


def _create_evaluation_plots(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Create and log evaluation plots to MLflow."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred_model, alpha=0.5, s=20, label=model_name)
    ax.scatter(y_true, y_pred_baseline, alpha=0.3, s=10, label="Baseline", color="orange")
    min_val = min(y_true.min(), y_pred_model.min(), y_pred_baseline.min())
    max_val = max(y_true.max(), y_pred_model.max(), y_pred_baseline.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("True Price (£)", fontsize=12)
    ax.set_ylabel("Predicted Price (£)", fontsize=12)
    ax.set_title(f"Predicted vs Actual Price — {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="both")
    plt.tight_layout()
    mlflow.log_figure(fig, "predicted_vs_actual.png")
    plt.close(fig)

    # 2. Residual Plot
    residuals = y_true - y_pred_model
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred_model, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted Price (£)", fontsize=12)
    ax.set_ylabel("Residual (£)", fontsize=12)
    ax.set_title(f"Residual Plot — {model_name}", fontsize=14, fontweight="bold")
    ax.ticklabel_format(style="plain", axis="both")
    plt.tight_layout()
    mlflow.log_figure(fig, "residuals.png")
    plt.close(fig)

    # 3. Error Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    abs_errors_model = np.abs(residuals)
    abs_errors_baseline = np.abs(y_true - y_pred_baseline)
    axes[0].hist(abs_errors_model, bins=50, alpha=0.7, label=model_name, edgecolor="black")
    axes[0].hist(abs_errors_baseline, bins=50, alpha=0.5, label="Baseline", edgecolor="black")
    axes[0].set_xlabel("Absolute Error (£)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Absolute Error Distribution", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].ticklabel_format(style="plain", axis="x")
    pct_errors_model = 100 * residuals / y_true
    pct_errors_baseline = 100 * (y_true - y_pred_baseline) / y_true
    axes[1].hist(pct_errors_model, bins=50, alpha=0.7, label=model_name, edgecolor="black")
    axes[1].hist(pct_errors_baseline, bins=50, alpha=0.5, label="Baseline", edgecolor="black")
    axes[1].set_xlabel("Percentage Error (%)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Percentage Error Distribution", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].set_xlim(-100, 100)
    plt.tight_layout()
    mlflow.log_figure(fig, "error_distributions.png")
    plt.close(fig)

    # 4. Model vs Baseline boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [abs_errors_model, abs_errors_baseline],
        tick_labels=[model_name, "Baseline"],
        showfliers=False,
    )
    ax.set_ylabel("Absolute Error (£)", fontsize=12)
    ax.set_title("Model vs Baseline: Error Comparison", fontsize=14, fontweight="bold")
    ax.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    mlflow.log_figure(fig, "model_vs_baseline.png")
    plt.close(fig)

    logger.info("Logged 4 evaluation plots to MLflow")


def train(experiment: Experiment) -> tuple[object, RegressionMetrics]:
    """Full training run: fit → evaluate → log to MLflow.

    For ``ModelType.perpetual`` (default): fits a single PerpetualBooster
    with the configured ``budget`` — no HPO or CV loop required.

    For ``ModelType.linear``: fits RidgeCV which self-selects alpha — no
    external HPO required.

    Args:
        experiment: Fully configured ``Experiment`` object.

    Returns:
        ``(fitted_pipeline, test_metrics)`` tuple.
    """
    setup_mlflow(experiment)
    # mlflow.sklearn.autolog() is intentionally omitted: it monkey-patches
    # Pipeline.predict() globally which triggers sklearn >= 1.6's check_is_fitted()
    # → get_tags() path that requires __sklearn_tags__, causing AttributeError with
    # PerpetualBooster. All params, metrics, plots, and artefacts are logged manually.

    run_name = f"{experiment.model_type}_train"
    with mlflow.start_run(run_name=run_name) as parent_run:
        log_params_and_tags(experiment)
        logger.info("MLflow run_id: %s", parent_run.info.run_id)

        # ---- Load & split data ----
        df = _load_data(experiment)
        df_pd = df.to_pandas()
        validate_features(df_pd, experiment.feature_config)

        df = pl.from_pandas(df_pd)
        df = _apply_missing_strategy(df, experiment.feature_config)
        df_pd = df.to_pandas()

        train_df_pd, test_df_pd = temporal_train_test_split(
            df_pd,
            test_years=experiment.test_years,
            stratify_col=experiment.stratify_by,
        )
        train_df_pd = train_df_pd.sort_values("date_of_transfer")

        feature_cols = experiment.feature_config.all_features
        X_train = train_df_pd[feature_cols]
        X_test = test_df_pd[feature_cols]
        y_train_log = np.log1p(train_df_pd["price"].to_numpy().astype(float))
        y_test_log = np.log1p(test_df_pd["price"].to_numpy().astype(float))

        mlflow.log_metrics({"n_train": len(X_train), "n_test": len(X_test)})

        # ---- Baseline ----
        baseline = MedianByGroupBaseline()
        baseline.fit(X_train, train_df_pd["price"])
        baseline_pred_raw = baseline.predict(X_test)
        baseline_pred_log = np.log1p(np.clip(baseline_pred_raw, 0, None))

        # ---- Fit model ----
        pipeline = _build_pipeline(experiment)
        logger.info("Fitting %s model…", experiment.model_type)
        pipeline.fit(X_train, y_train_log)
        y_pred_log = pipeline.predict(X_test)
        test_metrics_dict = evaluate_on_test(
            model_pred=y_pred_log,
            baseline_pred=baseline_pred_log,
            y_true=y_test_log,
            log_transformed=True,
        )
        test_metrics = test_metrics_dict["model"]
        baseline_test_metrics = test_metrics_dict["baseline"]

        mlflow.log_metrics(
            {f"baseline_{k}": v for k, v in baseline_test_metrics.to_dict().items()}
        )

        y_true_price = np.expm1(y_test_log)
        y_pred_price = np.expm1(y_pred_log)
        y_baseline_price = np.expm1(baseline_pred_log)
        _create_evaluation_plots(
            y_true=y_true_price,
            y_pred_model=y_pred_price,
            y_pred_baseline=y_baseline_price,
            model_name=experiment.model_type.value,
        )

        log_and_register_model(
            pipeline=pipeline,
            experiment=experiment,
            test_metrics=test_metrics,
            X_sample=X_test.head(100),
            y_sample=pipeline.predict(X_test.head(100)),
        )

        typer.echo(
            f"\n{'─' * 60}\n"
            f"  Model:    {experiment.model_type}\n"
            f"  Test:     {test_metrics.summary()}\n"
            f"  Run ID:   {parent_run.info.run_id}\n"
            f"{'─' * 60}"
        )

    return pipeline, test_metrics


def predict(
    input_data: pl.DataFrame | pl.LazyFrame,
    model_name: str = "uk_property_price",
    log_transformed: bool = True,
) -> np.ndarray:
    """Load the best registered model and return price predictions.

    Args:
        input_data: Polars DataFrame or LazyFrame with feature columns.
        model_name: Registered MLflow model name.
        log_transformed: If ``True``, predictions are exponentiated from log-space.

    Returns:
        1-D array of predicted prices in GBP.
    """
    from src.data.config import MODELS_DIR
    from src.models.registry import load_model

    pipeline = load_model(model_name=model_name, models_dir=MODELS_DIR)

    if isinstance(input_data, pl.LazyFrame):
        df: pl.DataFrame = input_data.collect()  # type: ignore[assignment]
    else:
        df = input_data
    input_pd = df.to_pandas()

    raw_preds = pipeline.predict(input_pd)
    return np.expm1(raw_preds) if log_transformed else raw_preds


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@train_app.command("run")
def train_command(
    model_type: Annotated[
        ModelType,
        typer.Option("--model-type", "-m", help="Model architecture to train."),
    ] = ModelType.perpetual,
    budget: Annotated[
        float,
        typer.Option(
            "--budget",
            help="Perpetual budget parameter — higher values improve accuracy at the cost "
            "of training time. Only used when --model-type=perpetual.",
        ),
    ] = 1.0,
    data_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to processed parquet file. Defaults to data/processed/uk_property_sales.parquet."
        ),
    ] = None,
    nrows: Annotated[
        int | None,
        typer.Option(help="Limit number of rows loaded (useful for dev/testing)."),
    ] = None,
    test_years: Annotated[
        str,
        typer.Option(
            help="Comma-separated years to hold out as test set (e.g. '2023,2024')."
        ),
    ] = "2024",
    missing_strategy: Annotated[
        MissingStrategy,
        typer.Option(
            "--missing-strategy", help="How to handle missing feature values."
        ),
    ] = MissingStrategy.impute,
    log_transform_target: Annotated[
        bool,
        typer.Option(
            "--log-transform/--no-log-transform",
            help="Log1p-transform sale price target.",
        ),
    ] = True,
    register: Annotated[
        bool,
        typer.Option(
            "--register/--no-register", help="Register best model in MLflow registry."
        ),
    ] = False,
    mlflow_uri: Annotated[
        str | None,
        typer.Option(
            "--mlflow-uri",
            help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var or local mlruns/.",
        ),
    ] = None,
    numeric_features: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated list of numeric feature columns to override defaults."
        ),
    ] = None,
    categorical_features: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated list of categorical feature columns to override defaults."
        ),
    ] = None,
) -> None:
    """Train a UK property price model with MLflow tracking."""
    from src.data.config import PROCESSED_DATA_PATH

    parsed_test_years = [int(y.strip()) for y in test_years.split(",")]

    feat_config = FeatureConfig(
        missing_strategy=missing_strategy,
        log_transform_target=log_transform_target,
    )
    if numeric_features:
        feat_config.numeric_features = [c.strip() for c in numeric_features.split(",")]
    if categorical_features:
        feat_config.categorical_features = [c.strip() for c in categorical_features.split(",")]

    experiment = Experiment(
        model_type=model_type,
        feature_config=feat_config,
        perpetual_config=PerpetualConfig(budget=budget),
        data_path=data_path or PROCESSED_DATA_PATH,
        nrows=nrows,
        test_years=parsed_test_years,
        register_model=register,
        mlflow_tracking_uri=mlflow_uri or Experiment().mlflow_tracking_uri,
    )

    train(experiment)


@train_app.command("predict")
def predict_command(
    input_path: Annotated[
        Path,
        typer.Argument(help="Path to parquet or CSV file with feature columns."),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output CSV path for predictions."),
    ] = Path("predictions.csv"),
    model_name: Annotated[
        str,
        typer.Option(help="Registered MLflow model name."),
    ] = "uk_property_price",
) -> None:
    """Generate price predictions from a trained model."""
    df: pl.DataFrame
    if input_path.suffix == ".parquet":
        df = pl.scan_parquet(input_path).collect()  # type: ignore[assignment]
    else:
        df = pl.scan_csv(input_path).collect()  # type: ignore[assignment]

    preds = predict(df, model_name=model_name)
    df = df.with_columns(pl.lit(preds).alias("predicted_price"))
    df.write_csv(output_path)
    typer.echo(f"Predictions saved to {output_path} ({len(df)} rows)")
