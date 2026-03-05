"""Training orchestrator and Typer CLI for UK property price models.

Top-level entry point::

    bytes-and-mortar train run --model-type xgboost --n-trials 50

The ``train()`` function orchestrates:
    1. Load processed data
    2. Temporal train/test split
    3. Optional row drop / imputation based on missing_strategy
    4. Optuna HPO with time-series CV (all three strategies supported)
    5. Retrain best pipeline on full training set
    6. Evaluate model and baseline on held-out test set
    7. Log all params, metrics, and model artefact to MLflow

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
from src.models.config import CVConfig, CVStrategy, Experiment, HPOConfig, ModelType
from src.models.cv import make_cv_splits, temporal_train_test_split
from src.models.evaluate import RegressionMetrics, evaluate_on_test
from src.models.registry import (
    log_and_register_model,
    log_cv_metrics,
    log_params_and_tags,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

train_app = typer.Typer(help="Train and evaluate UK property price models.")


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------


def _load_data(experiment: Experiment) -> pl.DataFrame:
    """Load processed parquet, optionally limiting rows.

    Uses lazy evaluation to efficiently load and filter data. Only loads necessary
    columns (features, target, date, and stratify) to minimize memory usage.
    """
    if not experiment.data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {experiment.data_path}. "
            "Run `bytes-and-mortar run` first to generate the dataset."
        )

    # Determine which columns to load
    required_cols = [
        *experiment.feature_config.all_features,
        "price",  # target column
        "date_of_transfer",  # needed for temporal splits and sorting
    ]

    # Add stratify column if specified
    if experiment.stratify_by and experiment.stratify_by not in required_cols:
        required_cols.append(experiment.stratify_by)

    # Use scan_parquet for lazy evaluation and select only required columns
    lazy_df = pl.scan_parquet(experiment.data_path).select(required_cols)

    if experiment.nrows:
        lazy_df = lazy_df.head(experiment.nrows)

    # Collect to materialize the DataFrame - explicit type annotation for type checker
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
        # Drop rows where any of the feature columns or target have null values
        df = df.drop_nulls(subset=[*config.all_features, target_col])
        logger.info(
            "missing_strategy=drop: removed %d rows (%d remain)",
            before - len(df),
            len(df),
        )
    return df


def _build_objective(model_type: ModelType):
    """Return the Optuna objective function for the given model type."""
    if model_type == ModelType.linear:
        from src.models.linear import linear_objective

        return linear_objective
    elif model_type == ModelType.xgboost:
        from src.models.xgboost_model import xgboost_objective

        return xgboost_objective
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _build_best_pipeline(model_type: ModelType, best_params: dict, feature_config: FeatureConfig):
    """Reconstruct the best pipeline with tuned hyperparameters."""
    if model_type == ModelType.linear:
        from src.models.linear import build_linear_pipeline

        return build_linear_pipeline(
            feature_config=feature_config,
            alpha=best_params["alpha"],
        )
    elif model_type == ModelType.xgboost:
        from src.models.xgboost_model import build_xgboost_pipeline

        return build_xgboost_pipeline(feature_config=feature_config, **best_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _create_evaluation_plots(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Create and log evaluation plots to MLflow.

    Args:
        y_true: True target values (in price space, GBP).
        y_pred_model: Model predictions (in price space, GBP).
        y_pred_baseline: Baseline predictions (in price space, GBP).
        model_name: Name for the model in plot titles.
    """
    # Set style for better-looking plots
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Predicted vs Actual (Model)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred_model, alpha=0.5, s=20, label=model_name)
    ax.scatter(y_true, y_pred_baseline, alpha=0.3, s=10, label="Baseline", color="orange")

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred_model.min(), y_pred_baseline.min())
    max_val = max(y_true.max(), y_pred_model.max(), y_pred_baseline.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel("True Price (£)", fontsize=12)
    ax.set_ylabel("Predicted Price (£)", fontsize=12)
    ax.set_title(f"Predicted vs Actual Price - {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="both")
    plt.tight_layout()
    mlflow.log_figure(fig, "predicted_vs_actual.png")
    plt.close(fig)

    # 2. Residual Plot (Model)
    residuals = y_true - y_pred_model
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred_model, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted Price (£)", fontsize=12)
    ax.set_ylabel("Residual (£)", fontsize=12)
    ax.set_title(f"Residual Plot - {model_name}", fontsize=14, fontweight="bold")
    ax.ticklabel_format(style="plain", axis="both")
    plt.tight_layout()
    mlflow.log_figure(fig, "residuals.png")
    plt.close(fig)

    # 3. Error Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute error distribution
    abs_errors_model = np.abs(residuals)
    abs_errors_baseline = np.abs(y_true - y_pred_baseline)

    axes[0].hist(abs_errors_model, bins=50, alpha=0.7, label=model_name, edgecolor="black")
    axes[0].hist(abs_errors_baseline, bins=50, alpha=0.5, label="Baseline", edgecolor="black")
    axes[0].set_xlabel("Absolute Error (£)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Absolute Error Distribution", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].ticklabel_format(style="plain", axis="x")

    # Percentage error distribution
    pct_errors_model = 100 * residuals / y_true
    pct_errors_baseline = 100 * (y_true - y_pred_baseline) / y_true

    axes[1].hist(pct_errors_model, bins=50, alpha=0.7, label=model_name, edgecolor="black")
    axes[1].hist(pct_errors_baseline, bins=50, alpha=0.5, label="Baseline", edgecolor="black")
    axes[1].set_xlabel("Percentage Error (%)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Percentage Error Distribution", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].set_xlim(-100, 100)  # Focus on reasonable range

    plt.tight_layout()
    mlflow.log_figure(fig, "error_distributions.png")
    plt.close(fig)

    # 4. Model vs Baseline Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_data = {
        "Model": abs_errors_model,
        "Baseline": abs_errors_baseline,
    }
    ax.boxplot(
        list(comparison_data.values()),
        tick_labels=list(comparison_data.keys()),
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
    """Full training run: HPO → retrain → evaluate → log to MLflow.

    MLflow autologging is enabled for sklearn and XGBoost to automatically
    capture training metrics, parameters, and artifacts.

    Args:
        experiment: Fully configured ``Experiment`` object.

    Returns:
        ``(fitted_pipeline, test_metrics)`` tuple.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- Setup MLflow ----
    setup_mlflow(experiment)

    # Enable autologging for sklearn and xgboost
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False,  # We manually log the final model with custom logic
        silent=True,
    )
    mlflow.xgboost.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False,  # We manually log the final model with custom logic
        silent=True,
    )

    with mlflow.start_run(run_name=f"{experiment.model_type}_hpo") as parent_run:
        log_params_and_tags(experiment)
        logger.info("MLflow run_id: %s", parent_run.info.run_id)

        # ---- Load & split data ----
        df = _load_data(experiment)
        # Convert to pandas for validate_features which expects pandas
        df_pd = df.to_pandas()
        validate_features(df_pd, experiment.feature_config)

        # Convert back to polars for efficient processing
        df = pl.from_pandas(df_pd)
        df = _apply_missing_strategy(df, experiment.feature_config)

        # Convert to pandas for cv functions which expect pandas
        df_pd = df.to_pandas()
        train_df_pd, test_df_pd = temporal_train_test_split(
            df_pd,
            test_years=experiment.test_years,
            stratify_col=experiment.stratify_by,
        )

        # Convert back to polars for efficient column operations
        train_df = pl.from_pandas(train_df_pd)
        test_df = pl.from_pandas(test_df_pd)

        feature_cols = experiment.feature_config.all_features
        target_col = "price"

        # Sort by date so row order matches the sorted order that
        # make_cv_splits uses internally when computing positional indices.
        train_df = train_df.sort("date_of_transfer")

        # Select feature columns and convert to pandas for sklearn compatibility
        # Using to_pandas() only when needed for sklearn
        X_train = train_df.select(feature_cols).to_pandas()  # noqa: N806
        X_test = test_df.select(feature_cols).to_pandas()  # noqa: N806

        # Log-transform target - extract as numpy array
        y_train_log = np.log1p(train_df[target_col].cast(pl.Float64).to_numpy())
        y_test_log = np.log1p(test_df[target_col].cast(pl.Float64).to_numpy())

        mlflow.log_metrics(
            {
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        # ---- Baseline (operates on raw data, not log-transformed) ----
        baseline = MedianByGroupBaseline()
        # Convert to pandas for baseline which expects pandas DataFrames
        baseline.fit(train_df.select(feature_cols).to_pandas(), train_df[target_col].to_pandas())
        # Baseline predicts in price space, so wrap in log1p for consistent comparison
        baseline_pred_raw = baseline.predict(X_test)
        baseline_pred_log = np.log1p(np.clip(baseline_pred_raw, 0, None))

        # ---- Time-series CV splits ----
        # Convert to pandas for make_cv_splits which expects pandas
        cv_splits = make_cv_splits(train_df_pd, experiment.cv_config)
        logger.info(
            "Running HPO: %d trials, %d CV folds",
            experiment.hpo_config.n_trials,
            len(cv_splits),
        )

        # ---- Optuna HPO ----
        sampler = (
            optuna.samplers.TPESampler(seed=42)
            if experiment.hpo_config.sampler == "tpe"
            else optuna.samplers.RandomSampler(seed=42)
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )

        objective = _build_objective(experiment.model_type)

        # Log each Optuna trial as a nested MLflow child run
        def _mlflow_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_rmse", trial.value)

        study.optimize(
            lambda trial: objective(
                trial,
                X_train=X_train,
                y_train=y_train_log,
                cv_splits=cv_splits,
                feature_config=experiment.feature_config,
            ),
            n_trials=experiment.hpo_config.n_trials,
            timeout=experiment.hpo_config.timeout_seconds,
            callbacks=[_mlflow_callback],
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_cv_rmse = study.best_value
        logger.info("Best params: %s  Best CV RMSE: %.4f", best_params, best_cv_rmse)
        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.log_metric("best_cv_rmse", best_cv_rmse)

        # Aggregate per-fold metrics from completed trials for the best trial
        # (already logged individually via callback; log summary here)
        completed = [t for t in study.trials if t.value is not None]
        cv_rmses = np.array([t.value for t in completed], dtype=float)
        log_cv_metrics(
            {
                "hpo_cv_rmse_mean": float(cv_rmses.mean()),
                "hpo_cv_rmse_std": float(cv_rmses.std()),
                "hpo_n_completed_trials": len(completed),
            }
        )

        # ---- Retrain on full training set with best params ----
        logger.info("Retraining on full training set with best params")
        best_pipeline = _build_best_pipeline(
            experiment.model_type, best_params, experiment.feature_config
        )
        best_pipeline.fit(X_train, y_train_log)

        # ---- Evaluate on held-out test set ----
        y_pred_log = best_pipeline.predict(X_test)
        test_metrics_dict = evaluate_on_test(
            model_pred=y_pred_log,
            baseline_pred=baseline_pred_log,
            y_true=y_test_log,
            log_transformed=True,
        )
        model_test_metrics = test_metrics_dict["model"]
        baseline_test_metrics = test_metrics_dict["baseline"]

        # Log baseline metrics for comparison
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_test_metrics.to_dict().items()})

        # ---- Create and log evaluation plots ----
        y_true_price = np.expm1(y_test_log)
        y_pred_price = np.expm1(y_pred_log)
        y_baseline_price = np.expm1(baseline_pred_log)
        _create_evaluation_plots(
            y_true=y_true_price,
            y_pred_model=y_pred_price,
            y_pred_baseline=y_baseline_price,
            model_name=experiment.model_type.value,
        )

        # ---- Log and optionally register model ----
        log_and_register_model(
            pipeline=best_pipeline,
            experiment=experiment,
            test_metrics=model_test_metrics,
            X_sample=X_test.head(100),
            y_sample=y_pred_log[:100],
        )

        typer.echo(
            f"\n{'─' * 60}\n"
            f"  Model:    {experiment.model_type}\n"
            f"  CV RMSE:  {best_cv_rmse:.4f} (log-price)\n"
            f"  Test:     {model_test_metrics.summary()}\n"
            f"  Baseline: {baseline_test_metrics.summary()}\n"
            f"  Run ID:   {parent_run.info.run_id}\n"
            f"{'─' * 60}"
        )

    return best_pipeline, model_test_metrics


def predict(
    input_data: pl.DataFrame | pl.LazyFrame,
    model_name: str = "uk_property_price",
    log_transformed: bool = True,
) -> np.ndarray:
    """Load the best registered model and return price predictions.

    Args:
        input_data: Polars DataFrame or LazyFrame with the same feature columns used during training.
        model_name: Registered MLflow model name.
        log_transformed: If ``True``, predictions are exponentiated from log-space.

    Returns:
        1-D array of predicted prices in GBP.
    """
    from src.data.config import MODELS_DIR
    from src.models.registry import load_model

    pipeline = load_model(model_name=model_name, models_dir=MODELS_DIR)

    # Convert to pandas for sklearn pipeline prediction
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
    ] = ModelType.xgboost,
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
        typer.Option(help="Comma-separated years to hold out as test set (e.g. '2023,2024')."),
    ] = "2024",
    cv_strategy: Annotated[
        CVStrategy,
        typer.Option("--cv-strategy", help="Time-series cross-validation strategy."),
    ] = CVStrategy.sliding_window,
    cv_n_splits: Annotated[
        int,
        typer.Option("--cv-n-splits", help="Number of CV folds."),
    ] = 5,
    cv_gap_months: Annotated[
        int,
        typer.Option("--cv-gap-months", help="Gap in months between train end and val start."),
    ] = 1,
    cv_window_months: Annotated[
        int | None,
        typer.Option("--cv-window-months", help="Training window (months) for sliding strategy."),
    ] = 24,
    n_trials: Annotated[
        int,
        typer.Option("--n-trials", help="Number of Optuna HPO trials."),
    ] = 50,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Maximum HPO time in seconds."),
    ] = None,
    missing_strategy: Annotated[
        MissingStrategy,
        typer.Option("--missing-strategy", help="How to handle missing feature values."),
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
        typer.Option("--register/--no-register", help="Register best model in MLflow registry."),
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
        typer.Option(help="Comma-separated list of numeric feature columns to override defaults."),
    ] = None,
    categorical_features: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated list of categorical feature columns to override defaults."
        ),
    ] = None,
) -> None:
    """Train a UK property price model with HPO and MLflow tracking."""
    from src.data.config import PROCESSED_DATA_PATH

    # Parse test years
    parsed_test_years = [int(y.strip()) for y in test_years.split(",")]

    # Build feature config
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
        cv_config=CVConfig(
            strategy=cv_strategy,
            n_splits=cv_n_splits,
            gap_months=cv_gap_months,
            window_months=cv_window_months,
        ),
        hpo_config=HPOConfig(
            n_trials=n_trials,
            timeout_seconds=timeout,
        ),
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
    # Use lazy evaluation for efficient loading
    df: pl.DataFrame
    if input_path.suffix == ".parquet":
        df = pl.scan_parquet(input_path).collect()  # type: ignore[assignment]
    else:
        df = pl.scan_csv(input_path).collect()  # type: ignore[assignment]

    preds = predict(df, model_name=model_name)
    df = df.with_columns(pl.lit(preds).alias("predicted_price"))
    df.write_csv(output_path)
    typer.echo(f"Predictions saved to {output_path} ({len(df)} rows)")
