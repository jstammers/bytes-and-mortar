"""MLflow experiment tracking, model logging, and registry management.

Provides a thin wrapper around the MLflow Python API so that ``train_model``
can interact with MLflow through clean, typed helper functions.

Tracking URI priority:
    1. ``Experiment.mlflow_tracking_uri`` (set in config, defaults to
       ``MLFLOW_TRACKING_URI`` env var if present)
    2. ``file://<project>/mlruns`` (local filesystem — no server required)

Model registry requires a database-backed URI (SQLite or Postgres).  The
``just mlflow-ui`` recipe launches a server with SQLite that supports full
registry features.  The local ``file://`` store supports artifact logging but
not the model registry; ``register_model=True`` will be a no-op with a warning
when using the default file store.

Usage::

    from src.models.registry import setup_mlflow, log_and_register_model

    setup_mlflow(experiment)
    with mlflow.start_run(run_name="perpetual_train") as run:
        log_and_register_model(pipeline, experiment, metrics, run)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

    from src.models.config import Experiment
    from src.models.evaluate import RegressionMetrics

logger = logging.getLogger(__name__)


def setup_mlflow(experiment: Experiment) -> str:
    """Configure MLflow tracking URI and create (or retrieve) the experiment.

    Must be called once before any ``mlflow.start_run()`` calls.

    Args:
        experiment: Experiment configuration.

    Returns:
        MLflow experiment ID.
    """
    mlflow.set_tracking_uri(experiment.mlflow_tracking_uri)

    # Experiment name = "<name>/<model_type>" for easy filtering in the UI
    experiment_name = f"{experiment.name}/{experiment.model_type}"
    mlflow_exp = mlflow.set_experiment(experiment_name)

    logger.info(
        "MLflow tracking URI: %s  experiment: '%s' (id=%s)",
        experiment.mlflow_tracking_uri,
        experiment_name,
        mlflow_exp.experiment_id,
    )
    return mlflow_exp.experiment_id


def log_params_and_tags(experiment: Experiment) -> None:
    """Log all experiment parameters and standard tags to the active MLflow run.

    Must be called inside an ``mlflow.start_run()`` context.

    Args:
        experiment: Experiment configuration.
    """
    mlflow.log_params(experiment.to_mlflow_params())
    mlflow.set_tags(
        {
            "model_type": experiment.model_type.value,
            "data_path": str(experiment.data_path),
        }
    )


def log_cv_metrics(cv_summary: dict[str, float]) -> None:
    """Log cross-validation summary metrics to the active MLflow run.

    Args:
        cv_summary: Output of ``cv_metrics_summary()``.
    """
    mlflow.log_metrics(cv_summary)


def log_and_register_model(
    pipeline: Pipeline,
    experiment: Experiment,
    test_metrics: RegressionMetrics,
    X_sample: pd.DataFrame | None = None,  # noqa: N803
    y_sample: np.ndarray | None = None,
) -> str | None:
    """Log the fitted pipeline to MLflow and optionally push to the registry.

    Logs the model artefact with an inferred input/output signature (if sample
    data is provided).  Also saves a local joblib copy to ``experiment.models_dir``.

    Args:
        pipeline: Fitted sklearn Pipeline to log.
        experiment: Experiment configuration.
        test_metrics: Final held-out test metrics for tagging.
        X_sample: Small sample of input features for signature inference.
        y_sample: Corresponding predictions or true values for signature.

    Returns:
        MLflow run ID of the logged model, or ``None`` if logging failed.
    """
    import joblib

    # Log test metrics
    prefixed = {f"test_{k}": v for k, v in test_metrics.to_dict().items()}
    mlflow.log_metrics(prefixed)

    # Infer model signature from sample data
    signature = None
    if X_sample is not None and y_sample is not None:
        try:
            signature = infer_signature(X_sample, y_sample)
        except Exception as exc:
            logger.warning("Could not infer model signature: %s", exc)

    # Log sklearn model artefact
    artifact_path = f"{experiment.model_type}_model"
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path=artifact_path,
        signature=signature,
        input_example=X_sample.head(5) if X_sample is not None else None,
    )

    # Also save a local copy for inference without MLflow
    experiment.models_dir.mkdir(parents=True, exist_ok=True)
    local_path = experiment.models_dir / f"{experiment.model_type}_best.joblib"
    joblib.dump(pipeline, local_path)
    logger.info("Saved model locally to %s", local_path)

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None

    # Register model in MLflow registry (requires database-backed tracking URI)
    if experiment.register_model and run_id:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=experiment.model_name,
            )
            logger.info(
                "Registered model '%s' version %s (run_id=%s)",
                experiment.model_name,
                result.version,
                run_id,
            )
        except Exception as exc:
            logger.warning(
                "Model registration skipped (requires database-backed tracking URI): %s",
                exc,
            )

    return run_id


def load_model(
    model_name: str,
    stage: str = "Production",
    models_dir: Path | None = None,
) -> Pipeline:
    """Load a model from the MLflow registry or local fallback.

    Tries the MLflow registry first; falls back to the local joblib file in
    ``models_dir`` if the registry lookup fails (e.g. file:// tracking store).

    Args:
        model_name: Registered model name (must match ``Experiment.model_name``).
        stage: Model stage to load (``"Production"``, ``"Staging"``, etc.).
        models_dir: Directory containing local joblib fallback.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If neither registry nor local model can be found.
    """
    import joblib

    try:
        model_uri = f"models:/{model_name}/{stage}"
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded model from registry: %s (%s)", model_name, stage)
        return pipeline
    except Exception as exc:
        logger.warning("Registry load failed (%s), trying local fallback", exc)

    if models_dir is not None:
        local_path = models_dir / f"{model_name}_best.joblib"
        if local_path.exists():
            pipeline = joblib.load(local_path)
            logger.info("Loaded model from local file: %s", local_path)
            return pipeline

    raise FileNotFoundError(
        f"Could not load model '{model_name}'. "
        "Ensure the model is registered or a local .joblib file exists."
    )
