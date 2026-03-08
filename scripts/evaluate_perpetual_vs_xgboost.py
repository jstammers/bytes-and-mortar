"""Evaluate Perpetual GBM vs XGBoost+Optuna vs Median Baseline.

Trains all three on a ~10% random sample of the full dataset and writes a
markdown comparison report to reports/.

Usage::

    uv run python scripts/evaluate_perpetual_vs_xgboost.py
    uv run python scripts/evaluate_perpetual_vs_xgboost.py --n-trials 50
    uv run python scripts/evaluate_perpetual_vs_xgboost.py --sample-frac 0.05

The script is intentionally standalone — it does not use MLflow so that it
can be run without a tracking server.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

# Ensure src/ is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.config import PROCESSED_DATA_PATH
from src.features.build_features import (
    FeatureConfig,
    MedianByGroupBaseline,
    MissingStrategy,
    build_feature_pipeline,
)
from src.models.config import CVConfig, CVStrategy
from src.models.cv import make_cv_splits, temporal_train_test_split
from src.models.evaluate import RegressionMetrics, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_YEARS = [2024]
SEED = 42

# Feature config shared by all ML models
_FEATURE_CONFIG = FeatureConfig(missing_strategy=MissingStrategy.impute)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sample(data_path: Path, fraction: float) -> pl.DataFrame:
    """Load a stratified random sample preserving year proportions."""
    required_cols = list(
        dict.fromkeys(
            [
                *_FEATURE_CONFIG.all_features,
                "price",
                "date_of_transfer",
            ]
        )
    )

    logger.info("Scanning parquet (lazy)…")
    df = pl.scan_parquet(data_path).select(required_cols).collect()
    n_total = len(df)

    # Stratified sample by year so test (2024) rows are proportionally represented
    df = df.with_columns(pl.col("date_of_transfer").dt.year().alias("_year"))
    sampled = df.group_by("_year").map_groups(
        lambda g: g.sample(fraction=fraction, seed=SEED, shuffle=True)
    )
    sampled = sampled.drop("_year")

    logger.info(
        "Loaded %d / %d rows (%.0f%% stratified sample)", len(sampled), n_total, fraction * 100
    )
    return sampled


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    metrics: RegressionMetrics
    time_seconds: float
    extra: dict | None = None  # optional: best params, budget, etc.


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def run_baseline(X_train, y_train_raw, X_test, y_test_raw) -> ModelResult:
    """MedianByGroupBaseline — predicts group-level median price."""
    t0 = time.perf_counter()
    baseline = MedianByGroupBaseline()
    baseline.fit(X_train, y_train_raw)
    y_pred = baseline.predict(X_test)
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(y_test_raw.to_numpy().astype(float), y_pred)
    return ModelResult(metrics=metrics, time_seconds=elapsed)


def run_perpetual(
    X_train,
    y_train_log: np.ndarray,
    X_test,
    y_test_log: np.ndarray,
    budget: float = 0.5,
) -> ModelResult:
    """Perpetual GBM — single fit, no HPO required."""
    from perpetual import PerpetualBooster

    t0 = time.perf_counter()

    feat_pipe = build_feature_pipeline(_FEATURE_CONFIG)
    X_train_t = feat_pipe.fit_transform(X_train, y_train_log)
    X_test_t = feat_pipe.transform(X_test)

    model = PerpetualBooster(objective="SquaredLoss", budget=budget)
    model.fit(X_train_t, y_train_log)

    elapsed = time.perf_counter() - t0

    y_pred = np.expm1(model.predict(X_test_t))
    y_true = np.expm1(y_test_log)
    metrics = compute_metrics(y_true, y_pred)
    return ModelResult(metrics=metrics, time_seconds=elapsed, extra={"budget": budget})


def run_xgboost_optuna(
    X_train,
    y_train_log: np.ndarray,
    X_test,
    y_test_log: np.ndarray,
    cv_splits: list,
    n_trials: int,
) -> ModelResult:
    """XGBoost with full Optuna HPO (sliding-window time-series CV)."""
    import optuna

    from src.models.xgboost_model import build_xgboost_pipeline, xgboost_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.perf_counter()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: xgboost_objective(
            trial,
            X_train=X_train,
            y_train=y_train_log,
            cv_splits=cv_splits,
            feature_config=_FEATURE_CONFIG,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    pipeline = build_xgboost_pipeline(feature_config=_FEATURE_CONFIG, **best_params)
    pipeline.fit(X_train, y_train_log)

    elapsed = time.perf_counter() - t0

    y_pred = np.expm1(pipeline.predict(X_test))
    y_true = np.expm1(y_test_log)
    metrics = compute_metrics(y_true, y_pred)
    return ModelResult(metrics=metrics, time_seconds=elapsed, extra=best_params)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

_MODELS = ["Median Baseline", "Perpetual", "XGBoost + Optuna"]


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def generate_report(
    results: dict[str, ModelResult],
    n_train: int,
    n_test: int,
    sample_fraction: float,
    n_trials: int,
) -> str:
    today = date.today().isoformat()
    baseline_m = results["Median Baseline"].metrics
    perp_m = results["Perpetual"].metrics
    xgb_m = results["XGBoost + Optuna"].metrics
    xgb_t = results["XGBoost + Optuna"].time_seconds
    perp_t = results["Perpetual"].time_seconds

    def rmse_gain(m: RegressionMetrics) -> str:
        pct = (baseline_m.rmse - m.rmse) / baseline_m.rmse
        return f"{pct:+.1%}"

    def mdape_gain(m: RegressionMetrics) -> str:
        diff = m.mdape - baseline_m.mdape
        return f"{diff:+.1%}pp"

    lines = [
        "# Perpetual vs XGBoost+Optuna: Model Comparison",
        "",
        f"**Date:** {today}  ",
        f"**Sample:** {sample_fraction:.0%} of `uk_property_sales.parquet` "
        f"(stratified by year)  ",
        f"**Train rows:** {n_train:,} | **Test rows (2024):** {n_test:,}  ",
        f"**XGBoost HPO:** {n_trials} Optuna trials, 5-fold sliding-window CV  ",
        f"**Perpetual budget:** {results['Perpetual'].extra['budget']}  ",
        "",
        "---",
        "",
        "## Accuracy (held-out 2024 test set, original price space)",
        "",
        "| Model | RMSE (£) | MAE (£) | MAPE | MdAPE | R² | vs baseline RMSE |",
        "|-------|----------|---------|------|-------|----|------------------|",
    ]

    for name in _MODELS:
        m = results[name].metrics
        gain = "" if name == "Median Baseline" else rmse_gain(m)
        lines.append(
            f"| **{name}** | £{m.rmse:,.0f} | £{m.mae:,.0f} | "
            f"{m.mape:.1%} | {m.mdape:.1%} | {m.r2:.4f} | {gain} |"
        )

    lines += [
        "",
        "## Runtime",
        "",
        "| Model | Time | Notes |",
        "|-------|------|-------|",
        f"| Median Baseline | {_fmt_time(results['Median Baseline'].time_seconds)}"
        f" | Group-median lookup |",
        f"| Perpetual | {_fmt_time(perp_t)}"
        f" | Single fit, budget={results['Perpetual'].extra['budget']} |",
        f"| XGBoost + Optuna | {_fmt_time(xgb_t)}"
        f" | {n_trials} trials × 5-fold CV + final refit |",
        "",
        f"**Perpetual speedup over XGBoost+Optuna:** {xgb_t / perp_t:.1f}×",
        "",
        "## XGBoost Best Hyperparameters",
        "",
        "```",
    ]

    for k, v in (results["XGBoost + Optuna"].extra or {}).items():
        lines.append(f"{k}: {v}")

    lines += [
        "```",
        "",
        "## Improvement Over Median Baseline",
        "",
        "| Model | RMSE reduction | MdAPE change |",
        "|-------|---------------|--------------|",
    ]
    for name in ["Perpetual", "XGBoost + Optuna"]:
        m = results[name].metrics
        rmse_pct = (baseline_m.rmse - m.rmse) / baseline_m.rmse
        mdape_pp = m.mdape - baseline_m.mdape
        lines.append(
            f"| {name} | {rmse_pct:+.1%} | {mdape_pp:+.2%}pp |"
        )

    # Recommendation
    lines += ["", "## Recommendation", ""]

    mdape_diff_pp = perp_m.mdape - xgb_m.mdape  # positive = perpetual worse
    rmse_rel_diff = (perp_m.rmse - xgb_m.rmse) / xgb_m.rmse
    speedup = xgb_t / perp_t

    accuracy_tie = abs(mdape_diff_pp) < 0.005 and abs(rmse_rel_diff) < 0.02

    if perp_m.mdape < xgb_m.mdape or accuracy_tie:
        lines.append(
            f"✅ **Favour Perpetual.** "
        )
        if accuracy_tie:
            lines.append(
                f"Accuracy is equivalent (MdAPE difference {mdape_diff_pp:+.2%}pp, "
                f"RMSE difference {rmse_rel_diff:+.1%}). "
            )
        else:
            lines.append(
                f"Perpetual achieves *better* accuracy "
                f"(MdAPE {perp_m.mdape:.1%} vs {xgb_m.mdape:.1%}, "
                f"RMSE reduction {-rmse_rel_diff:.1%}). "
            )
        lines.append(
            f"It is {speedup:.1f}× faster to train and requires no HPO, "
            f"which simplifies the API significantly. "
            f"Replace the XGBoost+Optuna pipeline with a single `PerpetualBooster(budget=0.5).fit(…)` call."
        )
    else:
        lines.append(
            f"⚠️ **Keep XGBoost+Optuna** for now. "
            f"Perpetual is {speedup:.1f}× faster but trades "
            f"{mdape_diff_pp:+.2%}pp in MdAPE "
            f"({perp_m.mdape:.1%} vs {xgb_m.mdape:.1%}). "
            f"Consider testing higher Perpetual budget values (0.7, 1.0) "
            f"to close the accuracy gap before discarding HPO."
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=25, help="Optuna HPO trials for XGBoost (default: 25)"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.10,
        help="Fraction of dataset to sample (default: 0.10)",
    )
    parser.add_argument(
        "--perpetual-budget",
        type=float,
        default=0.5,
        help="Perpetual budget parameter (default: 0.5)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to processed parquet. Defaults to PROCESSED_DATA_PATH from config.",
    )
    args = parser.parse_args()

    logger.info(
        "Config: sample=%.0f%%, n_trials=%d, perpetual_budget=%.2f",
        args.sample_frac * 100,
        args.n_trials,
        args.perpetual_budget,
    )

    # --- Load data ---
    data_path = args.data_path or PROCESSED_DATA_PATH
    df = load_sample(data_path, fraction=args.sample_frac)
    df_pd = df.to_pandas()

    feature_cols = _FEATURE_CONFIG.all_features

    # --- Temporal split ---
    train_pd, test_pd = temporal_train_test_split(
        df_pd, test_years=TEST_YEARS, stratify_col="property_type"
    )
    train_pd = train_pd.sort_values("date_of_transfer")
    logger.info("Train: %d rows | Test (2024): %d rows", len(train_pd), len(test_pd))

    X_train = train_pd[feature_cols]
    X_test = test_pd[feature_cols]
    y_train_raw = train_pd["price"]
    y_test_raw = test_pd["price"]
    y_train_log = np.log1p(y_train_raw.to_numpy().astype(float))
    y_test_log = np.log1p(y_test_raw.to_numpy().astype(float))

    # CV splits for XGBoost HPO
    cv_config = CVConfig(
        strategy=CVStrategy.sliding_window,
        n_splits=5,
        gap_months=1,
        window_months=24,
    )
    cv_splits = make_cv_splits(train_pd, cv_config)

    results: dict[str, ModelResult] = {}

    # --- 1. Median Baseline ---
    logger.info("─── Median Baseline ───")
    results["Median Baseline"] = run_baseline(X_train, y_train_raw, X_test, y_test_raw)
    m = results["Median Baseline"]
    logger.info("  %s  (%.1fs)", m.metrics.summary(), m.time_seconds)

    # --- 2. Perpetual ---
    logger.info("─── Perpetual (budget=%.2f) ───", args.perpetual_budget)
    results["Perpetual"] = run_perpetual(
        X_train, y_train_log, X_test, y_test_log, budget=args.perpetual_budget
    )
    m = results["Perpetual"]
    logger.info("  %s  (%s)", m.metrics.summary(), _fmt_time(m.time_seconds))

    # --- 3. XGBoost + Optuna ---
    logger.info("─── XGBoost + Optuna (%d trials) ───", args.n_trials)
    results["XGBoost + Optuna"] = run_xgboost_optuna(
        X_train, y_train_log, X_test, y_test_log, cv_splits, n_trials=args.n_trials
    )
    m = results["XGBoost + Optuna"]
    logger.info("  %s  (%s)", m.metrics.summary(), _fmt_time(m.time_seconds))

    # --- Report ---
    report = generate_report(
        results=results,
        n_train=len(train_pd),
        n_test=len(test_pd),
        sample_fraction=args.sample_frac,
        n_trials=args.n_trials,
    )

    reports_dir = Path(__file__).parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"perpetual_vs_xgboost_{date.today().isoformat()}.md"
    out_path.write_text(report)

    print("\n" + "═" * 70)
    print(report)
    print("═" * 70)
    logger.info("Report saved → %s", out_path)


if __name__ == "__main__":
    main()
