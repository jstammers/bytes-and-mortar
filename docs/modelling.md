# Modelling: UK Property Price Prediction

This document describes the design philosophy, modelling pipeline, and usage instructions for the `src/models` and `src/features` modules.

---

## Overview

The modelling layer sits downstream of the data pipeline. It reads the linked parquet output from `bytes-and-mortar run` and trains a regression model to predict residential sale price.

```
data/processed/uk_property_sales.parquet
        │
        ▼
  Feature Engineering          src/features/build_features.py
  ──────────────────           src/features/derived.py
  • Column selection & validation
  • Optional derived features (log_floor_area, floor_area_per_room,
    energy_rating_numeric)
  • Missing value handling
  • Ordinal encoding (low-cardinality categoricals)
  • Target encoding (district, ~300 categories)
  • Standard scaling (numerics)
        │
        ▼
  Temporal Train/Test Split    src/models/cv.py
  ─────────────────────────
  • Hold out one or more calendar years as final test set
  • Log class-distribution balance by property_type
        │
        ▼
  Fit Model                    src/models/train_model.py
  ──────────────────────────
  • PerpetualBooster(budget=1.0) — self-tuning, no HPO loop
  • Single fit on full training set
        │
        ▼
  Evaluate vs Baseline         src/models/evaluate.py
  ────────────────────
  • RMSE, MAE, MAPE, MdAPE, R²
  • Compare against MedianByGroupBaseline(property_type × district)
        │
        ▼
  Log to MLflow + Save Locally  src/models/registry.py
  ────────────────────────────
  • Params, metrics, model artefact per run
  • Optional: push to MLflow model registry
  • Always: save .joblib file to models/
        │
        ▼
  Investigate (optional)        src/models/investigate.py
  ──────────────────────
  • SHAP feature attribution via PerpetualBooster.predict_contributions()
  • Imputation bias — imputed vs complete-case performance split
  • Calibration — reliability diagram and bias by price decile
  • Regional performance — MdAPE by county
  • Permutation importance — noise feature detection
```

---

## Feature Engineering

### Configuration

Feature selection is driven by `FeatureConfig`, which is constructed by the CLI and wired into every downstream component. The defaults cover the most informative columns available in the linked dataset:

| Group | Columns | Transformer |
|---|---|---|
| **Numeric** | `year`, `month`, `total_floor_area`, `current_energy_efficiency`, `number_habitable_rooms`, `averageprice` | `StandardScaler` |
| **Categorical** | `property_type`, `old_new`, `duration`, `county`, `postcode_outward` | `OrdinalEncoder` |
| **Target-encoded** | `district` | sklearn `TargetEncoder` (CV=5) |

Override any group from the CLI:

```bash
bytes-and-mortar train run \
  --numeric-features "year,month,total_floor_area" \
  --categorical-features "property_type,old_new"
```

All columns are validated against the actual DataFrame schema before training begins, raising a clear `ValueError` listing any missing columns.

### Missing value strategies

EPC data is only available for ~50-70% of sales (the join is a left join on postcode + address). The three strategies handle this differently:

| Strategy | Behaviour | Best for |
|---|---|---|
| `impute` (default) | Median imputation for numerics, most-frequent for categoricals | All models; safest default |
| `drop` | Remove rows with any missing feature | Clean comparison; reduces dataset size |
| `passthrough` | Leave `NaN` in place | Perpetual — handles missing values natively |

```bash
bytes-and-mortar train run --missing-strategy passthrough
```

### Derived features

**File:** `src/features/derived.py`

`DerivedFeatureTransformer` is an optional sklearn-compatible preprocessing step that computes three new numeric columns before the main `ColumnTransformer`:

| Feature | Formula | Rationale |
|---|---|---|
| `log_floor_area` | `log1p(total_floor_area)` | Floor area is right-skewed; log transform improves its linear relationship with log-price |
| `floor_area_per_room` | `total_floor_area / max(rooms, 1)` | Space efficiency (density) — a proxy for property quality independent of absolute size |
| `energy_rating_numeric` | A=7, B=6, ..., G=1 | Treats EPC letter grade as a continuous ordinal signal rather than a nominal category |

Enable derived features via `FeatureConfig.derived_features=True` or the `--derived-features` CLI flag:

```bash
# Train with derived features
bytes-and-mortar train investigate --derived-features

# Or use the Python API
from src.features.build_features import FeatureConfig
config = FeatureConfig(derived_features=True)
```

The transformer is inserted as the first pipeline step; the `ColumnTransformer` then picks up the new columns automatically via `FeatureConfig.effective_numeric_features`.

### Target encoding for `district`

`district` has ~300 unique values and is the strongest geographic signal available. One-hot encoding would add 300 sparse columns; label encoding would impose a false ordinal relationship. Target encoding maps each district to a smoothed mean of `log(price)` computed with cross-validation inside the training fold, preventing target leakage.

For the Bayesian model (see below), district is encoded as an integer index and given a hierarchical prior — a more statistically principled alternative.

### Log transformation of price

UK house prices are strongly right-skewed (£50k bedsits to £10M penthouses). A `log1p` transformation maps the distribution to approximate normality, which is a prerequisite for the linear model and also makes the GBM loss landscape more balanced. All metrics are reported in original GBP space after `expm1` inversion.

---

## Model Architectures

### 1. Perpetual GBM (default)

**File:** `src/models/perpetual_model.py`

A self-tuning gradient boosting machine from [perpetual-ml](https://github.com/perpetual-ml/perpetual). Perpetual automatically selects the number of trees based on a single `budget` parameter — no hyperparameter search loop required.

**Benchmarks on 10% of the full UK property dataset (2.9M train rows, 87k 2024 test rows):**

| Model | MdAPE | RMSE (£) | Training time |
|---|---|---|---|
| Perpetual (budget=1.0) | 16.1% | £288,002 | 33s |
| Median baseline | 41.8% | £403,374 | <1s |

**The `budget` parameter:**

| Value | MdAPE | Time | When to use |
|---|---|---|---|
| 0.5 | 16.7% | 16s | Fast iteration, dev testing |
| 0.7 | 16.2% | 20s | Balanced speed/accuracy |
| 1.0 (default) | 16.1% | 33s | Production |

Increase `budget` further if accuracy is still insufficient after reviewing residuals.

### 2. Ridge regression on log-price

**File:** `src/models/linear.py`

A simple interpretable model: sklearn `RidgeCV` trained on `log1p(price)`. `RidgeCV` selects the optimal L2 regularisation strength via leave-one-out cross-validation over a log-spaced grid `[1e-3, 1e3]` — no external HPO required.

Ridge is preferred over plain OLS because UK property features are correlated (floor area, rooms, energy efficiency) and regularisation prevents overfitting on smaller regional datasets.

Linear models require imputed inputs — if `missing_strategy=passthrough` is specified, the pipeline automatically falls back to `impute` with a warning.

### 3. Bayesian hierarchical model (experimental)

**File:** `notebooks/pymc_property_price.py`

A PyMC-based Bayesian hierarchical regression explored in a [marimo](https://marimo.io) interactive notebook. The hierarchy is:

```
price_i ~ LogNormal(μ_i, σ)
μ_i  = α[district_i] + β_floor · floor_area_i + β_eff · energy_eff_i
α[d] ~ Normal(μ_α, σ_α)      <- district-level random intercepts
μ_α  ~ Normal(log(300k), 1.5) <- global mean log-price prior
σ_α  ~ HalfNormal(1)          <- between-district variation
σ    ~ HalfNormal(0.5)         <- within-district noise
```

**Why a hierarchical model?**

Partial pooling allows data-sparse districts (e.g. rural areas with few transactions) to borrow strength from the global distribution, rather than producing unreliable point estimates. This is particularly valuable when training on sub-regional datasets.

**Running the notebook:**

```bash
just install-notebooks
uv run marimo edit notebooks/pymc_property_price.py
```

**Integration path:** The notebook documents a `BayesianPropertyModel` sklearn wrapper stub. Once the experimental phase is complete, this can be promoted to `src/models/bayesian.py` and added to `ModelType` in `src/models/config.py`.

---

## Baseline

Every training run is automatically compared against a `MedianByGroupBaseline`:

- Predicts the median sale price for each `(property_type, district)` pair observed during training
- Fallback chain: group median → `property_type` median → global median (for unseen combinations)
- Logged alongside model metrics under `baseline_*` keys in MLflow

**Interpretation:** A model that does not beat the baseline adds no value over a simple lookup table. This is a minimum viable bar, not a high one.

---

## Cross-Validation Utilities

The functions in `src/models/cv.py` are available for exploratory analysis but are not part of the default training loop. The Perpetual and Ridge training paths fit directly on the full training set without k-fold CV — Perpetual self-tunes via `budget`; `RidgeCV` self-selects alpha via leave-one-out CV internally.

If you want to run CV splits manually (e.g. for model comparison studies), three strategies are supported:

- **Sliding window** — fixed-size training window advances per fold
- **Expanding window** — training set grows with each fold
- **Year-based** — one calendar year held out per fold

All strategies support a `gap_months` buffer between the end of the training window and the start of validation, preventing leakage from properties that take weeks to register with the Land Registry.

See `scripts/evaluate_perpetual_vs_xgboost.py` for an example of manual CV use in a comparative evaluation.

---

## Evaluation Metrics

All metrics are computed in **original GBP price space** (after `expm1` inversion) for interpretability:

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | sqrt(mean((y_hat - y)^2)) | Penalises large errors heavily; sensitive to outliers |
| **MAE** | mean(\|y_hat - y\|) | Average error in GBP; interpretable and robust |
| **MAPE** | mean(\|y_hat - y\| / y) | Scale-independent; useful across price ranges |
| **MdAPE** | median(\|y_hat - y\| / y) | Robust version of MAPE; less affected by outlier properties |
| **R²** | 1 - SS_res/SS_tot | Proportion of variance explained |

---

## Model Investigation and Explainability

**File:** `src/models/investigate.py`

The `train investigate` command trains a model and then runs a diagnostic suite designed to answer five questions about prediction errors.  All outputs are logged to MLflow under an `investigate/` artefact subfolder and viewable in the MLflow UI.

### Running an investigation

```bash
# Full investigation with defaults (Perpetual, budget=0.5 for speed)
bytes-and-mortar train investigate

# Higher-fidelity SHAP analysis with more samples
bytes-and-mortar train investigate --n-shap-samples 5000 --budget 1.0

# Compare baseline vs a model enriched with derived features
bytes-and-mortar train investigate --derived-features

# Change the regional breakdown column
bytes-and-mortar train investigate --region-col county
```

### 1. SHAP feature attribution

Uses `PerpetualBooster.predict_contributions()` to compute SHAP-style feature attributions without requiring the external `shap` library. Contributions are aligned to the ColumnTransformer output feature names and subsampled (default 2000 rows) for speed.

**Artefacts logged:**
- `investigate/shap_summary.png` — horizontal bar chart of mean |SHAP value| per feature (descending importance)
- `investigate/shap_dependence_<feature>.png` — scatter of transformed feature value vs SHAP value for the top-3 most important features

**What to look for:**
- Features with near-zero mean |SHAP| are contributing little signal and are candidates for removal (cross-reference with permutation importance)
- A flat dependence plot suggests a feature adds noise; a clear monotone trend confirms genuine predictive value
- Unexpected patterns (e.g. SHAP values that reverse direction mid-range) can indicate feature interactions or data quality issues

### 2. Imputation bias

Splits the test set into two groups — rows where at least one feature was `NaN` before the pipeline imputed it, and rows that were fully observed — and computes metrics separately.

**Artefact logged:** `investigate/imputation_bias.png`

**What to look for:**
- If imputed rows have materially higher RMSE or MdAPE than complete-case rows, the median/most-frequent fill is introducing systematic bias
- A large imputed fraction (>30%) means a significant share of predictions rely on filled values; consider `--missing-strategy drop` for a cleaner comparison
- Possible remedies: better EPC join coverage, feature-specific imputation strategies, or dropping EPC features with low join rates

### 3. Calibration

Sorts predictions into equal-count bins (default 10) and compares the mean predicted price against the mean actual price in each bin. A well-calibrated model should produce a near-diagonal reliability diagram.

**Artefact logged:** `investigate/calibration.png` (reliability diagram + bias-by-decile bar chart)

**What to look for:**
- Systematic over-prediction at the top of the price distribution (an upward bow in the reliability diagram) is common when training on right-skewed data
- Systematic under-prediction at the low end suggests the log-price transformation may not fully correct the skew, or that very cheap properties are under-represented
- A consistently positive bias across all bins indicates the model is globally overconfident upward — consider checking for target leakage

### 4. Regional performance

Computes RMSE, MAE, MAPE, MdAPE, and R² separately for each region (default: `county`). Regions with fewer than 30 test-set rows are excluded to avoid noisy estimates.

**Artefact logged:** `investigate/regional_performance.png`

**What to look for:**
- Counties far above the cross-region MdAPE average are underfit — they likely have distinctive price dynamics not captured by the shared model features
- High-error regions often correlate with sparse training data, unusual property mixes (e.g. rural counties with many large detached houses), or missing EPC coverage
- Consider adding region-specific features (e.g. interaction terms, or separate models per region) if the spread is large

### 5. Permutation importance

Shuffles each feature column independently and measures the average increase in MSE (log-space). Features with high `importance_mean` are load-bearing; features with `importance_mean ≈ 0` or negative (shuffling *helps*) are potential noise sources.

**Artefact logged:** `investigate/permutation_importance.png`

**What to look for:**
- Red bars (negative importance) flag features where the model does better without the feature's real signal — strong indicator of noise or overfitting
- Features with low importance and high correlation to another feature can safely be dropped without accuracy loss
- Cross-reference with SHAP: a feature with low mean |SHAP| and low permutation importance is a reliable removal candidate

### Python API

```python
import mlflow
import numpy as np
from src.models.investigate import run_investigations

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("uk_property_price/investigate")

with mlflow.start_run():
    summary = run_investigations(
        pipeline=fitted_pipeline,
        X_train=X_train,
        y_train_log=y_train_log,
        X_test_raw=X_test,      # pre-transform DataFrame — NaNs preserved
        y_test_log=y_test_log,
        feature_cols=feature_config.all_features,
        region_col="county",
        n_calibration_bins=10,
        n_shap_samples=2000,
        n_perm_repeats=5,
    )

# Access results programmatically
print("Worst calibration bias:", max(abs(b.bias) for b in summary.calibration_bins))
print("Imputed fraction:", summary.imputation_bias.imputed_fraction)
print("Top feature (SHAP):", summary.shap_feature_names[np.argmax(summary.shap_mean_abs)])
```

---

## MLflow Experiment Tracking

Each `bytes-and-mortar train run` invocation creates a **run** containing all experiment params, test metrics, evaluation plots, and the fitted model artefact.

Each `bytes-and-mortar train investigate` invocation creates one or two runs (a second if `--derived-features` is set) under an `investigate` experiment name, containing the standard test metrics plus all investigation artefacts.

Experiments are organised as `<name>/<model_type>` (e.g. `uk_property_price/perpetual`) so different architectures can be filtered and compared in the MLflow UI.

### Launching the UI

```bash
just mlflow-ui
# Opens at http://localhost:5000
```

> The `just mlflow-ui` recipe uses a SQLite backend (`mlruns.db`) which supports the full MLflow model registry. The default `file://` store supports artefact logging only.

### Model registry

To register the best model after training:

```bash
bytes-and-mortar train run --register --mlflow-uri sqlite:///mlruns.db
```

Registered models can be loaded by name and stage:

```python
from src.models.registry import load_model

pipeline = load_model("uk_property_price", stage="Production")
predictions = pipeline.predict(X_new)
```

---

## CLI Reference

### `train run` — train and evaluate a model

```
bytes-and-mortar train run [OPTIONS]

Options:
  -m, --model-type       [perpetual|linear]   Model architecture (default: perpetual)
  --budget               FLOAT                Perpetual budget (default: 1.0)
  --data-path            PATH                 Path to processed parquet (default: auto)
  --nrows                INT                  Limit rows loaded for development
  --test-years           TEXT                 Comma-separated holdout years (default: 2024)
  --missing-strategy     [impute|drop|         Missing value handling (default: impute)
                          passthrough]
  --log-transform /      --no-log-transform   Log1p-transform price target (default: on)
  --register /           --no-register        Register in MLflow registry (default: off)
  --mlflow-uri           TEXT                 MLflow tracking URI
  --numeric-features     TEXT                 Comma-separated numeric feature columns
  --categorical-features TEXT                 Comma-separated categorical feature columns
```

### `train investigate` — train then run diagnostic investigations

```
bytes-and-mortar train investigate [OPTIONS]

Options:
  -m, --model-type        [perpetual|linear]   Model architecture (default: perpetual)
  --budget                FLOAT                Perpetual budget (default: 0.5)
  --data-path             PATH                 Path to processed parquet (default: auto)
  --nrows                 INT                  Limit rows loaded
  --test-years            TEXT                 Comma-separated holdout years (default: 2024)
  --missing-strategy      [impute|drop|         Missing value handling (default: impute)
                           passthrough]
  --mlflow-uri            TEXT                 MLflow tracking URI
  --region-col            TEXT                 Column for regional breakdown (default: county)
  --n-shap-samples        INT                  Rows subsampled for SHAP (default: 2000)
  --n-perm-repeats        INT                  Permutation importance repeats (default: 5)
  --derived-features /    --no-derived-features Also run a second experiment with
                                               log_floor_area, floor_area_per_room,
                                               energy_rating_numeric added (default: off)
```

**MLflow artefacts produced by `train investigate`:**

| Artefact | Contents |
|---|---|
| `investigate/shap_summary.png` | Mean \|SHAP\| per feature (bar chart) |
| `investigate/shap_dependence_<feat>.png` | SHAP dependence for top-3 features |
| `investigate/imputation_bias.png` | MdAPE/RMSE split: imputed vs complete rows |
| `investigate/calibration.png` | Reliability diagram + bias by price decile |
| `investigate/regional_performance.png` | MdAPE by county (top 30 shown) |
| `investigate/permutation_importance.png` | MSE increase when each feature is shuffled |

### `train predict` — generate predictions from a registered model

```
bytes-and-mortar train predict INPUT_PATH [OPTIONS]

Arguments:
  INPUT_PATH    Path to parquet or CSV file with feature columns

Options:
  -o, --output  PATH    Output CSV path (default: predictions.csv)
  --model-name  TEXT    Registered MLflow model name (default: uk_property_price)
```

---

## Python API

### Training

```python
from src.models.config import Experiment, PerpetualConfig, ModelType
from src.features.build_features import FeatureConfig, MissingStrategy
from src.models.train_model import train, predict

experiment = Experiment(
    model_type=ModelType.perpetual,
    feature_config=FeatureConfig(
        numeric_features=["year", "month", "total_floor_area", "current_energy_efficiency"],
        categorical_features=["property_type", "old_new", "duration"],
        target_encode_features=["district"],
        missing_strategy=MissingStrategy.impute,
    ),
    perpetual_config=PerpetualConfig(budget=1.0),
    test_years=[2024],
    register_model=True,
)

pipeline, metrics = train(experiment)
print(metrics.summary())

# Later: load best model and predict
predictions = predict(input_df, model_name="uk_property_price")
```

### Training with derived features

```python
from src.features.build_features import FeatureConfig, MissingStrategy

config = FeatureConfig(
    missing_strategy=MissingStrategy.impute,
    derived_features=True,  # adds log_floor_area, floor_area_per_room, energy_rating_numeric
)
# config.all_features   -> original raw column names (for loading / validation)
# config.effective_numeric_features -> numeric columns including the three derived ones
```

### Using `DerivedFeatureTransformer` directly

```python
from src.features.derived import DerivedFeatureTransformer

transformer = DerivedFeatureTransformer()
X_enriched = transformer.fit_transform(X_raw)
# New columns: log_floor_area, floor_area_per_room, energy_rating_numeric
```

---

## Adding a New Model Architecture

1. Create `src/models/<name>.py` with a `build_<name>_pipeline(feature_config, **params) -> Pipeline` function
2. Add `<name>` to `ModelType` in `src/models/config.py`
3. Add a corresponding config dataclass (e.g. `<Name>Config`) if the model has hyperparameters
4. Wire into `_build_pipeline()` in `src/models/train_model.py`
5. Add tests in `tests/models/test_train_model.py`

The LightGBM case, for example, would be:

```python
# src/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from src.features.build_features import FeatureConfig, build_feature_pipeline

def build_lightgbm_pipeline(feature_config: FeatureConfig | None = None, **lgb_params) -> Pipeline:
    feature_pipeline = build_feature_pipeline(feature_config or FeatureConfig())
    return Pipeline([
        ("feature_pipeline", feature_pipeline),
        ("model", lgb.LGBMRegressor(**lgb_params)),
    ])
```
