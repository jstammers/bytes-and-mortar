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
  ──────────────────
  • Column selection & validation
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

EPC data is only available for ~50–70% of sales (the join is a left join on postcode + address). The three strategies handle this differently:

| Strategy | Behaviour | Best for |
|---|---|---|
| `impute` (default) | Median imputation for numerics, most-frequent for categoricals | All models; safest default |
| `drop` | Remove rows with any missing feature | Clean comparison; reduces dataset size |
| `passthrough` | Leave `NaN` in place | Perpetual — handles missing values natively |

```bash
bytes-and-mortar train run --missing-strategy passthrough
```

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
α[d] ~ Normal(μ_α, σ_α)      ← district-level random intercepts
μ_α  ~ Normal(log(300k), 1.5) ← global mean log-price prior
σ_α  ~ HalfNormal(1)          ← between-district variation
σ    ~ HalfNormal(0.5)         ← within-district noise
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
| **RMSE** | √(mean((ŷ − y)²)) | Penalises large errors heavily; sensitive to outliers |
| **MAE** | mean(\|ŷ − y\|) | Average error in £; interpretable and robust |
| **MAPE** | mean(\|ŷ − y\| / y) | Scale-independent; useful across price ranges |
| **MdAPE** | median(\|ŷ − y\| / y) | Robust version of MAPE; less affected by outlier properties |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained |

---

## MLflow Experiment Tracking

Each `bytes-and-mortar train run` invocation creates a **run** containing all experiment params, test metrics, evaluation plots, and the fitted model artefact.

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

```
bytes-and-mortar train run [OPTIONS]

Options:
  -m, --model-type     [perpetual|linear]  Model architecture (default: perpetual)
  --budget             FLOAT               Perpetual budget — higher = more accurate,
                                           slower (default: 1.0)
  --data-path          PATH                Path to processed parquet (default: auto)
  --nrows              INT                 Limit rows loaded for development
  --test-years         TEXT                Comma-separated holdout years (default: 2024)
  --missing-strategy   [impute|drop|       Missing value handling
                        passthrough]       (default: impute)
  --log-transform /    --no-log-transform  Log1p-transform price target (default: on)
  --register /         --no-register       Register in MLflow registry (default: off)
  --mlflow-uri         TEXT                MLflow tracking URI
  --numeric-features   TEXT                Comma-separated numeric feature columns
  --categorical-features TEXT              Comma-separated categorical feature columns
```

---

## Python API

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
