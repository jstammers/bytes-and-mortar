# Changelog

All notable changes to this project will be documented in this file.
## [0.2.0] - 2026-03-08

### Bug Fixes

- Resolve ty type-checker errors in pipeline and land_registry

### Ci/Cd

- Fix git-cliff template filter and add pull-requests permission
- (**changelog**) Checkout master before pushing generated changelog

### Features

- Rename package to bytes-and-mortar and add CI/CD
- Migrate from pandas to polars for dataframe processing
- (**pipeline**) Migrate to polars streaming API with temporal EPC join
- (**sources**) Parquet-first loading for all data sources (#9)
- (**app,frontend**) Add property valuation web app with FastAPI backend and React UI (#4)
- (**models**) Add ML training pipeline with HPO and MLflow tracking (#5)
- (**train**) Log visualisations and enable mlflow autologging
- (**data**) Add Perpetual vs XGBoost+Optuna evaluation
- (**models**) Deprecate XGBoost+Optuna, promote Perpetual GBM as default
- (**models**) Remove XGBoost+Optuna — perpetual-only from v0.2.0

### Performance

- (**pipeline**) Two-phase EPC join + partitioned parquet output (#8)
- (**data**) Fix OOM in EPC join with two-pass streaming execution (#10)

### Refactoring

- (**data**) Use bulk donwloading for EPC Certificates

### Testing

- (**integration**) Add M19 postcode integration tests and fix pipeline bugs


