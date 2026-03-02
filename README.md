# bytes-and-morter

A data pipeline for UK property valuation, combining Land Registry transactions, Energy Performance Certificates, and House Price Index data.

## Data sources

| Source | Description | License |
|--------|-------------|---------|
| [HM Land Registry Price Paid](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | Property sale transactions in England & Wales since 1995 | OGL v3.0 |
| [EPC Register](https://epc.opendatacommunities.org/) | Energy ratings, floor area, construction age, heating type | OGL v3.0 |
| [UK House Price Index](https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-may-2025) | ONS monthly area-level price indices and volumes | OGL v3.0 |

## Getting started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [just](https://github.com/casey/just) for task running (optional)

### Installation

```bash
uv sync --extra dev
```

Or with just:

```bash
just install
```

### Configuration

Create a `.env` file in the project root:

```env
EPC_API_TOKEN=your_token_here
```

Register for a free EPC API key at https://epc.opendatacommunities.org/login

## Usage

### CLI commands

```bash
# Download Land Registry data for 2024
uv run bytes-and-morter download-land-registry --year 2024

# Download EPC data for Westminster
uv run bytes-and-morter download-epc --local-authority E09000033

# Download UK House Price Index
uv run bytes-and-morter download-hpi

# Run the full pipeline (download + clean + link + save)
uv run bytes-and-morter run --year 2024

# Run pipeline on already-downloaded data
uv run bytes-and-morter run --skip-download

# Skip EPC if you don't have an API token
uv run bytes-and-morter run --year 2024 --skip-epc
```

### Using just

```bash
just download-lr 2024           # Download Land Registry for 2024
just download-epc E09000033     # Download EPC for Westminster
just download-hpi               # Download UK HPI
just run --year 2024            # Run full pipeline
```

## Development

```bash
just check       # Run all checks (lint + format + typecheck + tests)
just test        # Run tests only
just test-v      # Run tests with verbose output
just lint        # Run ruff linter
just fmt         # Format code with ruff
just typecheck   # Run ty type checker
just clean       # Remove caches and compiled files
```

## Project structure

```
bytes-and-morter/
├── src/
│   ├── data/
│   │   ├── config.py           # URLs, column definitions, data mappings
│   │   ├── make_dataset.py     # Typer CLI entrypoint
│   │   ├── pipeline.py         # Data linking and merging logic
│   │   └── sources/
│   │       ├── base.py         # Abstract DataSource base class
│   │       ├── land_registry.py
│   │       ├── epc.py
│   │       └── uk_hpi.py
│   ├── features/               # Feature engineering (planned)
│   ├── models/                 # Model training and prediction (planned)
│   └── visualization/          # Visualizations (planned)
├── tests/                      # Test suite
├── data/
│   ├── raw/                    # Downloaded source files
│   ├── interim/                # Intermediate transforms
│   └── processed/              # Final linked datasets
├── notebooks/                  # Jupyter notebooks for exploration
├── pyproject.toml              # Project config and dependencies
└── justfile                    # Task runner commands
```

## Pipeline overview

1. **Download** raw CSV data from each source
2. **Load** into pandas DataFrames with correct dtypes
3. **Clean** each source (standardise postcodes, parse dates, filter invalid records)
4. **Link** Land Registry sales to EPC records on postcode + address matching
5. **Enrich** with UK HPI area-level statistics on district + year-month
6. **Save** the linked dataset as Parquet or CSV
