# CLAUDE.md

This file provides context to Claude Code about the bytes-and-mortar project.

## Project overview

UK property data ingestion pipeline that downloads, cleans, links, and saves data from three public sources:

- **HM Land Registry Price Paid Data** - property transaction records for England and Wales
- **EPC (Energy Performance Certificates)** - building energy efficiency data (requires free API key)
- **UK House Price Index** - ONS area-level price statistics

## Tech stack

- **Python 3.11+** with **uv** for dependency management
- **pandas** for data processing
- **typer** for CLI interface
- **ruff** for linting and formatting
- **ty** for type checking
- **pytest** for testing
- **just** for task running

## Common commands

```bash
just install        # Install all dependencies
just test           # Run tests
just check          # Run all checks (lint + format + typecheck + tests)
just fmt            # Format code
just lint           # Run linter
just typecheck      # Run type checker
```

## Project structure

```
src/
  data/
    config.py           # URLs, column definitions, mappings
    make_dataset.py     # Typer CLI entrypoint
    pipeline.py         # Data linking/merging logic
    sources/
      base.py           # Abstract DataSource base class
      land_registry.py  # HM Land Registry Price Paid
      epc.py            # Energy Performance Certificates
      uk_hpi.py         # UK House Price Index
tests/                  # pytest tests mirroring src structure
```

## Code style

- Formatted with ruff (line length 100)
- Type-checked with ty
- All public data sources inherit from `DataSource` base class
- Each source implements `download()`, `load()`, `clean()` methods

## Environment variables

- `EPC_API_TOKEN` - required for EPC data. Register at https://epc.opendatacommunities.org/login
