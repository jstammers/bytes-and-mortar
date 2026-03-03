"""Create M19 postcode subset fixtures for pipeline integration tests.

Reads raw CSV files from data/raw/ and writes M19-filtered subsets to
data/test_fixtures/raw/, which is used by tests/test_integration_pipeline.py.

Usage (from project root):
    uv run python scripts/create_m19_fixtures.py
    uv run python scripts/create_m19_fixtures.py --src-dir /path/to/raw
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src.data.config import DATA_DIR, EPC_DOMESTIC_SCHEMA, LAND_REGISTRY_COLUMNS, RAW_DIR

_EPC_NULL_VALUES = ["", "N/A", "NO DATA!", "INVALID!", "null", "NULL"]
# Force string-typed columns only; let polars infer dates/numerics as strings from CSV.
_EPC_SCAN_OVERRIDES: dict[str, type[pl.DataType]] = {
    col: dtype for col, dtype in EPC_DOMESTIC_SCHEMA.items() if dtype in (pl.Utf8, pl.String)
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POSTCODE_PREFIX = "M19"
FIXTURE_RAW_DIR = DATA_DIR / "test_fixtures" / "raw"

# HPI RegionName values that correspond to districts found in M19 Land Registry data.
# LR districts for M19 are uppercase (e.g. "MANCHESTER"); HPI uses title case ("Manchester").
M19_HPI_REGIONS = ["Manchester", "Bury", "Wigan", "Stockport", "Bolton", "Greater Manchester"]


def create_land_registry_fixture(src_dir: Path, dest_dir: Path) -> Path:
    """Filter Land Registry pp-complete.csv to M19 postcodes.

    The source file has no header row and all values are plain CSV strings.
    The output preserves this format (no header) so that LandRegistryPricePaid.load()
    can read it unchanged.
    """
    src = src_dir / "pp-complete.csv"
    dest = dest_dir / "pp-complete.csv"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run the download commands first.")

    logger.info("Filtering Land Registry data for postcodes starting with %s...", POSTCODE_PREFIX)
    schema = {c: pl.String for c in LAND_REGISTRY_COLUMNS}
    lf = pl.scan_csv(src, has_header=False, schema=schema)

    # Postcode in the LR CSV has no braces; strip whitespace only to be safe.
    filtered = lf.filter(
        pl.col("postcode").str.strip_chars().str.starts_with(POSTCODE_PREFIX)
    )

    df = filtered.collect()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write without header to match the source format expected by load().
    df.write_csv(dest, include_header=False)
    logger.info("  Written %d rows to %s", len(df), dest)
    return dest


def create_epc_fixture(src_dir: Path, dest_dir: Path) -> Path:
    """Filter EPC bulk CSV to M19 postcodes.

    The source file has a header row; the output preserves it.
    """
    src = src_dir / "epc_domestic_bulk.csv"
    dest = dest_dir / "epc_domestic_bulk.csv"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run the download commands first.")

    logger.info("Filtering EPC data for postcodes starting with %s...", POSTCODE_PREFIX)
    lf = pl.scan_csv(src, schema_overrides=_EPC_SCAN_OVERRIDES, null_values=_EPC_NULL_VALUES)
    filtered = lf.filter(pl.col("postcode").str.starts_with(POSTCODE_PREFIX))

    df = filtered.collect()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dest)
    logger.info("  Written %d rows to %s", len(df), dest)
    return dest


def create_hpi_fixture(src_dir: Path, dest_dir: Path) -> Path:
    """Filter UK HPI data to regions relevant to the M19 postcode area.

    The HPI data is area-level (not postcode-level). M19 falls within the
    Greater Manchester region; the districts found in the LR data for M19
    postcodes are MANCHESTER, BURY, WIGAN, STOCKPORT, BOLTON.

    Note: LR stores districts in UPPERCASE; HPI uses title case. The pipeline
    join on district + year/month will only match when the cases align —
    the integration test surfaces this behaviour.
    """
    src = src_dir / "uk_hpi_full.csv"
    dest = dest_dir / "uk_hpi_full.csv"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run the download commands first.")

    logger.info("Filtering UK HPI data for regions: %s...", M19_HPI_REGIONS)
    lf = pl.scan_csv(src, infer_schema_length=10000)
    filtered = lf.filter(pl.col("RegionName").is_in(M19_HPI_REGIONS))

    df = filtered.collect()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dest)
    logger.info("  Written %d rows to %s", len(df), dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create M19 postcode fixture data.")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing raw CSV files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=FIXTURE_RAW_DIR,
        help=f"Output directory for fixture files (default: {FIXTURE_RAW_DIR})",
    )
    args = parser.parse_args()

    logger.info("Creating M19 fixtures from %s → %s", args.src_dir, args.dest_dir)
    create_land_registry_fixture(args.src_dir, args.dest_dir)
    create_epc_fixture(args.src_dir, args.dest_dir)
    create_hpi_fixture(args.src_dir, args.dest_dir)

    logger.info(
        "\nFixtures written to %s\n"
        "Run integration tests with:\n"
        "  just test-integration",
        args.dest_dir,
    )


if __name__ == "__main__":
    main()
