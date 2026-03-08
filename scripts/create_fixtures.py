"""Create postcode-filtered fixtures for pipeline integration tests.

Reads raw CSV files from data/raw/ and writes postcode-filtered subsets to
data/test_fixtures/raw/, which is used by tests/test_integration_pipeline.py.

HPI regions are derived automatically from the district values found in the
filtered Land Registry data, so no region list needs to be specified manually.

Usage (from project root):
    uv run python scripts/create_fixtures.py --postcode-prefix SW1
    uv run python scripts/create_fixtures.py --postcode-prefix W1 --src-dir /path/to/raw
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

FIXTURE_RAW_DIR = DATA_DIR / "test_fixtures" / "raw"


def create_land_registry_fixture(
    src_dir: Path, dest_dir: Path, postcode_prefix: str
) -> tuple[Path, list[str]]:
    """Filter Land Registry pp-complete.csv to the given postcode prefix.

    Returns the path to the written fixture and a list of title-cased HPI region
    names discovered from the district values in the filtered data.

    The source file has no header row and all values are plain CSV strings.
    The output preserves this format (no header) so that LandRegistryPricePaid.load()
    can read it unchanged.
    """
    src = src_dir / "pp-complete.csv"
    dest = dest_dir / "pp-complete.csv"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run the download commands first.")

    logger.info("Filtering Land Registry data for postcodes starting with %s...", postcode_prefix)
    schema = {c: pl.String for c in LAND_REGISTRY_COLUMNS}
    lf = pl.scan_csv(src, has_header=False, schema=schema)

    # Postcode in the LR CSV has no braces; strip whitespace only to be safe.
    filtered = lf.filter(pl.col("postcode").str.strip_chars().str.starts_with(postcode_prefix))

    df = filtered.collect()

    # Discover district names for HPI region filtering (LR stores UPPERCASE; HPI title-cases).
    hpi_regions = sorted({d.title() for d in df["district"].drop_nulls().to_list() if d})

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write without header to match the source format expected by load().
    df.write_csv(dest, include_header=False)
    logger.info("  Written %d rows to %s", len(df), dest)
    logger.info("  Discovered HPI regions: %s", hpi_regions)
    return dest, hpi_regions


def create_epc_fixture(src_dir: Path, dest_dir: Path, postcode_prefix: str) -> Path:
    """Filter EPC bulk CSV to the given postcode prefix.

    The source file has a header row; the output preserves it.
    """
    src = src_dir / "epc_domestic_bulk.csv"
    dest = dest_dir / "epc_domestic_bulk.csv"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run the download commands first.")

    logger.info("Filtering EPC data for postcodes starting with %s...", postcode_prefix)
    lf = pl.scan_csv(src, schema_overrides=_EPC_SCAN_OVERRIDES, null_values=_EPC_NULL_VALUES)
    filtered = lf.filter(pl.col("postcode").str.starts_with(postcode_prefix))

    df = filtered.collect()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dest)
    logger.info("  Written %d rows to %s", len(df), dest)
    return dest


def create_hpi_fixture(src_dir: Path, dest_dir: Path, hpi_regions: list[str]) -> Path:
    """Filter UK HPI data to the regions derived from the Land Registry fixture.

    The HPI data is area-level (not postcode-level). The regions to include are
    derived automatically from the district values in the filtered Land Registry data.

    Note: LR stores districts in UPPERCASE (e.g. 'MANCHESTER'); HPI uses title case
    ('Manchester'). The pipeline join on district + year/month will only match when the
    cases align — the integration test surfaces this behaviour.
    """
    src = src_dir / "uk_hpi_full.csv"
    dest = dest_dir / "uk_hpi_full.csv"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run the download commands first.")

    logger.info("Filtering UK HPI data for regions: %s...", hpi_regions)
    lf = pl.scan_csv(src, infer_schema_length=10000)
    filtered = lf.filter(pl.col("RegionName").is_in(hpi_regions))

    df = filtered.collect()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dest)
    logger.info("  Written %d rows to %s", len(df), dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create postcode-filtered fixture data for integration tests."
    )
    parser.add_argument(
        "--postcode-prefix",
        required=True,
        help="Postcode prefix to filter by (e.g. SW1, W1, EC1). Case-insensitive.",
    )
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

    postcode_prefix = args.postcode_prefix.strip().upper()
    logger.info(
        "Creating fixtures for postcode prefix %r: %s → %s",
        postcode_prefix,
        args.src_dir,
        args.dest_dir,
    )

    _, hpi_regions = create_land_registry_fixture(args.src_dir, args.dest_dir, postcode_prefix)
    create_epc_fixture(args.src_dir, args.dest_dir, postcode_prefix)
    create_hpi_fixture(args.src_dir, args.dest_dir, hpi_regions)

    # Write postcode prefix metadata so integration tests can discover it automatically.
    prefix_file = args.dest_dir / "postcode_prefix.txt"
    prefix_file.write_text(postcode_prefix)
    logger.info("  Written postcode prefix to %s", prefix_file)

    logger.info(
        "\nFixtures written to %s\nRun integration tests with:\n  just test-integration",
        args.dest_dir,
    )


if __name__ == "__main__":
    main()
