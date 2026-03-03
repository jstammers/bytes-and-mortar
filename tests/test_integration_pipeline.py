"""Integration tests for the full pipeline using M19 postcode fixture data.

These tests run end-to-end on a real-data subset (M19 postcode area, ~17k rows
per source) to verify correctness and streaming memory behaviour.

Fixture files must be present in data/test_fixtures/raw/ before running.
Generate them with:
    uv run python scripts/create_m19_fixtures.py

Run only integration tests:
    just test-integration

Skip integration tests during normal development:
    just test  # runs unit tests only (integration tests are skipped automatically)
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — needed at runtime by pytest tmp_path fixture

import polars as pl
import pytest

from src.data.config import DATA_DIR
from src.data.pipeline import run_pipeline
from src.data.sources.epc import EPCData
from src.data.sources.land_registry import LandRegistryPricePaid
from src.data.sources.uk_hpi import UKHousePriceIndex

FIXTURE_RAW_DIR = DATA_DIR / "test_fixtures" / "raw"
POSTCODE_PREFIX = "M19"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixture_dir() -> Path:
    """Return fixture directory, skipping if it does not exist."""
    if not FIXTURE_RAW_DIR.exists():
        pytest.skip(
            f"Integration fixture data not found at {FIXTURE_RAW_DIR}. "
            "Run: uv run python scripts/create_m19_fixtures.py"
        )
    return FIXTURE_RAW_DIR


# ---------------------------------------------------------------------------
# Module-scoped fixtures — load each source once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_raw_dir() -> Path:
    return _fixture_dir()


@pytest.fixture(scope="module")
def sales_lf(fixture_raw_dir: Path) -> pl.LazyFrame:
    """Cleaned Land Registry LazyFrame from the M19 fixture."""
    source = LandRegistryPricePaid(raw_dir=fixture_raw_dir)
    return source.clean(source.load())


@pytest.fixture(scope="module")
def epc_lf(fixture_raw_dir: Path) -> pl.LazyFrame:
    """Cleaned EPC LazyFrame from the M19 fixture."""
    source = EPCData(raw_dir=fixture_raw_dir)
    return source.clean(source.load())


@pytest.fixture(scope="module")
def hpi_lf(fixture_raw_dir: Path) -> pl.LazyFrame:
    """Cleaned UK HPI LazyFrame from the M19 fixture."""
    source = UKHousePriceIndex(raw_dir=fixture_raw_dir)
    return source.clean(source.load())


# ---------------------------------------------------------------------------
# Source loading tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestM19SourceLoading:
    def test_land_registry_loads_non_empty(self, sales_lf: pl.LazyFrame) -> None:
        df = sales_lf.collect()
        assert len(df) > 0, "Expected non-empty Land Registry fixture"

    def test_land_registry_all_m19_postcodes(self, sales_lf: pl.LazyFrame) -> None:
        """Every cleaned LR row must have an M19 postcode."""
        df = sales_lf.collect()
        non_m19 = df.filter(~pl.col("postcode").str.starts_with(POSTCODE_PREFIX))
        assert len(non_m19) == 0, (
            f"{len(non_m19)} rows have non-M19 postcodes after cleaning: "
            f"{non_m19['postcode'].unique().to_list()[:5]}"
        )

    def test_land_registry_has_required_columns(self, sales_lf: pl.LazyFrame) -> None:
        schema = sales_lf.collect_schema()
        for col in ("transaction_id", "price", "postcode", "date_of_transfer", "district"):
            assert col in schema, f"Missing expected column: {col}"

    def test_land_registry_no_null_postcodes(self, sales_lf: pl.LazyFrame) -> None:
        df = sales_lf.collect()
        null_count = df["postcode"].null_count()
        assert null_count == 0, f"Found {null_count} null postcodes after clean()"

    def test_land_registry_price_range(self, sales_lf: pl.LazyFrame) -> None:
        """clean() filters extreme prices; all prices should be in the valid range."""
        df = sales_lf.collect()
        assert (df["price"] > 10_000).all(), "Some prices are below the £10k minimum"
        assert (df["price"] < 50_000_000).all(), "Some prices exceed the £50M maximum"

    def test_epc_loads_non_empty(self, epc_lf: pl.LazyFrame) -> None:
        df = epc_lf.collect()
        assert len(df) > 0, "Expected non-empty EPC fixture"

    def test_epc_all_m19_postcodes(self, epc_lf: pl.LazyFrame) -> None:
        """Every cleaned EPC row must have an M19 postcode."""
        df = epc_lf.collect()
        non_m19 = df.filter(~pl.col("postcode").str.starts_with(POSTCODE_PREFIX))
        assert len(non_m19) == 0, (
            f"{len(non_m19)} EPC rows have non-M19 postcodes: "
            f"{non_m19['postcode'].unique().to_list()[:5]}"
        )

    def test_epc_has_required_columns(self, epc_lf: pl.LazyFrame) -> None:
        schema = epc_lf.collect_schema()
        for col in ("postcode", "current_energy_rating", "inspection_date"):
            assert col in schema, f"Missing expected EPC column: {col}"

    def test_hpi_loads_non_empty(self, hpi_lf: pl.LazyFrame) -> None:
        df = hpi_lf.collect()
        assert len(df) > 0, "Expected non-empty UK HPI fixture"


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestM19PipelineIntegration:
    def test_pipeline_sales_only(self, sales_lf: pl.LazyFrame, tmp_path: Path) -> None:
        """Pipeline runs with sales data only (no EPC, no HPI)."""
        result = run_pipeline(
            sales=sales_lf,
            output_name="m19_sales_only",
            output_dir=tmp_path,
            fmt="parquet",
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert (tmp_path / "m19_sales_only.parquet").exists()

    def test_pipeline_one_row_per_sale(
        self, sales_lf: pl.LazyFrame, epc_lf: pl.LazyFrame, tmp_path: Path
    ) -> None:
        """After the EPC join, each original sale must appear exactly once."""
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            output_name="m19_dedup",
            output_dir=tmp_path,
            fmt="parquet",
        )
        expected_rows = sales_lf.select(pl.len()).collect().item()
        assert len(result) == expected_rows, (
            f"Expected {expected_rows} rows (one per sale), got {len(result)}. "
            "The EPC join is creating duplicate sale rows."
        )

    def test_pipeline_epc_join_has_matches(
        self, sales_lf: pl.LazyFrame, epc_lf: pl.LazyFrame, tmp_path: Path
    ) -> None:
        """At least 10% of M19 sales should match an EPC record by postcode."""
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            output_name="m19_epc_matches",
            output_dir=tmp_path,
            fmt="parquet",
        )
        assert "current_energy_rating" in result.columns
        matched = result["current_energy_rating"].drop_nulls()
        match_rate = len(matched) / len(result)
        assert match_rate > 0.1, (
            f"Only {match_rate:.1%} of M19 sales matched an EPC record. "
            "Expected >10%. Check the postcode join logic."
        )

    def test_pipeline_output_postcodes_all_m19(
        self, sales_lf: pl.LazyFrame, epc_lf: pl.LazyFrame, tmp_path: Path
    ) -> None:
        """No non-M19 postcodes should appear in the pipeline output."""
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            output_name="m19_postcodes",
            output_dir=tmp_path,
            fmt="parquet",
        )
        non_m19 = result.filter(~pl.col("postcode").str.starts_with(POSTCODE_PREFIX))
        assert len(non_m19) == 0, (
            f"{len(non_m19)} output rows have non-M19 postcodes — "
            "EPC records from other postcodes are leaking into the join."
        )

    def test_pipeline_full_with_hpi(
        self,
        sales_lf: pl.LazyFrame,
        epc_lf: pl.LazyFrame,
        hpi_lf: pl.LazyFrame,
        tmp_path: Path,
    ) -> None:
        """Full pipeline (sales + EPC + HPI) runs without error and produces output."""
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            hpi=hpi_lf,
            output_name="m19_full",
            output_dir=tmp_path,
            fmt="parquet",
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert (tmp_path / "m19_full.parquet").exists()

    def test_pipeline_hpi_enrichment_joins(
        self,
        sales_lf: pl.LazyFrame,
        hpi_lf: pl.LazyFrame,
        tmp_path: Path,
    ) -> None:
        """HPI enrichment should add average price columns.

        NOTE: The LR 'district' column is stored in UPPERCASE (e.g. 'MANCHESTER')
        but HPI 'RegionName' uses title case ('Manchester'). If this test fails with
        a 0% match rate, the case mismatch between the two sources is the root cause.
        """
        result = run_pipeline(
            sales=sales_lf,
            hpi=hpi_lf,
            output_name="m19_hpi",
            output_dir=tmp_path,
            fmt="parquet",
        )
        hpi_price_cols = [c for c in result.columns if "averageprice" in c.lower()]
        assert hpi_price_cols, (
            "No HPI average price columns found in output. "
            "Check that enrich_with_hpi is selecting the expected columns."
        )
        matched = result[hpi_price_cols[0]].drop_nulls()
        match_rate = len(matched) / len(result)
        assert match_rate > 0.0, (
            "0% of sales matched an HPI record. "
            "Likely cause: LR 'district' is UPPERCASE but HPI 'RegionName' is title case — "
            "normalise one of them before joining."
        )

    def test_pipeline_output_is_parquet(self, sales_lf: pl.LazyFrame, tmp_path: Path) -> None:
        """Parquet output is a valid file that can be read back."""
        run_pipeline(
            sales=sales_lf,
            output_name="m19_roundtrip",
            output_dir=tmp_path,
            fmt="parquet",
        )
        dest = tmp_path / "m19_roundtrip.parquet"
        assert dest.exists()
        reloaded = pl.read_parquet(dest)
        assert len(reloaded) > 0
        assert "transaction_id" in reloaded.columns


# ---------------------------------------------------------------------------
# Deduplication tests — verify no duplicate rows in any pipeline output
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestM19PipelineDeduplication:
    """Verify that the pipeline never introduces duplicate rows.

    These tests run the full pipeline with each combination of sources and
    assert that every transaction_id (and every row) appears exactly once.
    Properties with multiple EPC certificates are the main source of
    fan-out; the pipeline should eliminate those duplicates before returning.
    """

    def test_no_duplicate_transaction_ids_sales_only(
        self, sales_lf: pl.LazyFrame, tmp_path: Path
    ) -> None:
        """Sales-only pipeline has one row per transaction_id."""
        result = run_pipeline(
            sales=sales_lf,
            output_name="m19_dedup_sales",
            output_dir=tmp_path,
            fmt="parquet",
        )
        n_unique = result["transaction_id"].n_unique()
        assert n_unique == len(result), (
            f"Duplicate transaction_ids found: {len(result)} rows but only "
            f"{n_unique} unique transaction_ids."
        )

    def test_no_duplicate_transaction_ids_with_epc(
        self, sales_lf: pl.LazyFrame, epc_lf: pl.LazyFrame, tmp_path: Path
    ) -> None:
        """EPC join does not create duplicate transaction_ids.

        Each sale property may have multiple EPC certificates. The pipeline
        must select the best single match per sale.
        """
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            output_name="m19_dedup_epc",
            output_dir=tmp_path,
            fmt="parquet",
        )
        n_unique = result["transaction_id"].n_unique()
        assert n_unique == len(result), (
            f"EPC join introduced {len(result) - n_unique} duplicate rows — "
            "the per-sale deduplication step is not working correctly."
        )

    def test_no_duplicate_transaction_ids_full_pipeline(
        self,
        sales_lf: pl.LazyFrame,
        epc_lf: pl.LazyFrame,
        hpi_lf: pl.LazyFrame,
        tmp_path: Path,
    ) -> None:
        """Full pipeline (sales + EPC + HPI) has one row per transaction_id."""
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            hpi=hpi_lf,
            output_name="m19_dedup_full",
            output_dir=tmp_path,
            fmt="parquet",
        )
        n_unique = result["transaction_id"].n_unique()
        assert n_unique == len(result), (
            f"Full pipeline produced {len(result) - n_unique} duplicate rows."
        )

    def test_no_duplicate_rows(
        self, sales_lf: pl.LazyFrame, epc_lf: pl.LazyFrame, tmp_path: Path
    ) -> None:
        """No completely identical rows appear in the output (row-level dedup)."""
        result = run_pipeline(
            sales=sales_lf,
            epc=epc_lf,
            output_name="m19_row_dedup",
            output_dir=tmp_path,
            fmt="parquet",
        )
        n_unique_rows = result.unique().shape[0]
        assert n_unique_rows == len(result), (
            f"{len(result) - n_unique_rows} completely identical rows found in output."
        )
