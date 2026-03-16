"""Tests for HPI storage separation in the pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path  # noqa: TC003

import polars as pl
import pytest

from src.data.pipeline import save_hpi_regional
from src.features.build_features import join_hpi_to_sales


@pytest.fixture
def hpi_df() -> pl.DataFrame:
    """Minimal cleaned HPI DataFrame (one region, two months)."""
    return pl.DataFrame(
        {
            "regionname": ["LONDON", "LONDON", "MANCHESTER", "MANCHESTER"],
            "date": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 1, 1), date(2023, 2, 1)],
            "year": [2023, 2023, 2023, 2023],
            "month": [1, 2, 1, 2],
            "averageprice": [500_000.0, 510_000.0, 200_000.0, 205_000.0],
            "index": [130.0, 131.0, 115.0, 116.0],
        }
    )


@pytest.fixture
def sales_df() -> pl.DataFrame:
    """Minimal sales DataFrame with district/year/month columns."""
    return pl.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "price": [480_000, 195_000, 220_000],
            "district": ["LONDON", "MANCHESTER", "BIRMINGHAM"],
            "year": [2023, 2023, 2023],
            "month": [1, 2, 1],
        }
    )


class TestSaveHpiRegional:
    def test_creates_parquet(self, tmp_path: Path, hpi_df: pl.DataFrame) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        assert dest.exists()
        assert dest.suffix == ".parquet"

    def test_custom_output_name(self, tmp_path: Path, hpi_df: pl.DataFrame) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path, output_name="my_hpi")
        assert dest.name == "my_hpi.parquet"

    def test_parquet_contains_region_column(self, tmp_path: Path, hpi_df: pl.DataFrame) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = pl.read_parquet(dest)
        assert "regionname" in result.columns

    def test_parquet_contains_price_columns(self, tmp_path: Path, hpi_df: pl.DataFrame) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = pl.read_parquet(dest)
        assert "averageprice" in result.columns

    def test_parquet_excludes_non_hpi_columns(self, tmp_path: Path) -> None:
        """Columns like transaction_id should not appear in the regional parquet."""
        df = pl.DataFrame(
            {
                "regionname": ["LONDON"],
                "date": [date(2023, 1, 1)],
                "year": [2023],
                "month": [1],
                "averageprice": [500_000.0],
                "transaction_id": ["T1"],  # sales column — should not be kept
            }
        )
        dest = save_hpi_regional(df.lazy(), output_dir=tmp_path)
        result = pl.read_parquet(dest)
        assert "transaction_id" not in result.columns

    def test_accepts_dataframe(self, tmp_path: Path, hpi_df: pl.DataFrame) -> None:
        """save_hpi_regional should accept a DataFrame as well as a LazyFrame."""
        dest = save_hpi_regional(hpi_df, output_dir=tmp_path)
        assert dest.exists()

    def test_row_count_preserved(self, tmp_path: Path, hpi_df: pl.DataFrame) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = pl.read_parquet(dest)
        assert len(result) == len(hpi_df)


class TestJoinHpiToSales:
    def test_returns_lazy_frame(
        self, tmp_path: Path, sales_df: pl.DataFrame, hpi_df: pl.DataFrame
    ) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = join_hpi_to_sales(sales_df.lazy(), dest)
        assert isinstance(result, pl.LazyFrame)

    def test_hpi_columns_added(
        self, tmp_path: Path, sales_df: pl.DataFrame, hpi_df: pl.DataFrame
    ) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = join_hpi_to_sales(sales_df.lazy(), dest).collect()
        assert "averageprice" in result.columns

    def test_matching_region_gets_values(
        self, tmp_path: Path, sales_df: pl.DataFrame, hpi_df: pl.DataFrame
    ) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = join_hpi_to_sales(sales_df.lazy(), dest).collect()
        london_row = result.filter(pl.col("district") == "LONDON")
        assert london_row["averageprice"][0] == 500_000.0

    def test_unmatched_region_gets_null(
        self, tmp_path: Path, sales_df: pl.DataFrame, hpi_df: pl.DataFrame
    ) -> None:
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = join_hpi_to_sales(sales_df.lazy(), dest).collect()
        birmingham_row = result.filter(pl.col("district") == "BIRMINGHAM")
        assert birmingham_row["averageprice"][0] is None

    def test_row_count_unchanged(
        self, tmp_path: Path, sales_df: pl.DataFrame, hpi_df: pl.DataFrame
    ) -> None:
        """Left join must never add rows."""
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = join_hpi_to_sales(sales_df.lazy(), dest).collect()
        assert len(result) == len(sales_df)

    def test_missing_hpi_path_returns_sales_unchanged(
        self, tmp_path: Path, sales_df: pl.DataFrame
    ) -> None:
        missing = tmp_path / "nonexistent.parquet"
        result = join_hpi_to_sales(sales_df.lazy(), missing).collect()
        assert list(result.columns) == list(sales_df.columns)
        assert len(result) == len(sales_df)

    def test_sales_missing_district_returns_unchanged(
        self, tmp_path: Path, hpi_df: pl.DataFrame
    ) -> None:
        sales_no_district = pl.DataFrame({"price": [100_000], "year": [2023], "month": [1]})
        dest = save_hpi_regional(hpi_df.lazy(), output_dir=tmp_path)
        result = join_hpi_to_sales(sales_no_district.lazy(), dest).collect()
        # Without district, join is skipped and columns are unchanged
        assert "averageprice" not in result.columns
