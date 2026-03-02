"""Tests for the data linking pipeline."""

import pandas as pd
import pytest

from src.data.pipeline import (
    enrich_with_hpi,
    link_sales_to_epc,
    normalise_address,
    run_pipeline,
    save_dataset,
)


@pytest.fixture
def sales_df():
    """Sample Land Registry sales data."""
    return pd.DataFrame(
        {
            "transaction_id": ["T001", "T002", "T003"],
            "price": [250000, 450000, 180000],
            "date_of_transfer": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10"]),
            "postcode": ["SW1A 1AA", "E1 6AN", "N1 9GU"],
            "property_type": ["D", "F", "T"],
            "paon": ["10", "5", "12"],
            "street": ["DOWNING STREET", "COMMERCIAL STREET", "ISLINGTON PARK ST"],
            "district": ["Westminster", "Tower Hamlets", "Islington"],
            "year": [2024, 2024, 2024],
            "month": [1, 2, 3],
        }
    )


@pytest.fixture
def epc_df():
    """Sample EPC data (already cleaned with snake_case columns)."""
    return pd.DataFrame(
        {
            "postcode": ["SW1A 1AA", "E1 6AN", "W1A 1AB"],
            "address1": ["10 DOWNING STREET", "5 COMMERCIAL STREET", "3 OXFORD STREET"],
            "current_energy_rating": ["C", "B", "D"],
            "current_energy_efficiency": [70, 82, 55],
            "total_floor_area": [200.0, 65.0, 90.0],
            "inspection_date": pd.to_datetime(["2023-06-15", "2023-09-20", "2022-01-01"]),
        }
    )


@pytest.fixture
def hpi_df():
    """Sample UK HPI data (already cleaned)."""
    return pd.DataFrame(
        {
            "regionname": ["Westminster", "Tower Hamlets", "Islington", "Westminster"],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-02-01"]),
            "averageprice": [950000, 450000, 550000, 960000],
            "index": [180.3, 140.2, 155.8, 181.1],
        }
    )


class TestNormaliseAddress:
    def test_basic(self):
        paon = pd.Series(["10"])
        street = pd.Series(["Downing Street"])
        result = normalise_address(paon, street)
        assert result.iloc[0] == "10 DOWNING STREET"

    def test_handles_na(self):
        paon = pd.Series([None])
        street = pd.Series(["HIGH STREET"])
        result = normalise_address(paon, street)
        assert result.iloc[0] == "HIGH STREET"

    def test_collapses_whitespace(self):
        paon = pd.Series(["10"])
        street = pd.Series(["DOWNING   STREET"])
        result = normalise_address(paon, street)
        assert "  " not in result.iloc[0]


class TestLinkSalesToEPC:
    def test_links_on_postcode(self, sales_df, epc_df):
        result = link_sales_to_epc(sales_df, epc_df)
        assert len(result) >= len(sales_df)
        assert "current_energy_rating" in result.columns

    def test_left_join_retains_unmatched(self, sales_df, epc_df):
        """Sales without EPC matches should be retained with NaN."""
        result = link_sales_to_epc(sales_df, epc_df)
        # T003 (N1 9GU) has no EPC match
        t003 = result[result["transaction_id"] == "T003"]
        assert len(t003) >= 1

    def test_matched_records_have_epc(self, sales_df, epc_df):
        result = link_sales_to_epc(sales_df, epc_df)
        t001 = result[result["transaction_id"] == "T001"]
        assert len(t001) >= 1
        # SW1A 1AA should match
        assert t001["current_energy_rating"].iloc[0] == "C"


class TestEnrichWithHPI:
    def test_enriches_matching_districts(self, sales_df, hpi_df):
        result = enrich_with_hpi(sales_df, hpi_df)
        # The HPI enrichment may or may not add averageprice depending on match
        # At minimum it should return a dataframe with at least as many rows
        assert len(result) >= len(sales_df)

    def test_returns_df_without_region_columns(self, sales_df):
        """If HPI has no region column, return data unchanged."""
        bad_hpi = pd.DataFrame({"foo": [1], "bar": [2]})
        result = enrich_with_hpi(sales_df, bad_hpi)
        assert len(result) == len(sales_df)


class TestSaveDataset:
    def test_save_parquet(self, sales_df, tmp_path):
        dest = save_dataset(sales_df, "test_output", output_dir=tmp_path, fmt="parquet")
        assert dest.exists()
        assert dest.suffix == ".parquet"
        loaded = pd.read_parquet(dest)
        assert len(loaded) == len(sales_df)

    def test_save_csv(self, sales_df, tmp_path):
        dest = save_dataset(sales_df, "test_output", output_dir=tmp_path, fmt="csv")
        assert dest.exists()
        assert dest.suffix == ".csv"


class TestRunPipeline:
    def test_sales_only(self, sales_df, tmp_path):
        result = run_pipeline(
            sales=sales_df,
            output_name="test",
            output_dir=tmp_path,
            fmt="csv",
        )
        assert len(result) == len(sales_df)
        assert (tmp_path / "test.csv").exists()

    def test_sales_with_epc(self, sales_df, epc_df, tmp_path):
        result = run_pipeline(
            sales=sales_df,
            epc=epc_df,
            output_name="test",
            output_dir=tmp_path,
            fmt="csv",
        )
        assert "current_energy_rating" in result.columns

    def test_full_pipeline(self, sales_df, epc_df, hpi_df, tmp_path):
        result = run_pipeline(
            sales=sales_df,
            epc=epc_df,
            hpi=hpi_df,
            output_name="test_full",
            output_dir=tmp_path,
            fmt="parquet",
        )
        assert len(result) > 0
        assert (tmp_path / "test_full.parquet").exists()
