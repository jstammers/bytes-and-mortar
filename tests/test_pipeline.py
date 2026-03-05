"""Tests for the data linking pipeline."""

from datetime import date

import polars as pl
import pytest

from src.data.pipeline import (
    _expand_abbreviations,  # private but stable enough to test directly
    enrich_with_hpi,
    link_sales_to_epc,
    normalise_address,
    run_pipeline,
    save_dataset,
)


@pytest.fixture
def sales_df():
    """Sample Land Registry sales data."""
    return pl.DataFrame(
        {
            "transaction_id": ["T001", "T002", "T003"],
            "price": [250000, 450000, 180000],
            "date_of_transfer": [date(2024, 1, 15), date(2024, 2, 20), date(2024, 3, 10)],
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
    """Sample EPC data (already cleaned with snake_case columns).

    Includes two certificates for the same address (BRN001) so we can test
    that the temporally closest one is selected per sale.
    """
    return pl.DataFrame(
        {
            "postcode": ["SW1A 1AA", "SW1A 1AA", "E1 6AN", "W1A 1AB"],
            "address1": [
                "10 DOWNING STREET",
                "10 DOWNING STREET",
                "5 COMMERCIAL STREET",
                "3 OXFORD STREET",
            ],
            "building_reference_number": ["BRN001", "BRN001", "BRN002", "BRN003"],
            "current_energy_rating": ["C", "D", "B", "D"],
            "current_energy_efficiency": [70, 60, 82, 55],
            "total_floor_area": [200.0, 200.0, 65.0, 90.0],
            # Two certificates for SW1A 1AA: one close to the Jan-2024 sale, one old.
            "inspection_date": [
                date(2023, 6, 15),  # 7 months before T001 sale — should be selected
                date(2019, 1, 10),  # 5 years before — further away
                date(2023, 9, 20),
                date(2022, 1, 1),
            ],
        }
    )


@pytest.fixture
def hpi_df():
    """Sample UK HPI data (already cleaned)."""
    return pl.DataFrame(
        {
            "regionname": ["Westminster", "Tower Hamlets", "Islington", "Westminster"],
            "date": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1), date(2024, 2, 1)],
            "averageprice": [950000.0, 450000.0, 550000.0, 960000.0],
            "index": [180.3, 140.2, 155.8, 181.1],
        }
    )


class TestNormaliseAddress:
    def test_basic(self):
        paon = pl.Series(["10"])
        street = pl.Series(["Downing Street"])
        result = normalise_address(paon, street)
        assert result[0] == "10 DOWNING STREET"

    def test_handles_na(self):
        paon = pl.Series([None], dtype=pl.String)
        street = pl.Series(["HIGH STREET"])
        result = normalise_address(paon, street)
        assert result[0] == "HIGH STREET"

    def test_collapses_whitespace(self):
        paon = pl.Series(["10"])
        street = pl.Series(["DOWNING   STREET"])
        result = normalise_address(paon, street)
        assert "  " not in result[0]


class TestLinkSalesToEPC:
    def test_returns_lazy_frame(self, sales_df, epc_df):
        result = link_sales_to_epc(sales_df, epc_df)
        assert isinstance(result, pl.LazyFrame)

    def test_accepts_lazy_inputs(self, sales_df, epc_df):
        result = link_sales_to_epc(sales_df.lazy(), epc_df.lazy())
        assert isinstance(result, pl.LazyFrame)

    def test_one_row_per_sale(self, sales_df, epc_df):
        """After join, each sale must appear exactly once."""
        result = link_sales_to_epc(sales_df, epc_df).collect()
        assert len(result) == len(sales_df)

    def test_links_on_postcode(self, sales_df, epc_df):
        result = link_sales_to_epc(sales_df, epc_df).collect()
        assert "current_energy_rating" in result.columns

    def test_left_join_retains_unmatched(self, sales_df, epc_df):
        """Sales without EPC matches should be retained with null EPC columns."""
        result = link_sales_to_epc(sales_df, epc_df).collect()
        # T003 (N1 9GU) has no EPC match
        t003 = result.filter(pl.col("transaction_id") == "T003")
        assert len(t003) == 1
        assert t003["current_energy_rating"][0] is None

    def test_matched_records_have_epc(self, sales_df, epc_df):
        result = link_sales_to_epc(sales_df, epc_df).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert len(t001) == 1
        # SW1A 1AA + "10 DOWNING STREET" matches — address+postcode score wins
        assert t001["current_energy_rating"][0] == "C"

    def test_selects_temporally_closest_epc(self, sales_df, epc_df):
        """When multiple pre-sale certificates exist, pick the most recent one."""
        result = link_sales_to_epc(sales_df, epc_df).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert len(t001) == 1
        # 2023-06-15 is the most recent cert before the 2024-01-15 sale (2019-01-10 is older).
        assert t001["inspection_date"][0] == date(2023, 6, 15)

    def test_address_match_preferred_over_postcode_only(self, sales_df):
        """A postcode+address match should beat a postcode-only match."""
        # Two EPC records for the same postcode: one with matching address, one without.
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA", "SW1A 1AA"],
                "address1": ["10 DOWNING STREET", "11 DOWNING STREET"],
                "current_energy_rating": ["C", "B"],
                "inspection_date": [date(2023, 1, 1), date(2023, 6, 1)],
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        # "10 DOWNING STREET" matches paon="10", street="DOWNING STREET"
        assert t001["current_energy_rating"][0] == "C"


class TestEnrichWithHPI:
    def test_returns_lazy_frame(self, sales_df, hpi_df):
        result = enrich_with_hpi(sales_df, hpi_df)
        assert isinstance(result, pl.LazyFrame)

    def test_enriches_matching_districts(self, sales_df, hpi_df):
        result = enrich_with_hpi(sales_df, hpi_df).collect()
        assert len(result) >= len(sales_df)

    def test_returns_df_without_region_columns(self, sales_df):
        """If HPI has no region column, return data unchanged."""
        bad_hpi = pl.DataFrame({"foo": [1], "bar": [2]})
        result = enrich_with_hpi(sales_df, bad_hpi).collect()
        assert len(result) == len(sales_df)


class TestSaveDataset:
    def test_save_parquet(self, sales_df, tmp_path):
        dest = save_dataset(sales_df, "test_output", output_dir=tmp_path, fmt="parquet")
        assert dest.exists()
        assert dest.suffix == ".parquet"
        loaded = pl.read_parquet(dest)
        assert len(loaded) == len(sales_df)

    def test_save_csv(self, sales_df, tmp_path):
        dest = save_dataset(sales_df, "test_output", output_dir=tmp_path, fmt="csv")
        assert dest.exists()
        assert dest.suffix == ".csv"

    def test_save_lazy_parquet(self, sales_df, tmp_path):
        """LazyFrame input should be written via sink_parquet (streaming)."""
        dest = save_dataset(sales_df.lazy(), "test_lazy", output_dir=tmp_path, fmt="parquet")
        assert dest.exists()
        assert dest.suffix == ".parquet"
        loaded = pl.read_parquet(dest)
        assert len(loaded) == len(sales_df)

    def test_save_lazy_csv(self, sales_df, tmp_path):
        dest = save_dataset(sales_df.lazy(), "test_lazy", output_dir=tmp_path, fmt="csv")
        assert dest.exists()
        assert dest.suffix == ".csv"

    def test_save_parquet_partitioned(self, sales_df, tmp_path):
        """DataFrame with partition_by writes a hive-partitioned directory."""
        dest = save_dataset(
            sales_df, "test_part", output_dir=tmp_path, fmt="parquet", partition_by=["year"]
        )
        assert dest.is_dir()
        parquet_files = list(dest.rglob("*.parquet"))
        assert parquet_files, "Expected at least one parquet file in partitioned output"
        loaded = pl.concat([pl.read_parquet(f) for f in parquet_files])
        assert len(loaded) == len(sales_df)

    def test_save_lazy_parquet_partitioned(self, sales_df, tmp_path):
        """LazyFrame with partition_by writes a hive-partitioned directory."""
        dest = save_dataset(
            sales_df.lazy(),
            "test_part_lazy",
            output_dir=tmp_path,
            fmt="parquet",
            partition_by=["year"],
        )
        assert dest.is_dir()
        parquet_files = list(dest.rglob("*.parquet"))
        assert parquet_files, "Expected at least one parquet file in partitioned output"
        loaded = pl.concat([pl.read_parquet(f) for f in parquet_files])
        assert len(loaded) == len(sales_df)


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

    def test_accepts_lazy_inputs(self, sales_df, epc_df, tmp_path):
        """run_pipeline should accept LazyFrame inputs."""
        result = run_pipeline(
            sales=sales_df.lazy(),
            epc=epc_df.lazy(),
            output_name="test_lazy",
            output_dir=tmp_path,
            fmt="parquet",
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sales_df)

    def test_one_row_per_sale_with_multiple_epcs(self, sales_df, epc_df, tmp_path):
        """With multiple EPC records per property, pipeline still yields one row per sale."""
        result = run_pipeline(
            sales=sales_df,
            epc=epc_df,
            output_name="test_dedup",
            output_dir=tmp_path,
            fmt="parquet",
        )
        assert len(result) == len(sales_df)


class TestExpandAbbreviations:
    """Unit tests for the _expand_abbreviations address normalisation helper."""

    def _apply(self, text: str) -> str:
        """Helper: apply _expand_abbreviations to a plain string via a single-row DataFrame."""
        df = pl.DataFrame({"addr": [text]})
        return df.select(_expand_abbreviations(pl.col("addr")).alias("out"))["out"][0]

    def test_expands_rd_to_road(self):
        assert self._apply("10 VICTORIA RD") == "10 VICTORIA ROAD"

    def test_expands_ave_to_avenue(self):
        assert self._apply("5 PARK AVE") == "5 PARK AVENUE"

    def test_expands_av_to_avenue(self):
        assert self._apply("5 PARK AV") == "5 PARK AVENUE"

    def test_expands_cres_to_crescent(self):
        assert self._apply("3 MAPLE CRES") == "3 MAPLE CRESCENT"

    def test_expands_cl_to_close(self):
        assert self._apply("7 OAK CL") == "7 OAK CLOSE"

    def test_expands_gdns_to_gardens(self):
        assert self._apply("12 ROSE GDNS") == "12 ROSE GARDENS"

    def test_number_letter_hyphen_normalised(self):
        """'1-A' should become '1A'."""
        assert self._apply("1-A HIGH STREET") == "1A HIGH STREET"

    def test_number_letter_space_normalised(self):
        """'1 A' (digit space letter) should become '1A'."""
        assert self._apply("1 A HIGH STREET") == "1A HIGH STREET"

    def test_does_not_expand_st(self):
        """ST is intentionally not expanded (Saint vs Street ambiguity)."""
        result = self._apply("ST JAMES ST")
        # Should be unchanged — ST handled by fuzzy matching instead
        assert result == "ST JAMES ST"

    def test_no_change_when_no_abbrevs(self):
        assert self._apply("10 DOWNING STREET") == "10 DOWNING STREET"


class TestBackwardOnlyMatching:
    """EPC certs issued after the sale date must never be linked (data leakage guard)."""

    def test_future_cert_not_matched(self, sales_df):
        """A certificate inspected after the sale date should produce no EPC match."""
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA"],
                "address1": ["10 DOWNING STREET"],
                # Inspection date is 18 months AFTER the 2024-01-15 sale
                "inspection_date": [date(2025, 7, 1)],
                "current_energy_rating": ["A"],
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert len(t001) == 1
        assert t001["current_energy_rating"][0] is None, (
            "Future-dated EPC must not be linked to an earlier sale"
        )

    def test_cert_on_sale_date_is_matched(self, sales_df):
        """A certificate with inspection_date == sale_date is valid (boundary condition)."""
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA"],
                "address1": ["10 DOWNING STREET"],
                "inspection_date": [date(2024, 1, 15)],  # same day as T001 sale
                "current_energy_rating": ["B"],
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert t001["current_energy_rating"][0] == "B"

    def test_most_recent_pre_sale_cert_selected(self, sales_df):
        """With one pre-sale and one post-sale cert, only the pre-sale one is considered."""
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA", "SW1A 1AA"],
                "address1": ["10 DOWNING STREET", "10 DOWNING STREET"],
                "inspection_date": [date(2022, 3, 1), date(2025, 1, 1)],
                "current_energy_rating": ["C", "A"],  # "C" is pre-sale, "A" is post-sale
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        # "A" is from the future; "C" is the only valid match
        assert t001["current_energy_rating"][0] == "C"

    def test_no_cert_at_all_leaves_null(self, sales_df):
        """If there are no EPC records at the postcode, EPC columns remain null."""
        epc = pl.DataFrame(
            {
                "postcode": ["W1A 1AB"],  # Different postcode — no match
                "address1": ["1 OXFORD STREET"],
                "inspection_date": [date(2022, 1, 1)],
                "current_energy_rating": ["D"],
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert t001["current_energy_rating"][0] is None


class TestAddressMatchingBehaviour:
    """Behaviour tests for the exact-address matching contract.

    Fuzzy matching is not yet implemented.  These tests document both what
    the pipeline DOES match (after abbreviation expansion) and what it
    deliberately leaves unmatched until a fuzzy fallback is added.
    """

    def test_abbreviation_expansion_enables_exact_match(self):
        """RD → ROAD expansion on both sides should allow a Phase-1 exact match."""
        # Sales: paon="10", street="DOWNING ROAD" → _addr_key = "10 DOWNING ROAD"
        # EPC:  address1="10 DOWNING RD"          → after expansion: "10 DOWNING ROAD"
        # Both resolve to the same key → exact match fires.
        sales_rd = pl.DataFrame(
            {
                "transaction_id": ["T001"],
                "price": [250000],
                "date_of_transfer": [date(2024, 1, 15)],
                "postcode": ["SW1A 1AA"],
                "property_type": ["D"],
                "paon": ["10"],
                "street": ["DOWNING ROAD"],
                "district": ["Westminster"],
                "year": [2024],
                "month": [1],
            }
        )
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA"],
                "address1": ["10 DOWNING RD"],
                "lmk_key": ["LMK002"],
                "current_energy_rating": ["C"],
                "inspection_date": [date(2023, 3, 1)],
            }
        )
        result = link_sales_to_epc(sales_rd, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert t001["current_energy_rating"][0] == "C"

    def test_st_vs_street_does_not_match(self, sales_df):
        """ST is not in the expansion list (Saint/Street ambiguity).

        'DOWNING ST' and 'DOWNING STREET' produce different _addr_keys after
        normalisation, so no match is made.  A fuzzy fallback (planned) will
        eventually handle this case.
        """
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA"],
                "address1": ["10 DOWNING ST"],
                "lmk_key": ["LMK001"],
                "current_energy_rating": ["B"],
                "inspection_date": [date(2023, 6, 1)],
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        # "10 DOWNING STREET" ≠ "10 DOWNING ST" after normalisation → no match
        assert t001["current_energy_rating"][0] is None

    def test_different_address_gives_null_epc(self, sales_df):
        """A sale with no matching address in its postcode retains null EPC columns."""
        epc = pl.DataFrame(
            {
                "postcode": ["SW1A 1AA"],
                "address1": ["11 DOWNING STREET"],  # different PAON
                "lmk_key": ["LMK001"],
                "current_energy_rating": ["D"],
                "inspection_date": [date(2023, 1, 1)],
            }
        )
        result = link_sales_to_epc(sales_df, epc).collect()
        t001 = result.filter(pl.col("transaction_id") == "T001")
        assert t001["current_energy_rating"][0] is None
