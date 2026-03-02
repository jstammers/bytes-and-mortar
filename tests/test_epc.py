"""Tests for EPC data source."""

import polars as pl
import pytest

from src.data.sources.epc import EPCData


@pytest.fixture
def tmp_raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def source(tmp_raw_dir):
    return EPCData(raw_dir=tmp_raw_dir, api_token="test-token")


@pytest.fixture
def sample_epc_csv(tmp_raw_dir):
    """Create a sample EPC CSV file."""
    tmp_raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = tmp_raw_dir / "epc_domestic.csv"
    header = "address1,address2,address3,postcode,building-reference-number,current-energy-rating,current-energy-efficiency,potential-energy-rating,potential-energy-efficiency,property-type,built-form,inspection-date,total-floor-area,number-habitable-rooms,tenure"
    rows = [
        "10 DOWNING STREET,,,SW1A 1AA,BRN001,C,70,B,82,House,Detached,2023-06-15,200,8,owner-occupied",
        "10 DOWNING STREET,,,SW1A 1AA,BRN001,D,60,C,72,House,Detached,2020-01-10,200,8,owner-occupied",
        "5 COMMERCIAL STREET,FLAT 2,,E1 6AN,BRN002,B,82,A,95,Flat,Purpose-built,2023-09-20,65,3,rental (private)",
        "12 ISLINGTON PARK ST,,,N1 9GU,BRN003,E,45,C,70,House,Terraced,2022-04-01,120,5,owner-occupied",
        "INVALID RECORD,,,,,X,999,,,,,,,,",
    ]
    filepath.write_text("\n".join([header, *rows]))
    return filepath


def test_load_epc_data(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv)
    assert len(df) == 5
    assert "address1" in df.columns
    assert "current-energy-rating" in df.columns


def test_clean_standardises_column_names(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv)
    cleaned = source.clean(df)
    # Hyphens should become underscores
    assert "current_energy_rating" in cleaned.columns
    assert "building_reference_number" in cleaned.columns


def test_clean_standardises_postcodes(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv)
    cleaned = source.clean(df)
    for pc in cleaned["postcode"]:
        assert pc == pc.upper()


def test_clean_filters_invalid_ratings(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv)
    cleaned = source.clean(df)
    valid_ratings = {"A", "B", "C", "D", "E", "F", "G"}
    assert cleaned["current_energy_rating"].is_in(valid_ratings).all()


def test_clean_deduplicates_by_building_ref(source, sample_epc_csv):
    """Should keep only the most recent EPC per building reference."""
    df = source.load(filepath=sample_epc_csv)
    cleaned = source.clean(df)
    # BRN001 appears twice; only the newer one (2023-06-15) should remain
    brn001 = cleaned.filter(pl.col("building_reference_number") == "BRN001")
    assert len(brn001) == 1
    assert brn001["inspection_date"][0].year == 2023


def test_clean_drops_missing_postcodes(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv)
    cleaned = source.clean(df)
    assert cleaned["postcode"].is_not_null().all()


def test_clean_numeric_conversions(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv)
    cleaned = source.clean(df)
    assert cleaned["current_energy_efficiency"].dtype.is_numeric()
    assert cleaned["total_floor_area"].dtype.is_numeric()


def test_no_api_token_warning(tmp_raw_dir, monkeypatch):
    """Should warn when no API token is set."""
    monkeypatch.delenv("EPC_API_TOKEN", raising=False)
    EPCData(raw_dir=tmp_raw_dir, api_token="")  # Should not raise, just warn


def test_download_requires_api_token(tmp_raw_dir, monkeypatch):
    """Download should raise ValueError with no API token."""
    monkeypatch.delenv("EPC_API_TOKEN", raising=False)
    source = EPCData(raw_dir=tmp_raw_dir, api_token="")
    with pytest.raises(ValueError, match="EPC_API_TOKEN"):
        source.download()
