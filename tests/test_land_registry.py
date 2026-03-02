"""Tests for Land Registry Price Paid data source."""

from pathlib import Path

import pytest

from src.data.sources.land_registry import LandRegistryPricePaid


@pytest.fixture
def tmp_raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def source(tmp_raw_dir):
    return LandRegistryPricePaid(raw_dir=tmp_raw_dir)


@pytest.fixture
def sample_csv(tmp_raw_dir):
    """Create a sample Land Registry CSV file with braces around values."""
    tmp_raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = tmp_raw_dir / "pp-2024.csv"
    rows = [
        "{ABC-123-DEF},100000,2024-01-15 00:00,{SW1A 1AA},{D},{N},{F},{10},{},{DOWNING STREET},{},{LONDON},{WESTMINSTER},{GREATER LONDON},{A},{A}",
        "{GHI-456-JKL},250000,2024-02-20 00:00,{E1 6AN},{S},{Y},{L},{5},{FLAT 2},{COMMERCIAL STREET},{SPITALFIELDS},{LONDON},{TOWER HAMLETS},{GREATER LONDON},{A},{A}",
        "{MNO-789-PQR},50000,2024-03-10 00:00,{SW1A 2AA},{T},{N},{F},{1},{},{WHITEHALL},{},{LONDON},{WESTMINSTER},{GREATER LONDON},{A},{A}",
        "{STU-012-VWX},5000,2024-04-01 00:00,{W1A 1AB},{F},{N},{L},{3},{},{OXFORD STREET},{},{LONDON},{WESTMINSTER},{GREATER LONDON},{A},{A}",
        "{YZA-345-BCD},200000,2024-05-15 00:00,{},{D},{N},{F},{7},{},{HIGH STREET},{},{LONDON},{CAMDEN},{GREATER LONDON},{A},{A}",
        "{DEL-678-ETE},300000,2024-06-01 00:00,{N1 9GU},{O},{N},{F},{12},{},{ISLINGTON PARK ST},{},{LONDON},{ISLINGTON},{GREATER LONDON},{A},{D}",
    ]
    filepath.write_text("\n".join(rows))
    return filepath


def test_load_parses_columns(source, sample_csv):
    df = source.load(filepath=sample_csv)
    assert "transaction_id" in df.columns
    assert "price" in df.columns
    assert "postcode" in df.columns
    assert "property_type" in df.columns
    assert len(df) == 6


def test_load_strips_braces(source, sample_csv):
    df = source.load(filepath=sample_csv)
    assert df["transaction_id"].iloc[0] == "ABC-123-DEF"
    assert df["postcode"].iloc[0] == "SW1A 1AA"


def test_load_with_nrows(source, sample_csv):
    df = source.load(filepath=sample_csv, nrows=2)
    assert len(df) == 2


def test_load_file_not_found(source):
    with pytest.raises(FileNotFoundError):
        source.load(filepath=Path("/nonexistent/file.csv"))


def test_clean_removes_deletions(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    # Row with record_status "D" should be removed
    assert "DEL-678-ETE" not in cleaned["transaction_id"].values


def test_clean_removes_missing_postcodes(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    assert cleaned["postcode"].notna().all()


def test_clean_removes_non_residential(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    # Property type "O" should be removed
    assert "O" not in cleaned["property_type"].values


def test_clean_filters_extreme_prices(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    # Price of 5000 should be filtered out (< 10,000 threshold)
    assert (cleaned["price"] >= 10_000).all()
    assert (cleaned["price"] < 50_000_000).all()


def test_clean_standardises_postcodes(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    for pc in cleaned["postcode"]:
        assert pc == pc.upper()
        assert "  " not in pc  # no double spaces


def test_clean_adds_year_month(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    assert "year" in cleaned.columns
    assert "month" in cleaned.columns
    assert cleaned["year"].iloc[0] == 2024


def test_clean_adds_outward_postcode(source, sample_csv):
    df = source.load(filepath=sample_csv)
    cleaned = source.clean(df)
    assert "postcode_outward" in cleaned.columns
    # SW1A 1AA -> SW1A
    assert "SW1A" in cleaned["postcode_outward"].values


def test_download_url_for_year(source):
    """Test that the download method constructs the correct URL for a year."""
    # We can't actually download, but we can verify the URL logic
    from src.data.config import LAND_REGISTRY_YEARLY_CSV

    url = LAND_REGISTRY_YEARLY_CSV.format(year=2024)
    assert "pp-2024.csv" in url


def test_download_url_for_complete(source):
    from src.data.config import LAND_REGISTRY_COMPLETE_CSV

    assert "pp-complete.csv" in LAND_REGISTRY_COMPLETE_CSV
