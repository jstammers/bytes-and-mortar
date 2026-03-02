"""Tests for UK House Price Index data source."""

from pathlib import Path

import polars as pl
import pytest

from src.data.sources.uk_hpi import UKHousePriceIndex


@pytest.fixture
def tmp_raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def source(tmp_raw_dir):
    return UKHousePriceIndex(raw_dir=tmp_raw_dir)


@pytest.fixture
def sample_hpi_csv(tmp_raw_dir):
    """Create a sample UK HPI CSV file."""
    tmp_raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = tmp_raw_dir / "uk_hpi_full.csv"
    header = "Date,RegionName,AreaCode,AveragePrice,Index,1m%Change,12m%Change,SalesVolume"
    rows = [
        "2024-01-01,London,E12000007,520000,150.5,0.5,3.2,8500",
        "2024-02-01,London,E12000007,525000,151.2,0.3,3.5,8200",
        "2024-01-01,Westminster,E09000033,950000,180.3,0.8,4.1,350",
        "2024-02-01,Westminster,E09000033,960000,181.1,0.4,4.3,320",
        "2024-01-01,Tower Hamlets,E09000030,450000,140.2,0.2,2.8,600",
        ",,,,,,,",
    ]
    filepath.write_text("\n".join([header, *rows]))
    return filepath


def test_load_hpi(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    assert len(df) == 6
    assert "RegionName" in df.columns


def test_load_file_not_found(source):
    with pytest.raises(FileNotFoundError):
        source.load(filepath=Path("/nonexistent/file.csv"))


def test_clean_standardises_column_names(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    cleaned = source.clean(df)
    assert "regionname" in cleaned.columns
    assert "averageprice" in cleaned.columns


def test_clean_parses_dates(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    cleaned = source.clean(df)
    assert "year" in cleaned.columns
    assert "month" in cleaned.columns
    assert cleaned["date"].dtype == pl.Date


def test_clean_drops_empty_rows(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    cleaned = source.clean(df)
    # The empty row should be dropped
    assert len(cleaned) == 5


def test_clean_numeric_columns(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    cleaned = source.clean(df)
    assert cleaned["averageprice"].dtype.is_numeric()
    assert cleaned["index"].dtype.is_numeric()


def test_get_area_prices(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    cleaned = source.clean(df)
    london = source.get_area_prices(cleaned, "London")
    assert len(london) == 2
    assert (london["regionname"] == "London").all()


def test_get_area_prices_case_insensitive(source, sample_hpi_csv):
    df = source.load(filepath=sample_hpi_csv)
    cleaned = source.clean(df)
    london = source.get_area_prices(cleaned, "london")
    assert len(london) == 2
