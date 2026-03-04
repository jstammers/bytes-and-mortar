"""Tests for EPC data source."""

from contextlib import suppress
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data.sources.epc import EPCData


@pytest.fixture
def tmp_raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def source(tmp_raw_dir):
    return EPCData(raw_dir=tmp_raw_dir, api_user="test-user", api_pass="test-pass")


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


def test_load_returns_lazy_frame(source, sample_epc_csv):
    """load() must return a LazyFrame — no data read at scan time."""
    result = source.load(filepath=sample_epc_csv)
    assert isinstance(result, pl.LazyFrame)


def test_load_epc_data(source, sample_epc_csv):
    df = source.load(filepath=sample_epc_csv).collect()
    assert len(df) == 5
    assert "address1" in df.columns
    assert "current-energy-rating" in df.columns


def test_clean_returns_lazy_frame(source, sample_epc_csv):
    """clean() must accept and return a LazyFrame."""
    lf = source.load(filepath=sample_epc_csv)
    result = source.clean(lf)
    assert isinstance(result, pl.LazyFrame)


def test_clean_standardises_column_names(source, sample_epc_csv):
    df = source.clean(source.load(filepath=sample_epc_csv)).collect()
    # Hyphens should become underscores
    assert "current_energy_rating" in df.columns
    assert "building_reference_number" in df.columns


def test_clean_standardises_postcodes(source, sample_epc_csv):
    df = source.clean(source.load(filepath=sample_epc_csv)).collect()
    for pc in df["postcode"]:
        assert pc == pc.upper()


def test_clean_filters_invalid_ratings(source, sample_epc_csv):
    df = source.clean(source.load(filepath=sample_epc_csv)).collect()
    valid_ratings = {"A", "B", "C", "D", "E", "F", "G"}
    assert df["current_energy_rating"].is_in(valid_ratings).all()


def test_clean_retains_all_certificates(source, sample_epc_csv):
    """All valid EPC records should be kept — no dedup by building reference.

    The pipeline selects the temporally closest certificate per sale,
    so the full history must be available.
    """
    df = source.clean(source.load(filepath=sample_epc_csv)).collect()
    # BRN001 appears twice with different inspection dates; both valid records retained.
    brn001 = df.filter(pl.col("building_reference_number") == "BRN001")
    assert len(brn001) == 2


def test_clean_drops_missing_postcodes(source, sample_epc_csv):
    df = source.clean(source.load(filepath=sample_epc_csv)).collect()
    assert df["postcode"].is_not_null().all()


def test_clean_numeric_conversions(source, sample_epc_csv):
    df = source.clean(source.load(filepath=sample_epc_csv)).collect()
    assert df["current_energy_efficiency"].dtype.is_numeric()
    assert df["total_floor_area"].dtype.is_numeric()


def test_no_api_token_warning(tmp_raw_dir, monkeypatch):
    """Should warn when no API token is set."""
    monkeypatch.delenv("EPC_API_TOKEN", raising=False)
    monkeypatch.delenv("EPC_API_USER", raising=False)
    monkeypatch.delenv("EPC_API_PASS", raising=False)
    EPCData(raw_dir=tmp_raw_dir)  # Should not raise, just warn


def test_download_requires_api_token(tmp_raw_dir, monkeypatch):
    """Download should raise ValueError with no API token."""
    monkeypatch.delenv("EPC_API_USER", raising=False)
    monkeypatch.delenv("EPC_API_PASS", raising=False)
    source = EPCData(raw_dir=tmp_raw_dir)
    with pytest.raises(ValueError, match="EPC_API_USER and EPC_API_PASS"):
        source.download()


@patch("src.data.sources.epc.requests.get")
def test_download_bulk_file_default(mock_get, source, tmp_raw_dir):
    """Default download() should use bulk file endpoint."""
    # Mock the download response
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "100"}
    # This would be a zip file in reality
    mock_response.iter_content = lambda chunk_size: []
    mock_get.return_value = mock_response

    # We expect it to fail since we're not providing actual zip data,
    # but we can verify the correct URL was called
    with suppress(Exception):
        source.download()

    # Verify the files API endpoint was called (bulk download)
    calls = mock_get.call_args_list
    assert any("api/v1/files" in str(call) for call in calls), (
        "Should call the bulk files endpoint by default"
    )


@patch("src.data.sources.epc.requests.get")
def test_download_search_when_requested(mock_get, source, tmp_raw_dir):
    """download(use_search=True) should use the search endpoint."""
    # Mock the search API response with multiple pages
    # First call: returns header + data rows
    mock_response_page1 = MagicMock()
    mock_response_page1.text = (
        "postcode,building-reference-number,current-energy-rating\n"
        "SW1A 1AA,BRN001,C\n"
        "E1 6AN,BRN002,B"
    )
    mock_response_page1.headers = {}

    # Second call: returns empty response to signal end of data
    mock_response_page2 = MagicMock()
    mock_response_page2.text = "postcode,building-reference-number,current-energy-rating"
    mock_response_page2.headers = {}

    mock_get.side_effect = [mock_response_page1, mock_response_page2]

    result = source.download(use_search=True, local_authority="E09000033")

    # Verify the search endpoint was called
    calls = mock_get.call_args_list
    assert any("domestic/search" in str(call) for call in calls), (
        "Should call the search endpoint when use_search=True"
    )
    assert result.name == "epc_domestic_E09000033.parquet"


def test_list_available_files_requires_token(tmp_raw_dir, monkeypatch):
    """_list_available_files() should raise without API token."""
    monkeypatch.delenv("EPC_API_USER", raising=False)
    monkeypatch.delenv("EPC_API_PASS", raising=False)
    source = EPCData(raw_dir=tmp_raw_dir)
    with pytest.raises(ValueError, match="EPC API token not set"):
        source._list_available_files()


@patch("src.data.sources.epc.requests.get")
def test_list_available_files(mock_get, source):
    """_list_available_files() should return available bulk files."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "files": {
            "all-domestic-certificates.zip": {"size": 6199109043},
            "domestic-2023.zip": {"size": 500000000},
        }
    }
    mock_get.return_value = mock_response

    files = source._list_available_files()

    assert "all-domestic-certificates.zip" in files
    assert "domestic-2023.zip" in files
    assert files["all-domestic-certificates.zip"]["size"] == 6199109043
