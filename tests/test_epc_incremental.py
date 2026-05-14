"""Tests for EPC incremental ingestion via the search API."""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data import manifest
from src.data.sources.epc import EPCData


@pytest.fixture
def tmp_raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def source(tmp_raw_dir):
    return EPCData(raw_dir=tmp_raw_dir, api_user="test-user", api_pass="test-pass")


def _make_bulk_parquet(raw_dir):
    df = pl.DataFrame(
        {
            "lmk-key": ["LMK-001", "LMK-002", "LMK-003"],
            "postcode": ["SW1A 1AA", "E1 6AN", "N1 9GU"],
            "current-energy-rating": ["C", "B", "E"],
            "inspection-date": [
                date(2024, 6, 15),
                date(2025, 1, 10),
                date(2025, 3, 1),
            ],
            "total-floor-area": [200.0, 65.0, 120.0],
        }
    )
    df.write_parquet(raw_dir / "epc_domestic_bulk.parquet")
    return df


def _mock_search_response(text: str):
    response = MagicMock()
    response.text = text
    response.headers = {}
    response.raise_for_status = MagicMock()
    return response


def test_incremental_requires_existing_bulk(source, tmp_raw_dir):
    """Missing bulk parquet → informative error."""
    manifest.write_manifest(source.name, raw_dir=tmp_raw_dir, last_data_through=date(2025, 3, 1))
    with pytest.raises(FileNotFoundError, match="cached bulk"):
        source.download(incremental=True)


def test_incremental_requires_manifest(source, tmp_raw_dir):
    """No manifest entry → cannot derive since-date."""
    _make_bulk_parquet(tmp_raw_dir)
    with pytest.raises(RuntimeError, match="last_data_through"):
        source.download(incremental=True)


@patch("src.data.sources.epc.requests.get")
@patch("src.data.sources.epc.time.sleep", lambda *_: None)
def test_fetch_delta_pages_with_since_date(mock_get, source, tmp_raw_dir):
    _make_bulk_parquet(tmp_raw_dir)
    manifest.write_manifest(source.name, raw_dir=tmp_raw_dir, last_data_through=date(2026, 1, 1))

    page1 = _mock_search_response(
        "lmk-key,postcode,current-energy-rating,inspection-date,total-floor-area\n"
        "LMK-100,SW2 1AB,C,2026-02-15,85.0\n"
        "LMK-101,E2 0AB,B,2026-02-20,72.0"
    )
    page2 = _mock_search_response(
        "lmk-key,postcode,current-energy-rating,inspection-date,total-floor-area"
    )
    mock_get.side_effect = [page1, page2]

    delta_path = source._fetch_delta_since(date(2026, 1, 1), max_pages=10)
    assert delta_path.exists()
    delta_df = pl.read_parquet(delta_path)
    assert set(delta_df["lmk-key"].to_list()) == {"LMK-100", "LMK-101"}

    first_call_params = mock_get.call_args_list[0].kwargs["params"]
    assert first_call_params["from-year"] == 2026
    assert first_call_params["from-month"] == 1


def test_merge_into_bulk_dedups_on_lmk_key(source, tmp_raw_dir):
    """When the same lmk-key appears in delta and bulk, keep the newest inspection."""
    _make_bulk_parquet(tmp_raw_dir)
    bulk_path = tmp_raw_dir / "epc_domestic_bulk.parquet"

    delta_df = pl.DataFrame(
        {
            "lmk-key": ["LMK-002", "LMK-200"],
            "postcode": ["E1 6AN", "SW2 1AB"],
            "current-energy-rating": ["A", "C"],
            "inspection-date": [date(2026, 4, 1), date(2026, 4, 5)],
            "total-floor-area": [70.0, 90.0],
        }
    )
    delta_path = tmp_raw_dir / "epc_domestic_delta_202604.parquet"
    delta_df.write_parquet(delta_path)

    source._merge_into_bulk(delta_path, bulk_path)

    merged = pl.read_parquet(bulk_path)
    assert set(merged["lmk-key"].to_list()) == {
        "LMK-001",
        "LMK-002",
        "LMK-003",
        "LMK-200",
    }
    lmk2 = merged.filter(pl.col("lmk-key") == "LMK-002")
    assert lmk2["current-energy-rating"].item() == "A"
    assert lmk2["inspection-date"].item() == date(2026, 4, 1)


def test_merge_handles_empty_delta(source, tmp_raw_dir):
    _make_bulk_parquet(tmp_raw_dir)
    bulk_path = tmp_raw_dir / "epc_domestic_bulk.parquet"
    delta_path = tmp_raw_dir / "epc_domestic_delta_202604.parquet"
    pl.DataFrame().write_parquet(delta_path)

    before = pl.read_parquet(bulk_path)
    source._merge_into_bulk(delta_path, bulk_path)
    after = pl.read_parquet(bulk_path)
    assert before.equals(after)


@patch("src.data.sources.epc.requests.get")
@patch("src.data.sources.epc.time.sleep", lambda *_: None)
def test_ingest_incremental_updates_manifest(mock_get, source, tmp_raw_dir):
    _make_bulk_parquet(tmp_raw_dir)
    manifest.write_manifest(source.name, raw_dir=tmp_raw_dir, last_data_through=date(2025, 3, 1))

    page1 = _mock_search_response(
        "lmk-key,postcode,current-energy-rating,inspection-date,total-floor-area\n"
        "LMK-300,SW3 1AA,D,2026-05-01,150.0"
    )
    page2 = _mock_search_response(
        "lmk-key,postcode,current-energy-rating,inspection-date,total-floor-area"
    )
    mock_get.side_effect = [page1, page2]

    cleaned_lf = source.ingest(incremental=True)
    cleaned = cleaned_lf.collect()
    assert "LMK-300" in cleaned["lmk_key"].to_list()

    entry = manifest.read_manifest(tmp_raw_dir)[source.name]
    assert entry["last_data_through"] == "2026-05-01"
    assert entry["row_count"] == 4
    assert entry["source_url"].endswith("/domestic/search")
