"""Tests for Land Registry monthly-delta incremental ingestion."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data import manifest
from src.data.sources.land_registry import LandRegistryPricePaid


@pytest.fixture
def tmp_raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def source(tmp_raw_dir):
    return LandRegistryPricePaid(raw_dir=tmp_raw_dir)


def _make_complete_parquet(raw_dir):
    """Write a synthetic cached bulk parquet with 4 known transactions."""
    df = pl.DataFrame(
        {
            "transaction_id": ["T-001", "T-002", "T-003", "T-004"],
            "price": [100_000, 250_000, 175_000, 320_000],
            "date_of_transfer": [
                "2024-01-15 00:00",
                "2024-02-20 00:00",
                "2024-03-10 00:00",
                "2024-04-05 00:00",
            ],
            "postcode": ["SW1A 1AA", "E1 6AN", "N1 9GU", "W1A 1AB"],
            "property_type": ["D", "S", "T", "F"],
            "old_new": ["N", "Y", "N", "N"],
            "duration": ["F", "L", "F", "L"],
            "paon": ["10", "5", "12", "3"],
            "saon": [None, "FLAT 2", None, None],
            "street": ["DOWNING ST", "COMMERCIAL", "ISLINGTON", "OXFORD"],
            "locality": [None, "SPITALFIELDS", None, None],
            "town_city": ["LONDON", "LONDON", "LONDON", "LONDON"],
            "district": ["WESTMINSTER", "TOWER HAMLETS", "ISLINGTON", "WESTMINSTER"],
            "county": ["GREATER LONDON"] * 4,
            "ppd_category": ["A"] * 4,
            "record_status": ["A"] * 4,
        }
    ).with_columns(
        pl.col("price").cast(pl.Int64),
        pl.col("date_of_transfer").str.to_datetime(format="%Y-%m-%d %H:%M"),
    )
    df.write_parquet(raw_dir / "pp-complete.parquet")
    return df


def _write_monthly_csv(raw_dir, rows):
    """Write a monthly-delta CSV in raw Land Registry brace-wrapped format."""
    path = raw_dir / "pp-monthly-update-new-version.csv"
    path.write_text("\n".join(rows))
    return path


def test_merge_monthly_addition(source, tmp_raw_dir):
    _make_complete_parquet(tmp_raw_dir)
    monthly_path = _write_monthly_csv(
        tmp_raw_dir,
        [
            "{T-005},420000,2024-05-01 00:00,{SW1A 2AA},{D},{N},{F},{20},{},{ST},{},{LON},{WESTMINSTER},{GLON},{A},{A}",
        ],
    )

    merged = source.merge_monthly(source.load(filepath=monthly_path)).collect()
    ids = set(merged["transaction_id"].to_list())
    assert ids == {"T-001", "T-002", "T-003", "T-004", "T-005"}
    new_row = merged.filter(pl.col("transaction_id") == "T-005")
    assert new_row["price"].item() == 420_000


def test_merge_monthly_change(source, tmp_raw_dir):
    _make_complete_parquet(tmp_raw_dir)
    monthly_path = _write_monthly_csv(
        tmp_raw_dir,
        [
            "{T-002},275000,2024-02-20 00:00,{E1 6AN},{S},{Y},{L},{5},{FLAT 2},{COMMERCIAL},{SPITALFIELDS},{LON},{TOWER HAMLETS},{GLON},{A},{C}",
        ],
    )

    merged = source.merge_monthly(source.load(filepath=monthly_path)).collect()
    rows = merged.filter(pl.col("transaction_id") == "T-002")
    assert len(rows) == 1
    assert rows["price"].item() == 275_000
    assert set(merged["transaction_id"].to_list()) == {
        "T-001",
        "T-002",
        "T-003",
        "T-004",
    }


def test_merge_monthly_deletion(source, tmp_raw_dir):
    _make_complete_parquet(tmp_raw_dir)
    monthly_path = _write_monthly_csv(
        tmp_raw_dir,
        [
            "{T-003},175000,2024-03-10 00:00,{N1 9GU},{T},{N},{F},{12},{},{ISLINGTON},{},{LON},{ISLINGTON},{GLON},{A},{D}",
        ],
    )

    merged = source.merge_monthly(source.load(filepath=monthly_path)).collect()
    assert "T-003" not in merged["transaction_id"].to_list()
    assert len(merged) == 3


def test_merge_monthly_mixed_acd(source, tmp_raw_dir):
    _make_complete_parquet(tmp_raw_dir)
    monthly_path = _write_monthly_csv(
        tmp_raw_dir,
        [
            "{T-005},420000,2024-05-01 00:00,{SW1A 2AA},{D},{N},{F},{20},{},{ST},{},{LON},{WESTMINSTER},{GLON},{A},{A}",
            "{T-002},275000,2024-02-20 00:00,{E1 6AN},{S},{Y},{L},{5},{FLAT 2},{COMMERCIAL},{SPITALFIELDS},{LON},{TOWER HAMLETS},{GLON},{A},{C}",
            "{T-003},175000,2024-03-10 00:00,{N1 9GU},{T},{N},{F},{12},{},{ISLINGTON},{},{LON},{ISLINGTON},{GLON},{A},{D}",
        ],
    )

    merged = source.merge_monthly(source.load(filepath=monthly_path)).collect()
    ids = set(merged["transaction_id"].to_list())
    assert ids == {"T-001", "T-002", "T-004", "T-005"}
    assert merged.filter(pl.col("transaction_id") == "T-002")["price"].item() == 275_000


def test_ingest_incremental_writes_parquet_and_manifest(source, tmp_raw_dir):
    _make_complete_parquet(tmp_raw_dir)
    monthly_path = _write_monthly_csv(
        tmp_raw_dir,
        [
            "{T-005},420000,2026-04-15 00:00,{SW1A 2AA},{D},{N},{F},{20},{},{ST},{},{LON},{WESTMINSTER},{GLON},{A},{A}",
        ],
    )

    with patch.object(LandRegistryPricePaid, "download_monthly", return_value=monthly_path):
        cleaned_lf = source.ingest(incremental=True)

    cleaned = cleaned_lf.collect()
    assert "T-005" in cleaned["transaction_id"].to_list()

    refreshed = pl.scan_parquet(tmp_raw_dir / "pp-complete.parquet").collect()
    assert len(refreshed) == 5

    entry = manifest.read_manifest(tmp_raw_dir)["land_registry_price_paid"]
    assert entry["row_count"] == 5
    assert entry["last_data_through"] == "2026-04-15"
    assert entry["source_url"].endswith("pp-monthly-update-new-version.csv")


def test_ingest_non_incremental_unchanged(source, tmp_raw_dir):
    """Non-incremental ingest still works through the base-class path."""
    _make_complete_parquet(tmp_raw_dir)

    # download() asks for pp-complete.csv; the parquet sibling is what load() picks up.
    # Patch the network call so this test never reaches the upstream URL.
    def fake_download(self, url, dest, desc=""):
        return dest

    with patch.object(LandRegistryPricePaid, "_download_file", fake_download):
        lf = source.ingest()

    df = lf.collect()
    assert len(df) == 4


def test_download_monthly_force_refetch(source, tmp_raw_dir):
    """download_monthly() must always re-fetch even when the file exists."""
    stale = tmp_raw_dir / "pp-monthly-update-new-version.csv"
    stale.write_text("stale-content")

    fake = MagicMock(side_effect=lambda url, dest, desc="": dest)
    with patch.object(LandRegistryPricePaid, "_download_file", fake):
        source.download_monthly()

    assert fake.called
    call_args = fake.call_args
    dest = call_args[0][1]
    assert not dest.exists() or dest.read_text() != "stale-content"
