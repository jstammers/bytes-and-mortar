"""Tests for the ingest manifest helpers."""

from datetime import UTC, date, datetime

import pytest

from src.data import manifest


@pytest.fixture
def tmp_raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_read_manifest_missing_returns_empty(tmp_raw_dir):
    assert manifest.read_manifest(tmp_raw_dir) == {}


def test_write_manifest_persists_entry(tmp_raw_dir):
    entry = manifest.write_manifest(
        "land_registry",
        raw_dir=tmp_raw_dir,
        last_data_through=date(2026, 4, 30),
        source_url="https://example.com/pp.csv",
        row_count=123,
    )
    assert entry["last_data_through"] == "2026-04-30"
    assert entry["source_url"] == "https://example.com/pp.csv"
    assert entry["row_count"] == 123
    assert entry["last_run_utc"].endswith("Z")

    read_back = manifest.read_manifest(tmp_raw_dir)
    assert read_back == {"land_registry": entry}


def test_write_manifest_preserves_other_sources(tmp_raw_dir):
    manifest.write_manifest("a", raw_dir=tmp_raw_dir, row_count=1)
    manifest.write_manifest("b", raw_dir=tmp_raw_dir, row_count=2)
    data = manifest.read_manifest(tmp_raw_dir)
    assert set(data.keys()) == {"a", "b"}
    assert data["a"]["row_count"] == 1
    assert data["b"]["row_count"] == 2


def test_write_manifest_updates_run_timestamp(tmp_raw_dir):
    entry1 = manifest.write_manifest("epc", raw_dir=tmp_raw_dir, row_count=10)
    entry2 = manifest.write_manifest("epc", raw_dir=tmp_raw_dir, row_count=20)
    assert entry2["row_count"] == 20
    assert entry2["last_run_utc"] >= entry1["last_run_utc"]


def test_write_manifest_preserves_unspecified_fields(tmp_raw_dir):
    manifest.write_manifest(
        "hpi",
        raw_dir=tmp_raw_dir,
        last_data_through=date(2026, 3, 31),
        source_url="https://example.com/hpi.csv",
        row_count=999,
    )
    manifest.write_manifest("hpi", raw_dir=tmp_raw_dir, row_count=1000)
    entry = manifest.read_manifest(tmp_raw_dir)["hpi"]
    assert entry["row_count"] == 1000
    assert entry["last_data_through"] == "2026-03-31"
    assert entry["source_url"] == "https://example.com/hpi.csv"


def test_get_last_through(tmp_raw_dir):
    assert manifest.get_last_through("epc", tmp_raw_dir) is None
    manifest.write_manifest("epc", raw_dir=tmp_raw_dir, last_data_through=date(2026, 1, 15))
    assert manifest.get_last_through("epc", tmp_raw_dir) == date(2026, 1, 15)


def test_coerce_date_accepts_datetime_and_string(tmp_raw_dir):
    manifest.write_manifest(
        "lr",
        raw_dir=tmp_raw_dir,
        last_data_through=datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
    )
    assert manifest.read_manifest(tmp_raw_dir)["lr"]["last_data_through"] == "2026-02-01"

    manifest.write_manifest("lr", raw_dir=tmp_raw_dir, last_data_through="2026-03-15")
    assert manifest.read_manifest(tmp_raw_dir)["lr"]["last_data_through"] == "2026-03-15"


def test_malformed_manifest_treated_as_empty(tmp_raw_dir):
    manifest.manifest_path(tmp_raw_dir).write_text("{not json")
    assert manifest.read_manifest(tmp_raw_dir) == {}
    manifest.write_manifest("x", raw_dir=tmp_raw_dir, row_count=1)
    assert manifest.read_manifest(tmp_raw_dir)["x"]["row_count"] == 1
