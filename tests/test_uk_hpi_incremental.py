"""Tests for UK HPI incremental refresh."""

from unittest.mock import patch

import polars as pl
import pytest

from src.data import manifest
from src.data.sources.uk_hpi import UKHousePriceIndex


@pytest.fixture
def tmp_raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def source(tmp_raw_dir):
    return UKHousePriceIndex(raw_dir=tmp_raw_dir)


def _write_hpi_csv(path, rows):
    header = "Date,RegionName,AreaCode,AveragePrice,Index"
    path.write_text("\n".join([header, *rows]))


def test_download_incremental_removes_cached_files(source, tmp_raw_dir):
    """When incremental=True, the cached csv and parquet must be wiped before fetch."""
    stale_csv = tmp_raw_dir / "uk_hpi_full.csv"
    stale_csv.write_text("stale")
    stale_parquet = tmp_raw_dir / "uk_hpi_full.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(stale_parquet)

    def fake_download(self, url, dest, desc=""):
        dest.write_text("new-content")
        return dest

    with patch.object(UKHousePriceIndex, "_download_file", fake_download):
        result = source.download(incremental=True)

    assert result == stale_csv
    assert stale_csv.read_text() == "new-content"
    assert not stale_parquet.exists()


def test_download_non_incremental_preserves_cache(source, tmp_raw_dir):
    """download() with default incremental=False keeps existing files."""
    cached = tmp_raw_dir / "uk_hpi_full.csv"
    cached.write_text("cached")

    def fake_download(self, url, dest, desc=""):
        return dest

    with patch.object(UKHousePriceIndex, "_download_file", fake_download):
        source.download()

    assert cached.read_text() == "cached"


def test_ingest_incremental_updates_manifest(source, tmp_raw_dir):
    """ingest(incremental=True) writes manifest with the cleaned frame's max date."""
    csv_rows = [
        "Date,RegionName,AreaCode,AveragePrice,Index",
        "01/03/2026,Westminster,E09000033,850000,145.2",
        "01/04/2026,Westminster,E09000033,855000,146.0",
        "01/04/2026,Tower Hamlets,E09000030,620000,138.5",
    ]

    def fake_download(self, url, dest, desc=""):
        # Simulate the upstream serving the latest snapshot.
        dest.write_text("\n".join(csv_rows))
        return dest

    with patch.object(UKHousePriceIndex, "_download_file", fake_download):
        cleaned_lf = source.ingest(incremental=True)

    cleaned = cleaned_lf.collect()
    assert cleaned.height == 3
    assert cleaned["date"].max().isoformat() == "2026-04-01"

    entry = manifest.read_manifest(tmp_raw_dir)[source.name]
    assert entry["last_data_through"] == "2026-04-01"
    assert entry["row_count"] == 3
    assert entry["source_url"].endswith(".csv")


def test_ingest_non_incremental_unchanged(source, tmp_raw_dir):
    """Non-incremental ingest works through the base-class path without manifest writes."""
    csv_path = tmp_raw_dir / "uk_hpi_full.csv"
    _write_hpi_csv(
        csv_path,
        ["01/01/2025,Camden,E09000007,700000,140.0"],
    )

    def fake_download(self, url, dest, desc=""):
        # Non-incremental: base class skips re-download when the file exists.
        return dest

    with patch.object(UKHousePriceIndex, "_download_file", fake_download):
        lf = source.ingest()
    assert lf.collect().height == 1
    assert source.name not in manifest.read_manifest(tmp_raw_dir)
