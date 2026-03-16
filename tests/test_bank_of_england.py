"""Tests for the Bank of England data source."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path  # noqa: TC003

import polars as pl
import pytest

from src.data.sources.bank_of_england import BankOfEnglandSeries


@pytest.fixture
def boe_csv_content() -> str:
    """Sample BoE bulk CSV in the format returned by the BoE API."""
    return textwrap.dedent("""\
        Date,IUDBEDR,IUMBV42,LPMVTVB
        01 Jan 2020,0.75,1.85,42.1
        02 Jan 2020,0.75,,
        01 Feb 2020,0.75,1.82,40.3
        01 Mar 2020,0.25,1.80,35.0
        01 Apr 2020,0.10,1.75,N/A
    """)


@pytest.fixture
def boe_source(tmp_path: Path) -> BankOfEnglandSeries:
    return BankOfEnglandSeries(raw_dir=tmp_path)


class TestBankOfEnglandInit:
    def test_default_series_codes(self, boe_source: BankOfEnglandSeries) -> None:
        from src.data.config import BOE_SERIES_CODES

        assert boe_source.series_codes == BOE_SERIES_CODES

    def test_custom_series_codes(self, tmp_path: Path) -> None:
        src = BankOfEnglandSeries(raw_dir=tmp_path, series_codes=["IUDBEDR"])
        assert src.series_codes == ["IUDBEDR"]


class TestBankOfEnglandLoad:
    def test_load_raises_if_missing(self, boe_source: BankOfEnglandSeries) -> None:
        with pytest.raises(FileNotFoundError, match=r"boe_series\.csv"):
            boe_source.load()

    def test_load_returns_lazy_frame(self, tmp_path: Path, boe_csv_content: str) -> None:
        csv_path = tmp_path / "boe_series.csv"
        csv_path.write_text(boe_csv_content)
        src = BankOfEnglandSeries(raw_dir=tmp_path)
        lf = src.load()
        assert isinstance(lf, pl.LazyFrame)


class TestBankOfEnglandClean:
    def _make_lf(self, csv_content: str) -> pl.LazyFrame:
        return pl.read_csv(io.StringIO(csv_content), null_values=["", "N/A", "n/a"]).lazy()

    def test_clean_produces_long_format(self, boe_csv_content: str) -> None:
        src = BankOfEnglandSeries()
        lf = self._make_lf(boe_csv_content)
        result = src.clean(lf).collect()
        assert "date" in result.columns
        assert "series" in result.columns
        assert "value" in result.columns

    def test_clean_date_dtype(self, boe_csv_content: str) -> None:
        src = BankOfEnglandSeries()
        lf = self._make_lf(boe_csv_content)
        result = src.clean(lf).collect()
        assert result["date"].dtype == pl.Date

    def test_clean_resamples_to_monthly(self, boe_csv_content: str) -> None:
        """Daily rows for Jan 2020 should collapse to one monthly row."""
        src = BankOfEnglandSeries()
        lf = self._make_lf(boe_csv_content)
        result = src.clean(lf).collect()
        iudbedr = result.filter(pl.col("series") == "IUDBEDR")
        # All three Jan dates share the same month-truncated date
        jan_rows = iudbedr.filter(pl.col("date").dt.month() == 1)
        assert len(jan_rows) == 1

    def test_clean_handles_null_values(self, boe_csv_content: str) -> None:
        src = BankOfEnglandSeries()
        lf = self._make_lf(boe_csv_content)
        result = src.clean(lf).collect()
        # N/A rows should yield null values (not crash)
        assert result["value"].dtype == pl.Float64

    def test_clean_series_codes_present(self, boe_csv_content: str) -> None:
        src = BankOfEnglandSeries()
        lf = self._make_lf(boe_csv_content)
        result = src.clean(lf).collect()
        series_in_result = set(result["series"].to_list())
        assert "IUDBEDR" in series_in_result
        assert "IUMBV42" in series_in_result
