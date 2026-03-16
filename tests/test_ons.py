"""Tests for the ONS time series data source."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import polars as pl
import pytest

from src.data.sources.ons import ONSTimeSeries


def _sample_ons_json() -> list[dict]:
    """Minimal ONS API response structure for two series."""
    return [
        {
            "series": "KAB9",
            "data": {
                "months": [
                    {"date": "2020 Jan", "value": "570.5"},
                    {"date": "2020 Feb", "value": "572.3"},
                    {"date": "2020 Mar", "value": "568.0"},
                ],
                "quarters": [],
                "years": [],
            },
        },
        {
            "series": "IHYQ",
            "data": {
                "months": [],
                "quarters": [
                    {"date": "2020 Q1", "value": "-0.2"},
                    {"date": "2020 Q2", "value": "-19.8"},
                ],
                "years": [],
            },
        },
    ]


@pytest.fixture
def ons_json_path(tmp_path: Path) -> Path:
    path = tmp_path / "ons_series.json"
    path.write_text(json.dumps(_sample_ons_json()))
    return path


@pytest.fixture
def ons_source(tmp_path: Path) -> ONSTimeSeries:
    return ONSTimeSeries(raw_dir=tmp_path)


class TestONSInit:
    def test_default_series(self, ons_source: ONSTimeSeries) -> None:
        from src.data.config import ONS_SERIES

        assert ons_source.series == list(ONS_SERIES)

    def test_custom_series(self, tmp_path: Path) -> None:
        src = ONSTimeSeries(raw_dir=tmp_path, series=[("KAB9", "EARN")])
        assert src.series == [("KAB9", "EARN")]


class TestONSLoad:
    def test_load_raises_if_missing(self, ons_source: ONSTimeSeries) -> None:
        with pytest.raises(FileNotFoundError, match=r"ons_series\.json"):
            ons_source.load()

    def test_load_returns_lazy_frame(self, ons_source: ONSTimeSeries, ons_json_path: Path) -> None:
        lf = ons_source.load(filepath=ons_json_path)
        assert isinstance(lf, pl.LazyFrame)

    def test_load_has_expected_columns(
        self, ons_source: ONSTimeSeries, ons_json_path: Path
    ) -> None:
        lf = ons_source.load(filepath=ons_json_path)
        cols = lf.collect_schema().names()
        assert "series" in cols
        assert "raw_date" in cols
        assert "value" in cols


class TestONSClean:
    def test_clean_date_dtype(self, ons_source: ONSTimeSeries, ons_json_path: Path) -> None:
        lf = ons_source.load(filepath=ons_json_path)
        result = ons_source.clean(lf).collect()
        assert result["date"].dtype == pl.Date

    def test_clean_monthly_dates(self, ons_source: ONSTimeSeries, ons_json_path: Path) -> None:
        lf = ons_source.load(filepath=ons_json_path)
        result = ons_source.clean(lf).collect()
        kab9 = result.filter(pl.col("series") == "KAB9")
        assert len(kab9) == 3

    def test_clean_quarterly_dates(self, ons_source: ONSTimeSeries, ons_json_path: Path) -> None:
        lf = ons_source.load(filepath=ons_json_path)
        result = ons_source.clean(lf).collect()
        ihyq = result.filter(pl.col("series") == "IHYQ")
        assert len(ihyq) == 2
        # Q1 should map to January
        q1_row = ihyq.filter(pl.col("date").dt.month() == 1)
        assert len(q1_row) == 1

    def test_clean_value_dtype(self, ons_source: ONSTimeSeries, ons_json_path: Path) -> None:
        lf = ons_source.load(filepath=ons_json_path)
        result = ons_source.clean(lf).collect()
        assert result["value"].dtype == pl.Float64

    def test_clean_empty_json(self, tmp_path: Path) -> None:
        path = tmp_path / "ons_series.json"
        path.write_text(json.dumps([]))
        src = ONSTimeSeries(raw_dir=tmp_path)
        lf = src.load(filepath=path)
        result = src.clean(lf).collect()
        assert result.is_empty()
