"""Tests for the LAD choropleth helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

geopandas = pytest.importorskip("geopandas")
folium = pytest.importorskip("folium")

from src.viz import maps as viz_maps  # noqa: E402


def _toy_geojson():
    """A tiny LAD-shaped geojson with two square polygons, LAD24NM names."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"LAD24NM": "Westminster", "LAD24CD": "E09000033"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"LAD24NM": "Camden", "LAD24CD": "E09000007"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                },
            },
        ],
    }


@pytest.fixture
def toy_lad_path(tmp_path, monkeypatch):
    """Pre-stage a toy geojson where download_lad_geojson would put one."""
    geo_dir = tmp_path / "data" / "geo"
    geo_dir.mkdir(parents=True)
    target = geo_dir / "lad_boundaries.geojson"
    target.write_text(json.dumps(_toy_geojson()))
    monkeypatch.setattr(viz_maps, "GEO_DIR", geo_dir)
    return target


def test_download_lad_geojson_skips_when_cached(toy_lad_path):
    """download_lad_geojson must not hit the network when the cache exists."""
    with patch.object(viz_maps.requests, "get") as mocked:
        result = viz_maps.download_lad_geojson()
    assert result == toy_lad_path
    mocked.assert_not_called()


def test_load_lad_geometries_normalises_district_key(toy_lad_path):
    gdf = viz_maps.load_lad_geometries()
    assert set(gdf.columns) == {"district", "lad_name", "geometry"}
    assert set(gdf["district"].tolist()) == {"WESTMINSTER", "CAMDEN"}
    assert set(gdf["lad_name"].tolist()) == {"Westminster", "Camden"}


def test_choropleth_returns_folium_map(toy_lad_path):
    fmap = viz_maps.choropleth(
        {"WESTMINSTER": 1.0, "CAMDEN": 2.0},
        title="Test",
        legend_name="value",
    )
    assert isinstance(fmap, folium.Map)
    fmap2 = viz_maps.choropleth(
        {"WESTMINSTER": 1.0},
        title="Partial",
        legend_name="value",
    )
    assert isinstance(fmap2, folium.Map)
