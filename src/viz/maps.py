"""Local Authority District choropleth helpers.

Functions here are intentionally lazy — geopandas/folium are heavy imports
and the notebook is the only consumer, so we delay them until call time.

Boundary source: ONS Open Geography Portal "Local Authority Districts
(May 2024) Boundaries UK BUC" (ultra-generalised, ~2MB). The geojson is
cached under ``data/geo/lad_boundaries.geojson`` on first call. Override
the URL via the ``LAD_GEOJSON_URL`` env var or by passing ``url=`` if the
ONS layer name changes.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import requests

from src.data.config import DATA_DIR

if TYPE_CHECKING:
    from pathlib import Path

    import folium
    import geopandas as gpd

logger = logging.getLogger(__name__)

GEO_DIR = DATA_DIR / "geo"
LAD_GEOJSON_FILENAME = "lad_boundaries.geojson"
LAD_GEOJSON_DEFAULT_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_May_2024_Boundaries_UK_BUC/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)


def lad_geojson_path() -> Path:
    return GEO_DIR / LAD_GEOJSON_FILENAME


def download_lad_geojson(url: str | None = None, *, force: bool = False) -> Path:
    """Fetch the LAD boundary geojson and cache under ``data/geo/``.

    Returns the path to the cached file. Skips the fetch when already cached
    unless ``force=True``.
    """
    dest = lad_geojson_path()
    if dest.exists() and not force:
        return dest

    target_url = url or os.environ.get("LAD_GEOJSON_URL") or LAD_GEOJSON_DEFAULT_URL
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading LAD boundaries from %s", target_url)
    response = requests.get(target_url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_lad_geometries(url: str | None = None) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame keyed by an uppercase ``district`` column.

    The geojson uses ``LAD24NM`` for names; the Land Registry pipeline
    uppercases district names in cleaning, so we mirror that here to make
    joins on ``district`` straightforward.
    """
    import geopandas as gpd

    path = download_lad_geojson(url=url)
    gdf = gpd.read_file(path)

    name_col = next(
        (c for c in ("LAD24NM", "LAD23NM", "LAD22NM", "LAD21NM", "lad_name") if c in gdf.columns),
        None,
    )
    if name_col is None:
        raise RuntimeError(
            f"Could not find a LAD name column in {path}; expected one of LAD24NM/LAD23NM/..."
        )
    gdf = gdf.rename(columns={name_col: "lad_name"})
    gdf["district"] = gdf["lad_name"].str.upper().str.strip()
    return gdf[["district", "lad_name", "geometry"]]


def choropleth(
    values_by_district: dict[str, float],
    *,
    title: str = "",
    legend_name: str = "",
    fill_color: str = "YlOrRd",
    nan_fill_color: str = "lightgrey",
    bins: int = 7,
    geometries: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Render a folium choropleth keyed on uppercase district names.

    Districts present in the geometries but missing from
    ``values_by_district`` are rendered in ``nan_fill_color``.
    """
    import folium

    gdf = geometries if geometries is not None else load_lad_geometries()
    gdf = gdf.copy()
    gdf["value"] = gdf["district"].map(values_by_district)

    centroid = gdf.geometry.union_all().centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=6, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf,
        columns=["district", "value"],
        key_on="feature.properties.district",
        fill_color=fill_color,
        fill_opacity=0.75,
        line_opacity=0.2,
        nan_fill_color=nan_fill_color,
        nan_fill_opacity=0.4,
        legend_name=legend_name or title,
        bins=bins,
    ).add_to(fmap)

    folium.GeoJson(
        gdf,
        style_function=lambda _: {"fillOpacity": 0, "color": "transparent"},
        tooltip=folium.GeoJsonTooltip(
            fields=["lad_name", "value"],
            aliases=["District:", f"{legend_name or 'Value'}:"],
            localize=True,
            sticky=False,
            labels=True,
        ),
        name="hover",
    ).add_to(fmap)

    return fmap
