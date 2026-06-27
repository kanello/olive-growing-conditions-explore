"""SoilGrids 250m REST API client for soil feature extraction."""
from __future__ import annotations

import time
from typing import Any

import geopandas as gpd
import pandas as pd
import requests


SOILGRIDS_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"

SOIL_VARIABLES = ["phh2o", "clay", "sand", "soc", "bdod"]

DEPTH_LAYERS = {
    "0-5cm": "0-5cm_mean",
    "5-15cm": "5-15cm_mean",
    "15-30cm": "15-30cm_mean",
}


def fetch_soilgrids(
    points_gdf: gpd.GeoDataFrame,
    variables: list[str] | None = None,
    depth: str = "0-30cm",
    batch_size: int = 1,
    request_delay_s: float = 1.0,
) -> pd.DataFrame:
    """Fetch SoilGrids values at each point via the REST API.

    SoilGrids is a free public API (no key required); rate-limit is ~1 req/s.

    Args:
        points_gdf: points in WGS-84 (EPSG:4326).
        variables: SoilGrids property names; defaults to SOIL_VARIABLES.
        depth: "0-30cm" returns mean of 0-5, 5-15, 15-30 cm layers.
        batch_size: points per request (API accepts 1 at a time for /query).
        request_delay_s: sleep between requests to respect rate limit.

    Returns:
        DataFrame indexed to match points_gdf, with one column per variable.
    """
    if variables is None:
        variables = SOIL_VARIABLES

    points_wgs84 = points_gdf.to_crs("EPSG:4326")
    records: list[dict[str, Any]] = []

    for _, row in points_wgs84.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        values = _query_point(lon, lat, variables)
        records.append(values)
        time.sleep(request_delay_s)

    return pd.DataFrame(records, index=points_gdf.index)


def _query_point(lon: float, lat: float, variables: list[str]) -> dict[str, float]:
    """Hit the SoilGrids REST endpoint for a single point; return variable values."""
    params = {
        "lon": lon,
        "lat": lat,
        "property": variables,
        "depth": ["0-5cm", "5-15cm", "15-30cm"],
        "value": ["mean"],
    }
    response = requests.get(SOILGRIDS_API, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return _parse_response(data, variables)


def _parse_response(data: dict[str, Any], variables: list[str]) -> dict[str, float]:
    """Average the 0-5, 5-15, 15-30 cm layers to produce a 0-30 cm mean."""
    result: dict[str, float] = {}
    for prop in data.get("properties", {}).get("layers", []):
        name = prop["name"]
        if name not in variables:
            continue
        depths = prop.get("depths", [])
        values = [
            d["values"]["mean"]
            for d in depths
            if d["label"] in DEPTH_LAYERS and d["values"].get("mean") is not None
        ]
        result[name] = sum(values) / len(values) if values else float("nan")
    return result
