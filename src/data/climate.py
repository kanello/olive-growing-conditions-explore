"""ERA5-Land / CHELSA climate variable extraction."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "2m_dewpoint_temperature",
]

DERIVED_FEATURES = [
    "mean_annual_temp",
    "min_temp_coldest_month",
    "max_temp_warmest_month",
    "annual_precip_mm",
    "summer_precip_mm",
    "frost_days_per_year",
]


def download_era5(
    variables: list[str],
    bbox: dict[str, float],
    years: list[int],
    output_dir: Path,
) -> list[Path]:
    """Download ERA5-Land monthly data via the CDS API.

    Requires a ~/.cdsapirc credentials file (free account at cds.climate.copernicus.eu).

    Args:
        variables: ERA5-Land short names (see ERA5_VARIABLES).
        bbox: {"north": ..., "south": ..., "east": ..., "west": ...} in WGS-84.
        years: list of years to download (recommend 1990–2020 for climatology).
        output_dir: destination directory; one .nc file per variable.

    Returns:
        List of paths to downloaded NetCDF files.
    """
    raise NotImplementedError


def extract_climate_at_points(
    points_gdf: gpd.GeoDataFrame,
    era5_dir: Path,
) -> pd.DataFrame:
    """Sample ERA5-Land climatology at each point and return derived features.

    Computes DERIVED_FEATURES from the raw ERA5 NetCDF files.
    Returns a DataFrame with one row per point, indexed to match points_gdf.
    """
    raise NotImplementedError


def _compute_frost_days(daily_tmin: "xr.DataArray") -> "xr.DataArray":  # noqa: F821
    """Count days per year where daily minimum temperature < 0 °C."""
    raise NotImplementedError
