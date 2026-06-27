"""Raster and vector geo utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, Point


def bbox_to_polygon(bbox: dict[str, float]) -> "shapely.geometry.Polygon":
    """Convert a bbox dict {"north", "south", "east", "west"} to a Shapely polygon."""
    return box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])


def reproject_match(src_path: Path, ref_path: Path, output_path: Path) -> Path:
    """Reproject and resample src to match the CRS, transform, and shape of ref.

    Args:
        src_path: source raster to reproject.
        ref_path: reference raster (target grid).
        output_path: where to write the reprojected raster.

    Returns:
        output_path.
    """
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height, ref_width = ref.height, ref.width

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )
    return output_path


def raster_to_points(raster_path: Path, band: int = 1) -> gpd.GeoDataFrame:
    """Convert every non-nodata pixel in a raster to a GeoDataFrame of points."""
    with rasterio.open(raster_path) as src:
        data = src.read(band)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    rows, cols = np.where(data != nodata if nodata is not None else np.ones_like(data, dtype=bool))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    values = data[rows, cols]

    return gpd.GeoDataFrame(
        {"value": values, "geometry": [Point(x, y) for x, y in zip(xs, ys)]},
        crs=crs,
    )


def write_suitability_tif(
    scores: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> None:
    """Write a float32 suitability score raster, inheriting CRS and transform from reference."""
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()

    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(scores.astype(np.float32), 1)
