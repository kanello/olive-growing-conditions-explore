"""Sentinel-2 L2A cloud-free composite builder and spectral index computation."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


BAND_WAVELENGTHS = {
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Red Edge 1",
    "B06": "Red Edge 2",
    "B07": "Red Edge 3",
    "B08": "NIR",
    "B8A": "Narrow NIR",
    "B11": "SWIR 1",
    "B12": "SWIR 2",
}


class BoundingBox(NamedTuple):
    west: float
    south: float
    east: float
    north: float


def build_composite(
    bbox: BoundingBox,
    date_range: tuple[str, str],
    bands: list[str] | None = None,
    cloud_cover_max: int = 20,
    output_dir: Path | None = None,
) -> Path:
    """Download and median-composite Sentinel-2 L2A scenes for a given bounding box.

    Uses the Copernicus Data Space STAC API (free account required).
    Returns path to a multi-band GeoTIFF.

    Args:
        bbox: (west, south, east, north) in WGS-84.
        date_range: ("YYYY-MM-DD", "YYYY-MM-DD") start/end dates.
        bands: list of band names; defaults to all 10 bands in BAND_WAVELENGTHS.
        cloud_cover_max: maximum scene cloud cover percentage to include.
        output_dir: where to write the composite GeoTIFF.
    """
    raise NotImplementedError


def compute_indices(
    composite_path: Path,
    band_order: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute NDVI, EVI, NDRE, and NDWI from a Sentinel-2 composite.

    Args:
        composite_path: path to a multi-band GeoTIFF produced by build_composite().
        band_order: list of band names matching the GeoTIFF band order.

    Returns:
        dict with keys "ndvi", "evi", "ndre", "ndwi" and 2-D float32 arrays.
    """
    raise NotImplementedError


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with zero-division guard."""
    return np.where((a + b) == 0, 0.0, (a - b) / (a + b)).astype(np.float32)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _safe_divide(nir, red)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"""
    denom = nir + 6 * red - 7.5 * blue + 1
    return np.where(denom == 0, 0.0, 2.5 * (nir - red) / denom).astype(np.float32)


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Red Edge NDVI — strong discriminator for olive canopies."""
    return _safe_divide(nir, red_edge)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_divide(green, nir)
