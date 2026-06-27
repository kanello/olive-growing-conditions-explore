"""Copernicus DEM (GLO-30) download and terrain feature computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter


TERRAIN_FEATURES = ["elevation", "slope", "northness", "eastness", "twi", "tpi", "roughness"]

# COG tiles are hosted publicly on AWS S3 — no credentials needed
_COG_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)

_TPI_WINDOW = 21    # ~630 m radius at 30 m resolution
_ROUGH_WINDOW = 9   # ~270 m radius


def _tile_urls(bbox: dict[str, float]) -> list[str]:
    """Return COG URLs for all 1° tiles that intersect the given bbox."""
    lons = range(int(np.floor(bbox["west"])), int(np.floor(bbox["east"])) + 1)
    lats = range(int(np.floor(bbox["south"])), int(np.floor(bbox["north"])) + 1)
    urls = []
    for lat in lats:
        for lon in lons:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            urls.append(_COG_URL.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon)))
    return urls


def download_cop_dem(
    bbox: dict[str, float],
    output_dir: Path,
    resolution: str = "GLO-30",
) -> Path:
    """Stream Copernicus DEM GLO-30 tiles for a bbox and save a clipped GeoTIFF.

    Uses rasterio's /vsicurl/ virtual filesystem to read only the required
    window from each Cloud Optimized GeoTIFF — no full tile download needed.
    No authentication required.

    Args:
        bbox: {"north": ..., "south": ..., "east": ..., "west": ...} in WGS-84.
        output_dir: directory to save the mosaicked DEM.
        resolution: "GLO-30" (30 m) only.

    Returns:
        Path to saved DEM GeoTIFF in EPSG:4326.
    """
    import rasterio
    import rasterio.merge

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dem.tif"

    if out_path.exists():
        print(f"DEM already exists: {out_path}")
        return out_path

    urls = _tile_urls(bbox)
    print(f"Fetching {len(urls)} tile(s) from Copernicus DEM GLO-30…")

    datasets = []
    for url in urls:
        try:
            ds = rasterio.open(f"/vsicurl/{url}")
            datasets.append(ds)
            print(f"  Opened: {url.split('/')[-2]}")
        except Exception as e:
            print(f"  Warning: could not open {url.split('/')[-2]}: {e}")

    if not datasets:
        raise RuntimeError("No DEM tiles could be opened. Check internet connectivity.")

    mosaic, transform = rasterio.merge.merge(
        datasets,
        bounds=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
    )
    for ds in datasets:
        ds.close()

    profile = {
        "driver": "GTiff",
        "dtype": mosaic.dtype,
        "width": mosaic.shape[2],
        "height": mosaic.shape[1],
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "nodata": -9999,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)

    print(f"Saved → {out_path}  ({mosaic.shape[2]}×{mosaic.shape[1]} px at ~30 m)")
    return out_path


def compute_terrain_features(dem_path: Path) -> tuple[dict[str, np.ndarray], object, object]:
    """Derive terrain features from a DEM GeoTIFF.

    Returns:
        Tuple of (features_dict, rasterio_transform, rasterio_crs).
        features_dict keys: elevation, slope, northness, eastness, twi, tpi, roughness.
        All arrays are float32, same shape as the input DEM.
    """
    import rasterio

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        cy = (src.bounds.top + src.bounds.bottom) / 2
        cell_x_m = abs(transform.a) * 111_320 * np.cos(np.radians(cy))
        cell_y_m = abs(transform.e) * 111_320
        cell_size_m = (cell_x_m + cell_y_m) / 2

    if nodata is not None:
        elevation[elevation == nodata] = np.nan

    slope_deg, aspect_deg = _slope_aspect(elevation, cell_size_m)
    northness, eastness = _northness_eastness(aspect_deg)
    twi = _twi_proxy(slope_deg)
    tpi = _tpi(elevation, window=_TPI_WINDOW)
    roughness = _roughness(elevation, window=_ROUGH_WINDOW)

    features = {
        "elevation": elevation,
        "slope": slope_deg,
        "northness": northness,
        "eastness": eastness,
        "twi": twi,
        "tpi": tpi,
        "roughness": roughness,
    }
    return features, transform, crs


def _slope_aspect(elevation: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return slope (degrees) and aspect (degrees, 0=N clockwise) from an elevation grid."""
    dy, dx = np.gradient(elevation, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)
    return np.degrees(slope_rad).astype(np.float32), np.degrees(aspect_rad).astype(np.float32)


def _northness_eastness(aspect_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert aspect to northness (cos) and eastness (sin) components."""
    rad = np.radians(aspect_deg)
    return np.cos(rad).astype(np.float32), np.sin(rad).astype(np.float32)


def _topographic_wetness_index(
    slope_deg: np.ndarray,
    flow_accumulation: np.ndarray,
    cell_area_m2: float,
) -> np.ndarray:
    """TWI = ln(flow_acc * cell_area / tan(slope)); higher values → wetter."""
    slope_rad = np.radians(np.clip(slope_deg, 0.1, 89.9))
    tan_slope = np.tan(slope_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.log((flow_accumulation * cell_area_m2) / tan_slope)
    return np.where(np.isfinite(twi), twi, 0.0).astype(np.float32)


def _twi_proxy(slope_deg: np.ndarray) -> np.ndarray:
    """Slope-based TWI proxy: -log(tan(slope)). Flat areas score high (wetter).

    Avoids D8 flow routing. Sufficient for SDM use where TWI is one of ~25
    predictors. Upgrade to proper D8 if TWI shows high feature importance.
    """
    slope_rad = np.radians(np.clip(slope_deg, 0.1, 89.9))
    return (-np.log(np.tan(slope_rad))).astype(np.float32)


def _tpi(elevation: np.ndarray, window: int = 21) -> np.ndarray:
    """Topographic Position Index: elevation minus local mean. Ridges positive, valleys negative."""
    local_mean = uniform_filter(elevation, size=window, mode="nearest")
    return (elevation - local_mean).astype(np.float32)


def _roughness(elevation: np.ndarray, window: int = 9) -> np.ndarray:
    """Surface roughness: local elevation range (max − min) in a moving window."""
    local_max = maximum_filter(elevation, size=window, mode="nearest")
    local_min = minimum_filter(elevation, size=window, mode="nearest")
    return (local_max - local_min).astype(np.float32)
