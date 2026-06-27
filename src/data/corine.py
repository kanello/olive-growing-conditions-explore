"""CORINE Land Cover download and olive grove (class 223) point sampling."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


OLIVE_CLASS = 223
ABSENCE_CLASSES = {211, 212, 213, 221, 222, 231, 242, 243}  # agricultural non-olive


def _code_col(gdf: gpd.GeoDataFrame) -> str:
    """Return the CORINE class code column name (handles both 'Code_18' and 'CODE_18')."""
    for candidate in ("Code_18", "CODE_18", "code_18"):
        if candidate in gdf.columns:
            return candidate
    raise KeyError(f"No CORINE code column found. Columns: {list(gdf.columns)}")

# EEA ArcGIS REST endpoint — no auth required, but max ~2000 features per request
_EEA_WFS_BASE = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services"
    "/Corine/CLC2018_WM/MapServer/0/query"
)


def download_corine(
    output_dir: Path,
    url: str | None = None,
) -> Path:
    """Download the CORINE Land Cover 2018 pan-European GeoPackage.

    CORINE has no token API — data is distributed as a ZIP file from the
    Copernicus Land Service portal. Steps to get the URL:

        1. Register free at https://land.copernicus.eu (EU Login)
        2. Go to: CORINE Land Cover → CLC 2018 → GeoPackage download
        3. Right-click the download button → copy link address
        4. Set CORINE_DOWNLOAD_URL=<that URL> in your .env file
           OR pass it directly as the `url` argument.

    Args:
        output_dir: directory to save the extracted GeoPackage.
        url: direct download URL from the portal. Falls back to the
             CORINE_DOWNLOAD_URL environment variable if not provided.

    Returns:
        Path to the extracted .gpkg file.
    """
    import requests
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for already-extracted GeoPackage or File GDB
    existing_gpkg = list(output_dir.glob("*.gpkg"))
    if existing_gpkg:
        print(f"CORINE already downloaded: {existing_gpkg[0]}")
        return existing_gpkg[0]
    existing_gdb = [p for p in output_dir.rglob("*.gdb") if p.is_dir()]
    if existing_gdb:
        print(f"CORINE already downloaded (GDB): {existing_gdb[0]}")
        return existing_gdb[0]

    download_url = url or os.environ.get("CORINE_DOWNLOAD_URL")
    if not download_url:
        raise ValueError(
            "No download URL provided. Set CORINE_DOWNLOAD_URL in your .env file.\n"
            "Get the URL by registering at https://land.copernicus.eu and copying\n"
            "the download link from the CORINE 2018 product page."
        )

    zip_path = output_dir / "corine_2018.zip"
    print(f"Downloading CORINE 2018 → {zip_path}")

    with requests.get(download_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="CORINE download"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Extracting {zip_path}…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)

    zip_path.unlink()

    # Return whichever format was extracted
    for gpkg in output_dir.glob("*.gpkg"):
        print(f"Done: {gpkg}")
        return gpkg
    for gdb in output_dir.rglob("*.gdb"):
        if gdb.is_dir():
            print(f"Done: {gdb}")
            return gdb

    raise RuntimeError(f"Extraction succeeded but no .gpkg or .gdb found in {output_dir}")


def download_corine_wfs(
    bbox: dict[str, float],
    output_path: Path,
    class_code: int = OLIVE_CLASS,
    max_features: int = 10_000,
) -> Path:
    """Query CORINE class-223 polygons from the EEA ArcGIS REST service for a bounding box.

    Does NOT require an account. Returns a GeoJSON file. Suitable for small areas
    (single country or sub-national region); paginates automatically.

    Note: the EEA WFS endpoint uses EPSG:3857 (Web Mercator) for its spatial filter.

    Args:
        bbox: {"west": ..., "south": ..., "east": ..., "north": ...} in WGS-84 decimal degrees.
        output_path: path to write the output GeoJSON file.
        class_code: CORINE class to filter (default 223 = olive groves).
        max_features: maximum total features to retrieve (safety cap).

    Returns:
        Path to the written GeoJSON file.
    """
    import math
    import json
    import requests

    def _to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
        x = lon * 20037508.342 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.342 / 180
        return x, y

    west, south = _to_web_mercator(bbox["west"], bbox["south"])
    east, north = _to_web_mercator(bbox["east"], bbox["north"])

    features: list[dict] = []
    offset = 0
    page_size = 1000

    print(f"Querying EEA WFS for CORINE class {class_code} in bbox…")
    while len(features) < max_features:
        params = {
            "where": f"CODE_18={class_code}",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 3857,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "CODE_18,Shape_Area",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "geojson",
        }
        r = requests.get(_EEA_WFS_BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        batch = data.get("features", [])
        features.extend(batch)
        print(f"  Retrieved {len(features)} features so far…")

        # ArcGIS signals more pages with exceededTransferLimit
        if not data.get("exceededTransferLimit") or not batch:
            break
        offset += page_size

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"Saved {len(features)} features → {output_path}")
    return output_path


def extract_olive_polygons(
    corine_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Return all class-223 (olive grove) polygons from a CORINE file or GDB.

    Args:
        corine_path: path to a .gpkg file, .gdb directory, or GeoJSON.
        bbox: optional (minx, miny, maxx, maxy) spatial filter in the file's
            native CRS to speed up reads on the pan-European dataset.
    """
    kwargs = {"bbox": bbox} if bbox is not None else {}
    gdf = gpd.read_file(corine_path, **kwargs)
    col = _code_col(gdf)
    return gdf[gdf[col].astype(int) == OLIVE_CLASS].copy()


def sample_presence_absence(
    corine_path: Path,
    n_samples: int = 5000,
    seed: int = 42,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Sample n_samples presence (olive) and n_samples pseudo-absence (non-olive ag) points.

    Points are sampled uniformly within each class's polygon area.
    Returns a GeoDataFrame with columns: geometry, label (1=olive, 0=absence).

    Args:
        corine_path: path to CORINE .gpkg, .gdb directory, or GeoJSON.
        n_samples: number of presence points (and absence points) to generate.
        seed: random seed.
        bbox: optional (minx, miny, maxx, maxy) spatial filter in native CRS.
    """
    rng = np.random.default_rng(seed)
    kwargs = {"bbox": bbox} if bbox is not None else {}
    gdf = gpd.read_file(corine_path, **kwargs)
    col = _code_col(gdf)

    presences = _random_points_in_polygons(
        gdf[gdf[col].astype(int) == OLIVE_CLASS], n_samples, rng
    )
    presences["label"] = 1

    absence_mask = gdf[col].astype(int).isin(ABSENCE_CLASSES)
    absences = _random_points_in_polygons(gdf[absence_mask], n_samples, rng)
    absences["label"] = 0

    combined = gpd.GeoDataFrame(
        pd.concat([presences, absences], ignore_index=True),
        crs=gdf.crs,
    )
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def _random_points_in_polygons(
    gdf: gpd.GeoDataFrame, n: int, rng: np.random.Generator
) -> gpd.GeoDataFrame:
    """Uniformly sample n points across all polygons, weighted by area."""
    areas = gdf.geometry.area.values
    weights = areas / areas.sum()
    counts = rng.multinomial(n, weights)

    points: list[Point] = []
    for geom, count in zip(gdf.geometry, counts):
        minx, miny, maxx, maxy = geom.bounds
        sampled = 0
        while sampled < count:
            candidates = rng.uniform([minx, miny], [maxx, maxy], size=(count * 4, 2))
            for x, y in candidates:
                pt = Point(x, y)
                if geom.contains(pt):
                    points.append(pt)
                    sampled += 1
                    if sampled >= count:
                        break

    return gpd.GeoDataFrame(geometry=points)
