"""Unified feature extraction pipeline: assembles all data sources into feature_matrix.parquet."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml


class FeatureExtractor:
    """Orchestrates extraction of spectral, climate, terrain, and soil features.

    Usage:
        extractor = FeatureExtractor.from_config("configs/kythnos.yaml")
        feature_matrix = extractor.extract(points_gdf)
        feature_matrix.to_parquet("data/features/feature_matrix.parquet")
    """

    SPECTRAL_SEASONS = ("summer", "winter")
    SPECTRAL_INDICES = ("ndvi", "evi", "ndre", "ndwi")

    def __init__(self, config: dict) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FeatureExtractor":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def extract(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Run all feature extractors and join results into a single DataFrame.

        Args:
            points_gdf: sample points with a "label" column (1=olive, 0=absence).

        Returns:
            DataFrame with ~25 feature columns + "label" column, one row per point.
        """
        spectral = self._extract_spectral(points_gdf)
        climate = self._extract_climate(points_gdf)
        terrain = self._extract_terrain(points_gdf)
        soil = self._extract_soil(points_gdf)

        features = pd.concat([spectral, climate, terrain, soil], axis=1)
        features["label"] = points_gdf["label"].values
        return features

    def _extract_spectral(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Sample spectral indices at each point for summer and winter composites."""
        from src.data import sentinel

        cols: dict[str, list] = {}
        for season in self.SPECTRAL_SEASONS:
            date_range = tuple(self.config["sentinel2"]["seasons"][season])
            composite_path = self._composite_path(season)
            if not composite_path.exists():
                sentinel.build_composite(
                    bbox=self._bbox(),
                    date_range=date_range,
                    bands=self.config["sentinel2"]["bands"],
                    cloud_cover_max=self.config["sentinel2"]["cloud_cover_max"],
                    output_dir=Path(self.config["data"]["raw_dir"]) / "sentinel2",
                )
            indices = sentinel.compute_indices(composite_path)
            for name, arr in indices.items():
                cols[f"{name}_{season}"] = self._sample_raster(arr, points_gdf)

        return pd.DataFrame(cols, index=points_gdf.index)

    def _extract_climate(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        from src.data import climate

        era5_dir = Path(self.config["data"]["raw_dir"]) / "era5"
        return climate.extract_climate_at_points(points_gdf, era5_dir)

    def _extract_terrain(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        from src.data import terrain

        dem_path = Path(self.config["data"]["raw_dir"]) / "dem" / "copernicus_dem.tif"
        features = terrain.compute_terrain_features(dem_path)
        rows = {k: self._sample_raster(arr, points_gdf) for k, arr in features.items()}
        return pd.DataFrame(rows, index=points_gdf.index)

    def _extract_soil(self, points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        from src.data import soil

        return soil.fetch_soilgrids(
            points_gdf,
            variables=self.config["soil"]["variables"],
            depth=self.config["soil"]["depth"],
        )

    def _composite_path(self, season: str) -> Path:
        return Path(self.config["data"]["raw_dir"]) / "sentinel2" / f"composite_{season}.tif"

    def _bbox(self):
        from src.data.sentinel import BoundingBox

        b = self.config["area"]["bbox"]
        return BoundingBox(west=b["west"], south=b["south"], east=b["east"], north=b["north"])

    @staticmethod
    def _sample_raster(arr, points_gdf: gpd.GeoDataFrame) -> list:
        """Placeholder: sample 2-D raster array at point locations (nearest pixel)."""
        raise NotImplementedError
