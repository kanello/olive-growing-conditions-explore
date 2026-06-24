"""
Sentinel-2 crop segmentation dataset.

Wraps torchgeo's Sentinel2 dataset with EuroCrops labels,
producing (image, mask) pairs for semantic segmentation.
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import RandomGeoSampler

import rasterio
import numpy as np


# Band order for Sentinel-2 L2A (all 13 bands)
SENTINEL2_BANDS = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B08", "B8A", "B09",
    "B11", "B12",
]

# Crop type label mapping (EuroCrops HCAT Level 1)
CROP_CLASSES = {
    0: "background",
    1: "cereals",
    2: "root_crops",
    3: "vegetables",
    4: "fruits",
    5: "oilseeds",
    6: "fodder",
    7: "other",
}

NUM_CLASSES = len(CROP_CLASSES)


class SentinelCropDataset(Dataset):
    """
    Patches of Sentinel-2 imagery with corresponding crop type masks.

    Args:
        imagery_dir: Root directory containing Sentinel-2 GeoTIFF files.
        labels_dir: Root directory containing crop label rasters.
        patch_size: Spatial size of each patch in pixels.
        bands: List of band names to load. Defaults to all 12 bands.
        transforms: Optional torchvision/albumentations transform pipeline.
    """

    def __init__(
        self,
        imagery_dir: str | Path,
        labels_dir: str | Path,
        patch_size: int = 256,
        bands: list[str] | None = None,
        transforms=None,
    ):
        self.imagery_dir = Path(imagery_dir)
        self.labels_dir = Path(labels_dir)
        self.patch_size = patch_size
        self.bands = bands or SENTINEL2_BANDS
        self.transforms = transforms

        self.image_paths = sorted(self.imagery_dir.glob("*.tif"))
        self.label_paths = sorted(self.labels_dir.glob("*.tif"))

        assert len(self.image_paths) == len(self.label_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        image = self._load_raster(self.image_paths[idx])  # (C, H, W)
        mask = self._load_raster(self.label_paths[idx], squeeze=True)  # (H, W)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _load_raster(self, path: Path, squeeze: bool = False) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
        if squeeze:
            data = data.squeeze(0)  # (H, W) for single-band masks
        return data
