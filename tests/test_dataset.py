"""Unit tests for the SentinelCropDataset."""

import numpy as np
import pytest
import rasterio
import torch
from pathlib import Path
from rasterio.transform import from_bounds

from src.data.dataset import SentinelCropDataset, NUM_CLASSES


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal fake dataset with 2 samples."""
    imagery_dir = tmp_path / "imagery"
    labels_dir = tmp_path / "labels"
    imagery_dir.mkdir()
    labels_dir.mkdir()

    transform = from_bounds(0, 0, 1, 1, 256, 256)
    profile = dict(driver="GTiff", dtype="float32", width=256, height=256, crs="EPSG:4326", transform=transform)

    for i in range(2):
        with rasterio.open(imagery_dir / f"img_{i:02d}.tif", "w", count=12, **profile) as dst:
            dst.write(np.random.rand(12, 256, 256).astype(np.float32))

        label_profile = {**profile, "dtype": "int32", "count": 1}
        with rasterio.open(labels_dir / f"img_{i:02d}.tif", "w", **label_profile) as dst:
            dst.write(np.random.randint(0, NUM_CLASSES, (1, 256, 256)).astype(np.int32))

    return imagery_dir, labels_dir


def test_dataset_length(tmp_dataset):
    imagery_dir, labels_dir = tmp_dataset
    ds = SentinelCropDataset(imagery_dir, labels_dir)
    assert len(ds) == 2


def test_sample_shapes(tmp_dataset):
    imagery_dir, labels_dir = tmp_dataset
    ds = SentinelCropDataset(imagery_dir, labels_dir)
    sample = ds[0]
    assert sample["image"].shape == (12, 256, 256)
    assert sample["mask"].shape == (256, 256)


def test_sample_dtypes(tmp_dataset):
    imagery_dir, labels_dir = tmp_dataset
    ds = SentinelCropDataset(imagery_dir, labels_dir)
    sample = ds[0]
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.int64


def test_mask_values_in_range(tmp_dataset):
    imagery_dir, labels_dir = tmp_dataset
    ds = SentinelCropDataset(imagery_dir, labels_dir)
    for i in range(len(ds)):
        mask = ds[i]["mask"]
        assert mask.min() >= 0
        assert mask.max() < NUM_CLASSES
