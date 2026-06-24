# Sentinel-2 Crop Segmentation

Semantic segmentation of agricultural land cover using Sentinel-2 multispectral satellite imagery and deep learning.

## Overview

This project uses freely available Sentinel-2 satellite imagery to segment and classify crop types across agricultural regions. It leverages the [`torchgeo`](https://github.com/microsoft/torchgeo) library for geospatial deep learning and a U-Net style architecture for pixel-wise classification.

**Key features:**
- Downloads and preprocesses Sentinel-2 L2A imagery via the Copernicus Data Space API
- Integrates with EuroCrops / Sen4AgriNet labeled datasets for supervised training
- Multi-class segmentation (field boundaries, crop types, non-agricultural land)
- Handles multi-spectral input (13 bands including NIR, SWIR, Red Edge)
- Inference pipeline that accepts a bounding box and returns a classified GeoTIFF

## Tech Stack

- `torchgeo` — geospatial datasets, samplers, transforms
- `segmentation-models-pytorch` — U-Net with pretrained encoder backbones
- `rasterio` / `geopandas` — raster and vector I/O
- `pytorch-lightning` — training loop
- `mlflow` — experiment tracking

## Project Structure

```
sentinel-crop-segmentation/
├── configs/            # Hydra/YAML experiment configs
├── data/
│   ├── raw/            # Downloaded imagery and labels (gitignored)
│   └── processed/      # Tiled, normalized patches
├── notebooks/          # Exploratory analysis and visualizations
├── scripts/            # Data download and preprocessing scripts
├── src/
│   ├── data/           # Dataset classes, transforms, samplers
│   ├── models/         # Model definitions and wrappers
│   └── utils/          # Metrics, visualization, geo utilities
└── tests/              # Unit tests
```

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Download a sample area (requires Copernicus account)
python scripts/download_sentinel.py --bbox 12.3,41.8,12.6,42.1 --start 2023-06-01 --end 2023-08-31

# Train
python src/train.py experiment=baseline

# Run inference on a bounding box
python scripts/infer.py --bbox 12.3,41.8,12.6,42.1 --checkpoint checkpoints/best.ckpt --output out.tif
```

## Data Sources

| Source | Description | License |
|--------|-------------|---------|
| [Copernicus Data Space](https://dataspace.copernicus.eu/) | Sentinel-2 L2A imagery | Free / Open |
| [EuroCrops](https://github.com/maja601/EuroCrops) | Harmonized crop type labels (EU) | CC-BY |
| [Sen4AgriNet](https://www.sen4agrinet.space/) | Multi-temporal Sentinel-2 + labels | CC-BY |

## Results

> Training in progress — results will be updated here.

| Model | Backbone | mIoU | F1 |
|-------|----------|------|----|
| U-Net | ResNet-34 | — | — |
| U-Net++ | EfficientNet-B4 | — | — |

## Setup

See [SETUP.md](SETUP.md) for full environment setup including Copernicus API credentials.
