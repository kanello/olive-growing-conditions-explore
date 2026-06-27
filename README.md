# Project Plan: Olive Grove Suitability Mapping in Greece

## Research Question

> Can I successfully grow olives on the island of Kythnos? If yes, where on the island should I look for land? If not, where in Greece is the nearest suitable environment?

---

## Overview

This project uses satellite imagery, climate data, terrain analysis, and soil data to build an olive growing suitability model for Greece. The model learns environmental conditions from locations where olives are already successfully cultivated, then applies that knowledge to map suitability across a target area — starting with the island of Kythnos.

The approach mirrors **species distribution modeling (SDM)**, a method used in ecology to predict habitat suitability for a species based on known presence locations and environmental variables.

---

## Phase 1 — Agricultural Research (Non-coding)

Before building anything, establish what olive trees actually need. This research feeds directly into feature selection in Phase 3.

Key variables to investigate:

- **Temperature**: chilling hour requirements, frost tolerance thresholds (critical minimum ~-10°C), summer heat tolerance
- **Water**: annual precipitation range, summer drought tolerance, irrigation requirements
- **Soil**: preferred pH range, drainage requirements, texture preferences, tolerance for rocky/calcareous soils
- **Terrain**: elevation limits, slope preferences, aspect (sun exposure)
- **Microclimate**: sea proximity effects, wind exposure

Sources to consult:

- FAO Ecocrop database (olive entry)
- UC Davis Olive Center publications
- Greek Ministry of Rural Development cultivar guides
- Academic literature on _Olea europaea_ distribution modeling

**Deliverable**: A `research/olive_growing_conditions.md` document summarising each variable with cited thresholds. This becomes the ground truth for evaluating whether the model's learned features make agronomic sense.

---

## Phase 2 — Ground Truth Data: Where Do Olives Currently Grow?

Rather than training a detector from scratch, use existing authoritative land cover datasets as positive sample locations.

**Primary source**: [CORINE Land Cover](https://land.copernicus.eu/pan-european/corine-land-cover) (class 223 — Olive groves), covering all of Greece at 100m resolution. Freely available from Copernicus Land Service.

**Secondary source**: OpenStreetMap `landuse=orchard` + `crop=olive` polygons for higher-resolution spot checks.

Tasks:

- [ ] Download CORINE Land Cover for Greece (2018 or 2022 edition)
- [ ] Extract class 223 polygons
- [ ] Sample N positive locations (olive) and N pseudo-absence locations (non-olive agricultural land) across Greece
- [ ] Validate a random subset against Google Earth imagery

**Output**: `data/samples/olive_presence_absence.geojson` — a point dataset with binary labels used for model training.

---

## Phase 3 — Feature Extraction

For each sample point, extract a feature vector from multiple environmental data sources. This is the core data engineering work.

### 3a. Satellite Spectral Features (Sentinel-2)

Source: [Copernicus Data Space](https://dataspace.copernicus.eu/) — free, requires account registration.

Extract cloud-free composites for two seasons (summer peak / winter low) and compute:

- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- NDRE (Red Edge NDVI) — particularly sensitive to evergreen olive canopies
- NDWI (Water Index) — proxy for moisture stress

Key implementation note: Sentinel-2's Red Edge bands (B05, B06, B07) are one of the strongest discriminators for olive groves vs. other crops due to the silver-grey leaf surface.

### 3b. Climate Features

Source: [CHELSA](https://chelsa-climate.org/) or [ERA5-Land](https://cds.climate.copernicus.eu/) via CDS API.

Variables:

- Mean annual temperature
- Mean minimum temperature of the coldest month (frost risk)
- Mean maximum temperature of the warmest month (heat stress)
- Annual precipitation
- Summer precipitation (June–August) — proxy for drought stress
- Number of frost days per year

### 3c. Terrain Features

Source: [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) (30m resolution, free).

Variables:

- Elevation (m)
- Slope (degrees)
- Aspect (degrees, converted to northness/eastness components)
- Topographic Wetness Index (TWI) — proxy for soil moisture and drainage

### 3d. Soil Features

Source: [SoilGrids 250m](https://soilgrids.org/) via REST API.

Variables:

- Soil pH (0–30cm depth)
- Clay content (%)
- Sand content (%) — proxy for drainage
- Organic carbon content
- Bulk density

**Output**: `data/features/feature_matrix.parquet` — one row per sample, ~25 feature columns + label.

---

## Phase 4 — Suitability Model

Train a classifier on the feature matrix to distinguish olive-suitable from non-suitable conditions. Use predicted probability as the suitability score.

### Model choice

**Random Forest** (primary) — interpretable, handles mixed feature types, provides feature importances that can be validated against Phase 1 agronomic knowledge.

**XGBoost** (comparison) — typically higher accuracy, useful to benchmark against RF.

### Validation

- Spatial cross-validation (not random split — nearby points are correlated, so standard k-fold overestimates performance)
- Evaluate with AUC-ROC and Brier score
- Sanity check: feature importance ranking should roughly match known agronomic importance from Phase 1 research

### Interpretability

- SHAP values to explain individual predictions
- Partial dependence plots for top features (e.g., "suitability vs. annual precipitation")

**Output**: `models/suitability_rf.pkl` + evaluation report in `notebooks/04_model_evaluation.ipynb`

---

## Phase 5 — Inference: Mapping Kythnos

Apply the trained model across every pixel in the Kythnos bounding box.

Steps:

- [ ] Clip all feature layers to Kythnos extent (~37.35–37.50°N, 24.38–24.48°E)
- [ ] Build a raster feature stack (align all layers to same grid/CRS)
- [ ] Run model inference pixel-by-pixel
- [ ] Output a suitability GeoTIFF (float32, 0–1 score)
- [ ] Visualize as an interactive map overlaid on terrain/satellite basemap

Visualization targets:

- Suitability heatmap over Kythnos
- Top 10% most suitable areas highlighted as candidate zones
- Comparison panel: Kythnos suitability distribution vs. known olive-growing regions (e.g., Messinia, Crete)

**Output**: `outputs/kythnos_suitability.tif` + `notebooks/05_kythnos_map.ipynb`

---

## Repo Structure

```
sentinel-olive-suitability/
├── research/
│   └── olive_growing_conditions.md   # Phase 1 notes and citations
├── data/
│   ├── raw/                          # Downloaded source files (gitignored)
│   ├── samples/                      # Presence/absence points
│   └── features/                     # Extracted feature matrix
├── notebooks/
│   ├── 01_eda_corine.ipynb           # Explore CORINE olive grove distribution
│   ├── 02_feature_extraction.ipynb   # Extract and validate features
│   ├── 03_model_training.ipynb       # Train and tune classifier
│   ├── 04_model_evaluation.ipynb     # Spatial CV, SHAP, PDP plots
│   └── 05_kythnos_map.ipynb          # Final suitability map
├── src/
│   ├── data/
│   │   ├── corine.py                 # CORINE download + sampling
│   │   ├── sentinel.py               # Sentinel-2 composite builder
│   │   ├── climate.py                # ERA5/CHELSA extraction
│   │   ├── terrain.py                # DEM processing
│   │   └── soil.py                   # SoilGrids API client
│   ├── features/
│   │   └── extractor.py              # Unified feature extraction pipeline
│   ├── models/
│   │   └── suitability.py            # Model training + inference
│   └── utils/
│       ├── geo.py                    # Raster/vector utilities
│       └── visualization.py          # Map plotting helpers
├── outputs/                          # Final maps and reports (gitignored for large files)
├── tests/
├── configs/
│   └── kythnos.yaml                  # Inference config for target area
└── pyproject.toml
```

---

## Data Sources Summary

| Data                  | Source                        | Resolution | License           |
| --------------------- | ----------------------------- | ---------- | ----------------- |
| Olive grove locations | CORINE Land Cover (class 223) | 100m       | Free / Copernicus |
| Satellite imagery     | Sentinel-2 L2A                | 10–20m     | Free / Copernicus |
| Climate               | ERA5-Land or CHELSA           | ~1km       | Free / CC-BY      |
| Terrain               | Copernicus DEM GLO-30         | 30m        | Free / Copernicus |
| Soil                  | SoilGrids 250m                | 250m       | Free / CC-BY      |

All data sources are freely available. No API keys required except a free Copernicus Data Space account.

---

## Success Criteria

The project is complete when it can answer, with a map and quantitative reasoning:

1. What percentage of Kythnos pixels score above the suitability threshold used for known olive regions?
2. Which areas of the island score highest, and what drives that score (terrain, climate, soil)?
3. If Kythnos scores poorly overall, which nearby Cycladic islands score comparably to known olive-producing regions?
