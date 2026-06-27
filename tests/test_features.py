"""Unit tests for feature extraction and model utilities."""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
import geopandas as gpd

from src.data.sentinel import ndvi, evi, ndre, ndwi
from src.data.corine import _random_points_in_polygons
from src.models.suitability import train, evaluate, predict_raster
from src.utils.geo import bbox_to_polygon


# --- Spectral index tests ---

def test_ndvi_range():
    nir = np.array([0.5, 0.8, 0.2])
    red = np.array([0.1, 0.4, 0.2])
    result = ndvi(nir, red)
    assert np.all(result >= -1) and np.all(result <= 1)


def test_ndvi_zero_division():
    nir = np.array([0.0])
    red = np.array([0.0])
    assert ndvi(nir, red)[0] == pytest.approx(0.0)


def test_evi_shape():
    nir = np.ones((10, 10)) * 0.5
    red = np.ones((10, 10)) * 0.1
    blue = np.ones((10, 10)) * 0.05
    assert evi(nir, red, blue).shape == (10, 10)


# --- Geo utility tests ---

def test_bbox_to_polygon():
    bbox = {"north": 37.5, "south": 37.35, "east": 24.48, "west": 24.38}
    poly = bbox_to_polygon(bbox)
    assert poly.bounds == (24.38, 37.35, 24.48, 37.5)


# --- Model tests ---

def _make_dummy_data(n: int = 200, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_train_predict_shape():
    X, y = _make_dummy_data()
    model = train(X, y, n_estimators=10)
    scores = predict_raster(model, X.values.T.reshape(X.shape[1], 10, 20))
    assert scores.shape == (10, 20)
    assert scores.dtype == np.float32


def test_evaluate_keys():
    X, y = _make_dummy_data()
    model = train(X, y, n_estimators=10)
    metrics = evaluate(model, X, y)
    assert "auc_roc" in metrics and "brier_score" in metrics
    assert 0 <= metrics["auc_roc"] <= 1
