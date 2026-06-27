"""Map and chart helpers for suitability analysis outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_suitability_map(
    tif_path: Path,
    threshold: float = 0.5,
    title: str = "Olive Suitability",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a suitability GeoTIFF with a threshold overlay.

    Highlights pixels above threshold in a distinct colour.
    """
    import rasterio

    with rasterio.open(tif_path) as src:
        scores = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, label in [
        (axes[0], scores, "Suitability score (0–1)"),
        (axes[1], (scores >= threshold).astype(np.float32), f"Score ≥ {threshold}"),
    ]:
        im = ax.imshow(data, extent=extent, origin="upper", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"{title}\n{label}")
        fig.colorbar(im, ax=ax, fraction=0.03)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Top {top_n} feature importances")
    plt.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Beeswarm SHAP summary plot."""
    import shap

    fig, _ = plt.subplots(figsize=figsize)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    return fig


def plot_pdp(
    model,
    X: "pd.DataFrame",
    features: list[str],
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Partial dependence plots for a list of feature names."""
    from sklearn.inspection import PartialDependenceDisplay

    fig, axes = plt.subplots(1, len(features), figsize=figsize)
    if len(features) == 1:
        axes = [axes]
    PartialDependenceDisplay.from_estimator(model, X, features, ax=axes)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CORINE / geographic distribution plots
# ---------------------------------------------------------------------------

_COUNTRY_COLOURS = {
    "GRC": "#2D6A1F",
    "ITA": "#E07B39",
    "TUR": "#C94040",
    "ESP": "#4A90D9",
    "PRT": "#8B5CF6",
}

_COUNTRY_NAMES = {
    "GRC": "Greece",
    "ITA": "Italy",
    "TUR": "Turkey",
    "ESP": "Spain",
    "PRT": "Portugal",
}


def plot_corine_olive_distribution(
    olive_polygons: "gpd.GeoDataFrame",
    sample_points: "gpd.GeoDataFrame | None" = None,
    country_col: str | None = None,
    basemap_source: str = "CartoDB.Positron",
    figsize: tuple[int, int] = (12, 10),
    zoom: int | str = 8,
    dpi: int = 150,
) -> plt.Figure:
    """Plot CORINE class-223 olive polygons on a contextily basemap.

    Args:
        olive_polygons: GeoDataFrame of olive grove polygons (any CRS).
        sample_points: optional presence/absence points with a "label" column.
        country_col: column in olive_polygons to colour polygons by country;
            if None all polygons are drawn in a single olive green.
        basemap_source: contextily provider string, e.g. "CartoDB.Positron".
        figsize: figure size in inches.
    """
    import contextily as cx
    import geopandas as gpd

    web = olive_polygons.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if country_col and country_col in web.columns:
        for iso3, group in web.groupby(country_col):
            colour = _COUNTRY_COLOURS.get(str(iso3), "#7CBA5A")
            group.plot(ax=ax, color=colour, edgecolor="none", alpha=0.6,
                       label=_COUNTRY_NAMES.get(str(iso3), str(iso3)))
        ax.legend(loc="upper right", title="Country")
    else:
        web.plot(ax=ax, color="#7CBA5A", edgecolor="none", alpha=0.6,
                 label="Olive groves (CORINE 223)")
        ax.legend(loc="upper right")

    if sample_points is not None:
        pts = sample_points.to_crs("EPSG:3857")
        pres = pts[pts["label"] == 1]
        abse = pts[pts["label"] == 0]
        if not pres.empty:
            pres.plot(ax=ax, color="#2D6A1F", markersize=3, alpha=0.5,
                      marker="o", label=f"Presence (n={len(pres):,})")
        if not abse.empty:
            abse.plot(ax=ax, color="#CC4444", markersize=3, alpha=0.5,
                      marker="x", label=f"Absence (n={len(abse):,})")
        ax.legend(loc="upper right")

    # Resolve dotted provider string, e.g. "CartoDB.Positron"
    provider = cx.providers
    for part in basemap_source.split("."):
        provider = provider[part]
    cx.add_basemap(ax, source=provider, zoom=zoom)

    ax.set_title("CORINE Olive Grove Distribution (Class 223)")
    ax.set_xlabel("Easting (EPSG:3857)")
    ax.set_ylabel("Northing (EPSG:3857)")
    plt.tight_layout()
    return fig


def plot_mediterranean_olive_comparison(
    olive_by_country: "dict[str, gpd.GeoDataFrame]",
    lat_lines: list[float] | None = None,
    figsize: tuple[int, int] = (18, 10),
) -> plt.Figure:
    """Two-panel Mediterranean comparison of olive grove distribution by country.

    Left panel: basemap with all five countries' olive polygons and latitude
    reference lines. Right panel: latitude histogram of polygon centroids per
    country — shows where in the 30–47 °N band each country's olives concentrate.

    Args:
        olive_by_country: mapping of ISO-3 code → GeoDataFrame of olive polygons.
            Expected keys: "GRC", "ITA", "TUR", "ESP", "PRT".
        lat_lines: decimal-degree latitudes for reference lines; defaults to
            [35.0, 37.5, 40.0, 42.5].
        figsize: figure size in inches.
    """
    import contextily as cx
    import geopandas as gpd
    import math

    if lat_lines is None:
        lat_lines = [35.0, 37.5, 40.0, 42.5]

    # Web mercator northing for reference lines (approximate)
    def _lat_to_northing(lat_deg: float) -> float:
        lat_rad = math.radians(lat_deg)
        return 6_378_137 * math.log(math.tan(math.pi / 4 + lat_rad / 2))

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
    )

    # --- Left panel: map ---
    legend_labels = []
    all_centroids: dict[str, list[float]] = {}

    for iso3, gdf in olive_by_country.items():
        colour = _COUNTRY_COLOURS.get(iso3, "#888888")
        name = _COUNTRY_NAMES.get(iso3, iso3)
        web = gdf.to_crs("EPSG:3857")
        web.plot(ax=ax_map, color=colour, edgecolor="none", alpha=0.55)

        area_km2 = gdf.to_crs("EPSG:3857").geometry.area.sum() / 1e6
        legend_labels.append(
            plt.Rectangle((0, 0), 1, 1, fc=colour, alpha=0.7,
                           label=f"{name}  ({area_km2:,.0f} km²)")
        )

        # Centroids in WGS-84 for latitude histogram
        wgs = gdf.to_crs("EPSG:4326")
        all_centroids[iso3] = wgs.geometry.centroid.y.tolist()

    # Latitude reference lines
    for lat in lat_lines:
        northing = _lat_to_northing(lat)
        ax_map.axhline(northing, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_map.text(
            ax_map.get_xlim()[0] if ax_map.get_xlim()[0] != 0 else -1_200_000,
            northing + 15_000,
            f"{lat}°N",
            fontsize=8, color="black", alpha=0.7,
        )

    cx.add_basemap(ax_map, source=cx.providers.CartoDB.Positron, zoom="auto")
    ax_map.legend(handles=legend_labels, loc="lower left", title="Country (CORINE olive area)")
    ax_map.set_title("Mediterranean Olive Grove Distribution")
    ax_map.set_xlabel("Easting (EPSG:3857)")
    ax_map.set_ylabel("Northing (EPSG:3857)")

    # --- Right panel: latitude histogram ---
    bins = np.arange(30, 47, 0.5)
    for iso3, lats in all_centroids.items():
        if not lats:
            continue
        colour = _COUNTRY_COLOURS.get(iso3, "#888888")
        name = _COUNTRY_NAMES.get(iso3, iso3)
        ax_hist.hist(
            lats, bins=bins, orientation="horizontal",
            color=colour, alpha=0.55, label=name, density=True,
        )

    for lat in lat_lines:
        ax_hist.axhline(lat, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax_hist.set_xlabel("Density")
    ax_hist.set_ylabel("Latitude (°N)")
    ax_hist.set_ylim(30, 47)
    ax_hist.set_title("Latitudinal spread")
    ax_hist.legend(fontsize=8)
    ax_hist.yaxis.set_label_position("right")
    ax_hist.yaxis.tick_right()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Terrain visualisation
# ---------------------------------------------------------------------------

def plot_terrain_panels(
    features: "dict[str, np.ndarray]",
    transform,
    figsize: tuple[int, int] = (16, 12),
    dpi: int = 150,
    title: str = "",
    points_gdf=None,
) -> plt.Figure:
    """4-panel terrain overview: hillshade, slope, northness, TPI.

    Args:
        features: dict from compute_terrain_features() — keys: elevation, slope, northness, tpi.
        transform: rasterio Affine transform (second return value of compute_terrain_features).
        figsize: figure size in inches.
        dpi: output resolution.
        title: optional suptitle.
        points_gdf: optional GeoDataFrame of points (WGS-84) to overlay on each panel.
            If it has a "label" column, presence (1) = green dots, absence (0) = red crosses.
    """
    import rasterio.transform as rt

    elev = features["elevation"]
    h, w = elev.shape
    left, top = rt.xy(transform, 0, 0)
    right, bottom = rt.xy(transform, h, w)
    extent = [left, right, bottom, top]

    ls = mcolors.LightSource(azdeg=315, altdeg=35)
    hillshade = ls.hillshade(np.nan_to_num(elev), vert_exag=4, dx=30, dy=30)

    panels = [
        (hillshade,            "Hillshade (elevation)",  "gray",    None, None),
        (features["slope"],    "Slope (°)",               "YlOrRd",  0,    45),
        (features["northness"],"Northness (cos aspect)",  "RdBu",   -1,    1),
        (features["tpi"],      "TPI — ridges / valleys",  "BrBG_r",  None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    for ax, (data, label, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(data, extent=extent, origin="upper",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)

        if points_gdf is not None:
            pts = points_gdf.to_crs("EPSG:4326") if points_gdf.crs.to_epsg() != 4326 else points_gdf
            if "label" in pts.columns:
                for lbl, colour, marker in [(1, "#2D6A1F", "o"), (0, "#CC4444", "x")]:
                    sub = pts[pts["label"] == lbl]
                    if not sub.empty:
                        ax.scatter(sub.geometry.x, sub.geometry.y,
                                   c=colour, s=4, alpha=0.5, marker=marker, linewidths=0.5)
            else:
                ax.scatter(pts.geometry.x, pts.geometry.y, c="#2D6A1F", s=4, alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    return fig
