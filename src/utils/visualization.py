"""
Visualization helpers for imagery and segmentation masks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch import Tensor


# Color palette for crop classes
CLASS_COLORS = {
    0: (0.85, 0.85, 0.85),  # background - light gray
    1: (0.94, 0.90, 0.55),  # cereals - wheat yellow
    2: (0.60, 0.30, 0.10),  # root crops - brown
    3: (0.40, 0.80, 0.40),  # vegetables - green
    4: (0.95, 0.50, 0.20),  # fruits - orange
    5: (0.90, 0.85, 0.20),  # oilseeds - bright yellow
    6: (0.30, 0.65, 0.30),  # fodder - dark green
    7: (0.70, 0.70, 0.90),  # other - lavender
}

CLASS_NAMES = {
    0: "Background", 1: "Cereals", 2: "Root Crops",
    3: "Vegetables", 4: "Fruits", 5: "Oilseeds",
    6: "Fodder", 7: "Other",
}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask (H, W) to RGB image (H, W, 3)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        rgb[mask == cls_id] = color
    return rgb


def plot_prediction(
    image: Tensor | np.ndarray,
    gt_mask: Tensor | np.ndarray,
    pred_mask: Tensor | np.ndarray,
    rgb_bands: tuple[int, int, int] = (3, 2, 1),  # B04, B03, B02
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Side-by-side plot: RGB composite | ground truth mask | predicted mask.

    Args:
        image: (C, H, W) tensor or array.
        gt_mask: (H, W) ground truth labels.
        pred_mask: (H, W) predicted labels.
        rgb_bands: Band indices to use for the RGB composite.
    """
    if hasattr(image, "numpy"):
        image = image.numpy()
    if hasattr(gt_mask, "numpy"):
        gt_mask = gt_mask.numpy()
    if hasattr(pred_mask, "numpy"):
        pred_mask = pred_mask.numpy()

    # Build normalized RGB composite
    r, g, b = [image[i] for i in rgb_bands]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(gt_mask))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_to_rgb(pred_mask))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Shared legend
    patches = [
        mpatches.Patch(color=COLOR, label=CLASS_NAMES[cls_id])
        for cls_id, COLOR in CLASS_COLORS.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    return fig
