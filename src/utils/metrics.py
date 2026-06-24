"""
Evaluation utilities for segmentation tasks.
"""

import numpy as np
import torch
from torch import Tensor


def compute_confusion_matrix(preds: Tensor, targets: Tensor, num_classes: int) -> np.ndarray:
    """Compute a flattened confusion matrix over a batch."""
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    mask = (targets >= 0) & (targets < num_classes)
    return np.bincount(
        num_classes * targets[mask].astype(int) + preds[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def iou_from_confusion(conf_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute per-class IoU and mean IoU from a confusion matrix.

    Returns:
        per_class_iou: Array of shape (num_classes,)
        miou: Scalar mean IoU (ignores NaN classes)
    """
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    per_class_iou = np.where(union > 0, intersection / union, np.nan)
    miou = float(np.nanmean(per_class_iou))
    return per_class_iou, miou
