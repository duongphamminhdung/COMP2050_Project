# [OWN WORK] Evaluation metrics for binary image segmentation
# Implements: Dice, IoU, pixel accuracy, sensitivity, specificity, HD95

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _to_numpy(pred, target, threshold=0.5):
    """Convert tensors to numpy arrays and binarize predictions."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > threshold).astype(np.float32)
    target = (target > threshold).astype(np.float32)
    return pred, target


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-7):
    """Compute Dice coefficient (F1 score) for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    pred, target = _to_numpy(pred, target, threshold)
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou(pred, target, threshold=0.5, smooth=1e-7):
    """Compute Intersection over Union (Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|
    """
    pred, target = _to_numpy(pred, target, threshold)
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred, target, threshold=0.5):
    """Compute pixel-wise accuracy."""
    pred, target = _to_numpy(pred, target, threshold)
    correct = (pred == target).sum()
    total = pred.size
    return correct / total


def sensitivity(pred, target, threshold=0.5, smooth=1e-7):
    """Compute sensitivity (recall, true positive rate).

    Sensitivity = TP / (TP + FN)
    """
    pred, target = _to_numpy(pred, target, threshold)
    pred = pred.flatten()
    target = target.flatten()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred, target, threshold=0.5, smooth=1e-7):
    """Compute specificity (true negative rate).

    Specificity = TN / (TN + FP)
    """
    pred, target = _to_numpy(pred, target, threshold)
    pred = pred.flatten()
    target = target.flatten()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def hausdorff_distance_95(pred, target, threshold=0.5):
    """Compute 95th percentile Hausdorff distance.

    Measures the maximum boundary distance at the 95th percentile,
    making it robust to outlier boundary points.

    Returns 0.0 if either prediction or target is empty.
    """
    pred, target = _to_numpy(pred, target, threshold)

    # Handle batch dimension: process each sample separately
    if pred.ndim == 4:  # (B, C, H, W)
        distances = []
        for i in range(pred.shape[0]):
            d = hausdorff_distance_95(pred[i], target[i], threshold=0.5)
            distances.append(d)
        return np.mean(distances)

    # Squeeze channel dimension if present
    pred = pred.squeeze()
    target = target.squeeze()

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return np.inf

    # Compute distance from each boundary point to the nearest boundary in the other mask
    pred_border = pred - np.maximum(0, pred - _erode(pred))
    target_border = target - np.maximum(0, target - _erode(target))

    if pred_border.sum() == 0:
        pred_border = pred
    if target_border.sum() == 0:
        target_border = target

    # Distance from pred border to target, and vice versa
    dt_target = distance_transform_edt(1 - target_border)
    dt_pred = distance_transform_edt(1 - pred_border)

    d_pred_to_target = dt_target[pred_border > 0]
    d_target_to_pred = dt_pred[target_border > 0]

    if len(d_pred_to_target) == 0 and len(d_target_to_pred) == 0:
        return 0.0

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return np.percentile(all_distances, 95)


def _erode(mask):
    """Simple morphological erosion using distance transform."""
    if mask.sum() == 0:
        return mask
    dt = distance_transform_edt(mask)
    return (dt > 1).astype(mask.dtype)


def compute_all_metrics(pred, target, threshold=0.5):
    """Compute all segmentation metrics at once.

    Args:
        pred: Predicted logits or probabilities, shape (B, 1, H, W) or (B, H, W).
        target: Ground truth mask, shape (B, 1, H, W) or (B, H, W).
        threshold: Binarization threshold for predictions.

    Returns:
        Dictionary with keys: dice, iou, accuracy, sensitivity, specificity, hd95.
    """
    return {
        "dice": dice_coefficient(pred, target, threshold),
        "iou": iou(pred, target, threshold),
        "accuracy": pixel_accuracy(pred, target, threshold),
        "sensitivity": sensitivity(pred, target, threshold),
        "specificity": specificity(pred, target, threshold),
        "hd95": hausdorff_distance_95(pred, target, threshold),
    }
