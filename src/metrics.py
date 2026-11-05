"""
metrics.py
----------
Evaluation metrics for binary hair segmentation.

Metrics computed:
  - Dice Score (F1)
  - IoU (Jaccard Index)
  - Precision
  - Recall

All metrics operate on binary tensors (0 or 1).
Sigmoid + threshold is applied inside where needed.
"""

import torch


def _prepare(logits: torch.Tensor, targets: torch.Tensor,
             threshold: float = 0.5):
    """
    Converts raw logits to binary predictions and flattens both tensors.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(-1)
    targets = targets.view(-1)
    return preds, targets


def dice_score(logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Dice Score = 2 * TP / (2 * TP + FP + FN)
    Range: 0 (worst) → 1 (perfect)
    """
    preds, targets = _prepare(logits, targets, threshold)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    return ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    IoU = TP / (TP + FP + FN)
    Range: 0 (worst) → 1 (perfect)
    """
    preds, targets = _prepare(logits, targets, threshold)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    return ((tp + smooth) / (tp + fp + fn + smooth)).item()


def precision_score(logits: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Precision = TP / (TP + FP)
    Measures: of pixels predicted as hair, how many are actually hair?
    """
    preds, targets = _prepare(logits, targets, threshold)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    return ((tp + smooth) / (tp + fp + smooth)).item()


def recall_score(logits: torch.Tensor, targets: torch.Tensor,
                 threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Recall = TP / (TP + FN)
    Measures: of actual hair pixels, how many did we correctly find?
    """
    preds, targets = _prepare(logits, targets, threshold)
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


def compute_all_metrics(logits: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5) -> dict:
    """
    Compute all 4 metrics at once.
    Returns a dict with keys: dice, iou, precision, recall
    """
    preds, tgts = _prepare(logits, targets, threshold)
    smooth = 1e-6
    tp = (preds * tgts).sum()
    fp = (preds * (1 - tgts)).sum()
    fn = ((1 - preds) * tgts).sum()

    dice      = ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item()
    iou       = ((tp + smooth) / (tp + fp + fn + smooth)).item()
    precision = ((tp + smooth) / (tp + fp + smooth)).item()
    recall    = ((tp + smooth) / (tp + fn + smooth)).item()

    return {
        "dice":      dice,
        "iou":       iou,
        "precision": precision,
        "recall":    recall,
    }
