"""
Evaluation metrics for medical image classification.

Supports:
    - Multi-label AUC (per-class + macro average)
    - Multi-label F1 score
    - Multi-class accuracy, precision, recall
    - Calibration: Expected Calibration Error (ECE)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def compute_auc(
    labels: torch.Tensor,
    probs: torch.Tensor,
    average: str = "macro",
) -> float:
    """
    Compute ROC-AUC for multi-label classification.

    Args:
        labels:  (N, C) binary ground truth.
        probs:   (N, C) predicted probabilities in [0, 1].
        average: 'macro' | 'micro' | 'weighted'.

    Returns:
        Scalar AUC score.
    """
    y_true = labels.numpy()
    y_score = probs.numpy()

    # Skip classes with only one unique label value (AUC undefined)
    valid_cols = [c for c in range(y_true.shape[1]) if len(np.unique(y_true[:, c])) > 1]
    if not valid_cols:
        return float("nan")

    return float(roc_auc_score(y_true[:, valid_cols], y_score[:, valid_cols], average=average))


def compute_f1(
    labels: torch.Tensor,
    probs: torch.Tensor,
    threshold: float = 0.5,
    average: str = "macro",
) -> float:
    """
    Compute F1 score for multi-label classification.

    Args:
        labels:    (N, C) binary ground truth.
        probs:     (N, C) predicted probabilities.
        threshold: Binarisation threshold.
        average:   'macro' | 'micro' | 'samples'.

    Returns:
        Scalar F1 score.
    """
    y_true = labels.numpy().astype(int)
    y_pred = (probs.numpy() >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) for a single-class probability.

    Args:
        probs:  (N,) predicted probabilities.
        labels: (N,) binary labels.
        n_bins: Number of calibration bins.

    Returns:
        ECE value in [0, 1].
    """
    p = probs.numpy()
    y = labels.numpy()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += mask.mean() * abs(acc - conf)

    return float(ece)
