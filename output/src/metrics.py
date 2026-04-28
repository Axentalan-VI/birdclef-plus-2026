"""Macro-averaged ROC-AUC that skips classes with no positives (competition metric)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def birdclef_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """y_true, y_pred: (N, C) arrays. Skips classes where y_true.sum() == 0."""
    assert y_true.shape == y_pred.shape
    scores = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        try:
            scores.append(roc_auc_score(yt, y_pred[:, c]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else float("nan")
