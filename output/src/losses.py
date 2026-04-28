"""Loss functions: weighted BCE + class-balanced focal BCE."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogits(nn.Module):
    """BCE with per-class pos_weight and optional per-sample weight."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight.to(logits.device)
        )
        # mean over classes, then optional sample weighting
        loss = loss.mean(dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


class FocalBCE(nn.Module):
    """Focal BCE (multi-label). alpha balances pos/neg; gamma focuses on hard examples."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        loss = loss.mean(dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def compute_pos_weight(targets_sum: torch.Tensor, n_samples: int, cap: float = 50.0) -> torch.Tensor:
    """Inverse-frequency pos_weight, capped to avoid exploding gradients on rare classes."""
    pos = targets_sum.clamp(min=1.0)
    neg = (n_samples - pos).clamp(min=1.0)
    w = neg / pos
    return w.clamp(max=cap)
