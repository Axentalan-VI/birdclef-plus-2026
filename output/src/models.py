"""Model definitions: timm CNN backbone (Model A) + embedding MLP head (Model B)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TimmMelClassifier(nn.Module):
    """timm backbone on 1-channel log-mel input -> num_classes logits."""

    def __init__(self, backbone: str = "tf_efficientnet_b0.ns_jft_in1k", num_classes: int = 234, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,      # get raw features
            global_pool="avg",
            drop_rate=drop_rate,
        )
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, N_MELS, T)
        f = self.backbone(x)
        return self.head(f)


class EmbeddingHead(nn.Module):
    """Model B: lightweight MLP on frozen external embeddings (Perch v2 / BirdNET)."""

    def __init__(self, emb_dim: int, num_classes: int = 234, hidden: int = 512, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, emb_dim)
        return self.net(x)


def build_model(kind: str, **kw) -> nn.Module:
    if kind == "timm":
        return TimmMelClassifier(**kw)
    if kind == "embhead":
        return EmbeddingHead(**kw)
    raise ValueError(f"unknown model kind: {kind}")
