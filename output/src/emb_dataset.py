"""In-memory dataset for Model B: cached external embeddings -> MLP head."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CACHE_DIR, DEFAULTS
from .datasets import encode_labels

EMB_DIR = CACHE_DIR / "emb"
EMB_INDEX = CACHE_DIR / "emb_index.parquet"


class EmbeddingDataset(Dataset):
    def __init__(self, index: pd.DataFrame, label_weights=DEFAULTS.label_weights):
        self.df = index.reset_index(drop=True)
        self.label_weights = label_weights

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(EMB_DIR / f"{row['sample_id']}.npy").astype(np.float32)
        y = encode_labels(row["labels"], row.get("secondary_labels", ""), row["weight_group"], self.label_weights)
        return {
            "x": torch.from_numpy(x),
            "target": torch.from_numpy(y).float(),
            "weight": torch.tensor(1.0, dtype=torch.float32),
        }


def build_emb_datasets(val_fold: int):
    df = pd.read_parquet(EMB_INDEX)
    val_mask = (df["source"] == "soundscape") & (df["fold"] == val_fold)
    train_ds = EmbeddingDataset(df[~val_mask])
    val_ds = EmbeddingDataset(df[val_mask])
    return train_ds, val_ds


def embedding_dim() -> int:
    df = pd.read_parquet(EMB_INDEX)
    sid = df.iloc[0]["sample_id"]
    return int(np.load(EMB_DIR / f"{sid}.npy").shape[0])
