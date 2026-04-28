"""Torch datasets for training Model A (CNN on log-mel)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import CLIP_SAMPLES, logmel, random_crop
from .config import (
    CACHE_DIR,
    DEFAULTS,
    N_MELS,
    SPEC_FRAMES,
    TRAIN_SOUNDSCAPES_DIR,
)
from .taxonomy import class_to_idx, num_classes

MEL_DIR = CACHE_DIR / "mel"


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
def encode_labels(
    primary: str | Sequence[str],
    secondary: str | Sequence[str] | None,
    weight_group: str,
    label_weights=DEFAULTS.label_weights,
) -> np.ndarray:
    """Return a length-`num_classes()` float vector of target probabilities.

    weight_group in {"soundscape","primary"}:
      - soundscape: primary labels -> 1.0
      - primary:    primary labels -> label_weights.primary
      - secondary labels always -> label_weights.secondary (unless already set higher)
    """
    c2i = class_to_idx()
    y = np.zeros(num_classes(), dtype=np.float32)

    def _split(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        if isinstance(v, str):
            return [t for t in v.replace(",", ";").split(";") if t]
        return list(v)

    prim_val = (
        1.0 if weight_group == "soundscape" else float(label_weights.primary)
    )
    for code in _split(primary):
        if code in c2i:
            y[c2i[code]] = max(y[c2i[code]], prim_val)
    for code in _split(secondary):
        if code in c2i:
            y[c2i[code]] = max(y[c2i[code]], float(label_weights.secondary))
    return y


# ---------------------------------------------------------------------------
# Spec augmentations
# ---------------------------------------------------------------------------
@dataclass
class SpecAugCfg:
    time_masks: int = 2
    time_mask_max: int = 40
    freq_masks: int = 2
    freq_mask_max: int = 16
    time_shift_max: int = 20


def spec_augment(mel: np.ndarray, cfg: SpecAugCfg, rng: np.random.Generator) -> np.ndarray:
    out = mel.copy()
    n_mels, n_frames = out.shape
    # time shift
    if cfg.time_shift_max > 0:
        s = int(rng.integers(-cfg.time_shift_max, cfg.time_shift_max + 1))
        if s != 0:
            out = np.roll(out, s, axis=1)
    # time masks
    for _ in range(cfg.time_masks):
        t = int(rng.integers(0, cfg.time_mask_max + 1))
        if t == 0:
            continue
        t0 = int(rng.integers(0, max(1, n_frames - t)))
        out[:, t0 : t0 + t] = out.mean()
    # freq masks
    for _ in range(cfg.freq_masks):
        f = int(rng.integers(0, cfg.freq_mask_max + 1))
        if f == 0:
            continue
        f0 = int(rng.integers(0, max(1, n_mels - f)))
        out[f0 : f0 + f, :] = out.mean()
    return out


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------
class MelCacheDataset(Dataset):
    """Reads pre-computed mel `.npy` shards produced by `data_prep.build_mel_cache`.

    For training: applies SpecAugment (+ optional mixup via the training loop),
    + optional background mixing with unlabeled soundscapes (waveform domain is
    not available here; we mix in mel domain with randomly sampled soundscape
    mels flagged source == 'unlabeled_ss' — produced separately).
    """

    def __init__(
        self,
        index: pd.DataFrame,
        train: bool,
        spec_aug: SpecAugCfg | None = None,
        bg_index: pd.DataFrame | None = None,
        bg_mix_prob: float = DEFAULTS.bg_mix_prob,
        seed: int = 0,
    ):
        self.df = index.reset_index(drop=True)
        self.train = train
        self.spec_aug = spec_aug
        self.bg = bg_index.reset_index(drop=True) if bg_index is not None else None
        self.bg_mix_prob = bg_mix_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, sample_id: str) -> np.ndarray:
        arr = np.load(MEL_DIR / f"{sample_id}.npy").astype(np.float32)
        # standardize shape -> (N_MELS, SPEC_FRAMES)
        if arr.shape[1] < SPEC_FRAMES:
            pad = SPEC_FRAMES - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)))
        elif arr.shape[1] > SPEC_FRAMES:
            arr = arr[:, :SPEC_FRAMES]
        # per-image normalize
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        return arr

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mel = self._load_mel(row["sample_id"])

        if self.train and self.bg is not None and self.rng.random() < self.bg_mix_prob:
            bg_row = self.bg.iloc[int(self.rng.integers(0, len(self.bg)))]
            bg_mel = self._load_mel(bg_row["sample_id"])
            alpha = float(self.rng.uniform(0.2, 0.5))
            mel = (1 - alpha) * mel + alpha * bg_mel

        if self.train and self.spec_aug is not None:
            mel = spec_augment(mel, self.spec_aug, self.rng)

        y = encode_labels(row["labels"], row.get("secondary_labels", ""), row["weight_group"])
        sample_weight = 1.0  # reserved; per-sample weight can be scaled by rating, etc.

        return {
            "mel": torch.from_numpy(mel[None, :, :]).float(),  # (1, N_MELS, T)
            "target": torch.from_numpy(y).float(),
            "weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_train_val_datasets(
    val_fold: int,
    spec_aug: SpecAugCfg | None = None,
    bg_mix_prob: float = DEFAULTS.bg_mix_prob,
):
    idx_path = CACHE_DIR / "index.parquet"
    index = pd.read_parquet(idx_path)

    # val = labeled soundscapes in the held-out fold
    val_mask = (index["source"] == "soundscape") & (index["fold"] == val_fold)
    train_mask = ~val_mask
    train_df = index[train_mask].reset_index(drop=True)
    val_df = index[val_mask].reset_index(drop=True)

    # bg pool = labeled soundscapes in training folds (serve as "real-world" spectra).
    bg_df = index[(index["source"] == "soundscape") & (index["fold"] != val_fold)]
    bg_df = bg_df.reset_index(drop=True) if len(bg_df) else None

    train_ds = MelCacheDataset(train_df, train=True, spec_aug=spec_aug, bg_index=bg_df, bg_mix_prob=bg_mix_prob)
    val_ds = MelCacheDataset(val_df, train=False)
    return train_ds, val_ds


__all__ = [
    "MelCacheDataset",
    "SpecAugCfg",
    "encode_labels",
    "build_train_val_datasets",
]
