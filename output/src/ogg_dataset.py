"""On-the-fly Torch dataset that decodes ogg + computes log-mel inside the
DataLoader worker. No pre-computed cache on disk — suitable for Kaggle runtimes
where notebook output is capped at ~20 GB.

Each epoch samples one 5-sec window per focal clip (top-energy with prob p,
random crop otherwise) and reads labeled soundscape segments exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import (
    CLIP_SAMPLES,
    load_audio,
    logmel,
    random_crop,
    topk_energy_crops,
)
from .config import (
    DEFAULTS,
    SAMPLE_RATE,
    SPEC_FRAMES,
    TRAIN_AUDIO_DIR,
    TRAIN_CSV,
    TRAIN_SOUNDSCAPES_DIR,
    TRAIN_SS_LABELS_CSV,
)
from .datasets import SpecAugCfg, encode_labels, spec_augment


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
@dataclass
class SampleRow:
    path: Path
    source: str              # "focal" | "soundscape"
    labels: str              # semicolon-separated
    secondary_labels: str
    weight_group: str        # "primary" | "soundscape"
    seg_start_sec: float     # -1.0 for focal (pick crop at load time)
    rating: float
    fold: int


def build_index(folds_csv: Path | None = None) -> pd.DataFrame:
    """Returns a DataFrame with one row per training sample (focal clip or
    labeled soundscape segment) and a 'fold' column (focal -> -1)."""
    rows: List[dict] = []

    train = pd.read_csv(TRAIN_CSV)
    for r in train.to_dict("records"):
        rows.append(
            {
                "path": str(TRAIN_AUDIO_DIR / r["filename"]),
                "source": "focal",
                "labels": r["primary_label"],
                "secondary_labels": r.get("secondary_labels", "") or "",
                "weight_group": "primary",
                "seg_start_sec": -1.0,
                "rating": float(r.get("rating", 0.0) or 0.0),
                "fold": -1,
            }
        )

    ss = pd.read_csv(TRAIN_SS_LABELS_CSV)
    folds = (
        pd.read_csv(folds_csv).set_index("filename")["fold"].to_dict()
        if folds_csv is not None and folds_csv.exists() else {}
    )

    def _to_seconds(v) -> float:
        # Supports "HH:MM:SS", "MM:SS", or a numeric string/number.
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            h, m, sec = parts[-3], parts[-2], parts[-1]
            return h * 3600.0 + m * 60.0 + sec
        return float(s)

    for r in ss.to_dict("records"):
        rows.append(
            {
                "path": str(TRAIN_SOUNDSCAPES_DIR / r["filename"]),
                "source": "soundscape",
                "labels": r["primary_label"],
                "secondary_labels": "",
                "weight_group": "soundscape",
                "seg_start_sec": _to_seconds(r["start"]),
                "rating": 5.0,
                "fold": int(folds.get(r["filename"], 0)),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class OggOnTheFlyDataset(Dataset):
    """Decodes ogg + mel at __getitem__ time.

    For focal clips: with prob `top_energy_prob` the top-energy 5-sec window is
    used; otherwise a random 5-sec window. For soundscape rows the exact
    segment referenced by `seg_start_sec` is read.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train: bool,
        spec_aug: SpecAugCfg | None = None,
        top_energy_prob: float = 0.7,
        bg_pool: pd.DataFrame | None = None,
        bg_mix_prob: float = DEFAULTS.bg_mix_prob,
        seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.spec_aug = spec_aug
        self.top_energy_prob = top_energy_prob
        self.bg_pool = bg_pool.reset_index(drop=True) if bg_pool is not None else None
        self.bg_mix_prob = bg_mix_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    # -- audio IO
    def _load_clip(self, row) -> np.ndarray:
        y = load_audio(row["path"])
        if row["source"] == "soundscape":
            start = int(row["seg_start_sec"] * SAMPLE_RATE)
            seg = y[start : start + CLIP_SAMPLES]
            if len(seg) < CLIP_SAMPLES:
                seg = np.pad(seg, (0, CLIP_SAMPLES - len(seg)))
            return seg
        # focal: top-energy or random crop
        if self.train and self.rng.random() < self.top_energy_prob:
            crops = topk_energy_crops(y, k=3, n_samples=CLIP_SAMPLES)
            return crops[int(self.rng.integers(0, len(crops)))]
        if self.train:
            return random_crop(y, CLIP_SAMPLES, rng=self.rng)
        # val focal: deterministic top-1 energy crop
        return topk_energy_crops(y, k=1, n_samples=CLIP_SAMPLES)[0]

    # -- mel
    def _mel(self, y: np.ndarray) -> np.ndarray:
        m = logmel(y)
        if m.shape[1] < SPEC_FRAMES:
            m = np.pad(m, ((0, 0), (0, SPEC_FRAMES - m.shape[1])))
        elif m.shape[1] > SPEC_FRAMES:
            m = m[:, :SPEC_FRAMES]
        return (m - m.mean()) / (m.std() + 1e-6)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            y = self._load_clip(row)
        except Exception:
            y = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        mel = self._mel(y)

        if self.train and self.bg_pool is not None and self.rng.random() < self.bg_mix_prob:
            bg_row = self.bg_pool.iloc[int(self.rng.integers(0, len(self.bg_pool)))]
            try:
                bg_y = self._load_clip(bg_row)
                bg_mel = self._mel(bg_y)
                alpha = float(self.rng.uniform(0.2, 0.5))
                mel = (1 - alpha) * mel + alpha * bg_mel
            except Exception:
                pass

        if self.train and self.spec_aug is not None:
            mel = spec_augment(mel, self.spec_aug, self.rng)

        y_vec = encode_labels(row["labels"], row.get("secondary_labels", ""), row["weight_group"])
        return {
            "mel": torch.from_numpy(mel[None, :, :]).float(),
            "target": torch.from_numpy(y_vec).float(),
            "weight": torch.tensor(float(row.get("rating", 1.0) or 1.0) / 5.0 + 0.2, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_on_the_fly(val_fold: int, folds_csv: Path, spec_aug: SpecAugCfg | None = None,
                    bg_mix_prob: float = DEFAULTS.bg_mix_prob):
    df = build_index(folds_csv=folds_csv)
    val_mask = (df["source"] == "soundscape") & (df["fold"] == val_fold)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    bg_df = df[(df["source"] == "soundscape") & (df["fold"] != val_fold)].reset_index(drop=True)
    train_ds = OggOnTheFlyDataset(train_df, train=True, spec_aug=spec_aug,
                                  bg_pool=bg_df if len(bg_df) else None, bg_mix_prob=bg_mix_prob)
    val_ds = OggOnTheFlyDataset(val_df, train=False)
    return train_ds, val_ds


__all__ = ["OggOnTheFlyDataset", "build_index", "build_on_the_fly"]
