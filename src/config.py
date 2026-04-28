"""Global constants and path resolution for the BirdCLEF+ 2026 pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the Kaggle competition data. Override with BIRDCLEF_DATA env var.
DATA_ROOT = Path(os.environ.get("BIRDCLEF_DATA", "E:/Kaggle/BirdCLEF+ 2026/data"))

TRAIN_AUDIO_DIR = DATA_ROOT / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_ROOT / "train_soundscapes"
TEST_SOUNDSCAPES_DIR = DATA_ROOT / "test_soundscapes"
TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_SS_LABELS_CSV = DATA_ROOT / "train_soundscapes_labels.csv"
TAXONOMY_CSV = DATA_ROOT / "taxonomy.csv"
SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"

# Artifacts produced by this repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"
RUNS_DIR = REPO_ROOT / "runs"
ASSETS_DIR = REPO_ROOT / "kaggle_assets"

for d in (CACHE_DIR, RUNS_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio / spectrogram
# ---------------------------------------------------------------------------
SAMPLE_RATE = 32_000
CLIP_SECONDS = 5.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)  # 160_000

N_FFT = 1024
HOP_LENGTH = 320          # 10 ms at 32 kHz
WIN_LENGTH = 1024
N_MELS = 128
FMIN = 20
FMAX = 16_000
SPEC_FRAMES = CLIP_SAMPLES // HOP_LENGTH + 1  # 501

TOP_DB = 80.0             # log-mel clipping range

# Soundscape files are 60s -> 12 segments of 5s each.
SOUNDSCAPE_SECONDS = 60
SEGMENTS_PER_SOUNDSCAPE = SOUNDSCAPE_SECONDS // int(CLIP_SECONDS)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
@dataclass
class LabelWeights:
    soundscape: float = 1.0
    primary: float = 0.7
    secondary: float = 0.3


@dataclass
class TrainDefaults:
    num_folds: int = 5
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mixup_alpha: float = 0.5
    bg_mix_prob: float = 0.5
    label_weights: LabelWeights = field(default_factory=LabelWeights)


DEFAULTS = TrainDefaults()
