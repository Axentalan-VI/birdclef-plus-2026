"""Shared audio / spectrogram utilities.

Used both offline (cache building, training) and in the Kaggle CPU notebook.
Deliberately depends only on numpy + soundfile at inference time; torch / librosa
are imported lazily so the module can be re-used in a pure-numpy submission env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import soundfile as sf

from .config import (
    CLIP_SAMPLES,
    CLIP_SECONDS,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    TOP_DB,
    WIN_LENGTH,
)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load mono float32 audio at SAMPLE_RATE. Resamples with librosa if needed."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa  # lazy
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32, copy=False)


def split_into_segments(y: np.ndarray, seg_samples: int = CLIP_SAMPLES) -> np.ndarray:
    """Pad or crop to multiple of `seg_samples`, then reshape -> (n_segments, seg_samples)."""
    n = len(y)
    if n == 0:
        return np.zeros((1, seg_samples), dtype=np.float32)
    n_full = int(np.ceil(n / seg_samples))
    pad = n_full * seg_samples - n
    if pad > 0:
        y = np.pad(y, (0, pad), mode="constant")
    return y.reshape(n_full, seg_samples)


def random_crop(y: np.ndarray, n_samples: int = CLIP_SAMPLES, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if len(y) <= n_samples:
        out = np.zeros(n_samples, dtype=np.float32)
        out[: len(y)] = y
        return out
    start = int(rng.integers(0, len(y) - n_samples + 1))
    return y[start : start + n_samples]


def topk_energy_crops(y: np.ndarray, k: int = 3, n_samples: int = CLIP_SAMPLES) -> List[np.ndarray]:
    """Return up to k non-overlapping 5-sec crops ranked by signal energy."""
    if len(y) <= n_samples:
        out = np.zeros(n_samples, dtype=np.float32)
        out[: len(y)] = y
        return [out]
    segs = split_into_segments(y, n_samples)
    energies = (segs ** 2).mean(axis=1)
    order = np.argsort(-energies)
    return [segs[i] for i in order[: min(k, len(segs))]]


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------
_MEL_FILTERBANK: np.ndarray | None = None
_HANN_WIN: np.ndarray | None = None


def _mel_filterbank() -> np.ndarray:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        import librosa  # lazy
        _MEL_FILTERBANK = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
        ).astype(np.float32)
    return _MEL_FILTERBANK


def _hann() -> np.ndarray:
    global _HANN_WIN
    if _HANN_WIN is None:
        _HANN_WIN = np.hanning(WIN_LENGTH).astype(np.float32)
    return _HANN_WIN


def logmel(y: np.ndarray) -> np.ndarray:
    """Compute (N_MELS, T) log-mel dB spectrogram for a single waveform."""
    import librosa  # lazy, only used offline
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, center=True)
    power = (S.real ** 2 + S.imag ** 2).astype(np.float32)
    mel = _mel_filterbank() @ power
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=TOP_DB)
    return mel_db.astype(np.float32)


def batch_logmel(segments: np.ndarray) -> np.ndarray:
    """(B, T_samples) waveforms -> (B, 1, N_MELS, T_frames) normalized log-mel."""
    out = np.stack([logmel(s) for s in segments], axis=0)
    # per-image mean/std normalization
    m = out.mean(axis=(1, 2), keepdims=True)
    s = out.std(axis=(1, 2), keepdims=True) + 1e-6
    out = (out - m) / s
    return out[:, None, :, :]


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------
def iter_soundscape_segments(path: str | Path, sr: int = SAMPLE_RATE) -> Iterable[np.ndarray]:
    """Yield consecutive 5-sec segments from a soundscape file."""
    y = load_audio(path, sr=sr)
    for seg in split_into_segments(y, CLIP_SAMPLES):
        yield seg


__all__ = [
    "load_audio",
    "split_into_segments",
    "random_crop",
    "topk_energy_crops",
    "logmel",
    "batch_logmel",
    "iter_soundscape_segments",
]
