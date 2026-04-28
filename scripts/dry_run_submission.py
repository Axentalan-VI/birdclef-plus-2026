"""Local dry-run of the Kaggle submission pipeline.

Points the inference code at a small subset of `train_soundscapes/` to verify:
  1. Model weights load and run on CPU.
  2. Output is a valid submission.csv (rows = files * 12, 234 species columns).
  3. Projected runtime at 600 files fits under 90 min.

Usage:
  python -m scripts.dry_run_submission --weights-dir kaggle_assets --n-files 20
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf

from src.config import (
    CLIP_SAMPLES,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    TOP_DB,
    TRAIN_SOUNDSCAPES_DIR,
    WIN_LENGTH,
)
from src.taxonomy import class_list

CLASSES = class_list()
NC = len(CLASSES)
MEL_FB = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype(np.float32)


def load_audio(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y.astype(np.float32, copy=False)


def segment_file(y: np.ndarray) -> np.ndarray:
    total = int(np.ceil(len(y) / CLIP_SAMPLES)) * CLIP_SAMPLES
    if len(y) < total:
        y = np.pad(y, (0, total - len(y)))
    return y.reshape(-1, CLIP_SAMPLES)


def logmel_batch(segs: np.ndarray) -> np.ndarray:
    out = np.empty((segs.shape[0], N_MELS, CLIP_SAMPLES // HOP_LENGTH + 1), dtype=np.float32)
    for i, s in enumerate(segs):
        S = librosa.stft(s, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, center=True)
        P = (S.real ** 2 + S.imag ** 2).astype(np.float32)
        out[i] = librosa.power_to_db(MEL_FB @ P, top_db=TOP_DB)
    m = out.mean(axis=(1, 2), keepdims=True)
    sd = out.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((out - m) / sd)[:, None, :, :]


def make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = os.cpu_count() or 4
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", type=Path, default=Path("kaggle_assets"))
    ap.add_argument("--n-files", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("dry_run_submission.csv"))
    ap.add_argument("--target-total-files", type=int, default=600, help="used only for runtime projection")
    args = ap.parse_args()

    onnx_files = sorted(args.weights_dir.glob("*.int8.onnx"))
    if not onnx_files:
        raise SystemExit(f"no .int8.onnx found under {args.weights_dir}")
    sessions = [make_session(p) for p in onnx_files]
    print(f"loaded {len(sessions)} model(s): {[p.name for p in onnx_files]}")

    files = sorted(TRAIN_SOUNDSCAPES_DIR.glob("*.ogg"))[: args.n_files]
    if not files:
        raise SystemExit(f"no .ogg files in {TRAIN_SOUNDSCAPES_DIR}")
    print(f"dry-running on {len(files)} file(s)")

    row_ids: list[str] = []
    probs_chunks: list[np.ndarray] = []
    t0 = time.time()
    for p in files:
        y = load_audio(p)
        segs = segment_file(y)
        mels = logmel_batch(segs).astype(np.float32)
        logits = np.zeros((mels.shape[0], NC), dtype=np.float32)
        for s in sessions:
            logits += s.run(["logits"], {"mel": mels})[0]
        logits /= len(sessions)
        probs = 1.0 / (1.0 + np.exp(-logits))
        if probs.shape[0] >= 3:
            pad = np.pad(probs, ((1, 1), (0, 0)), mode="edge")
            probs = 0.2 * pad[:-2] + 0.6 * pad[1:-1] + 0.2 * pad[2:]
        probs = probs * (0.8 + 0.2 * probs.max(axis=0, keepdims=True))
        for i in range(probs.shape[0]):
            row_ids.append(f"{p.stem}_{(i+1)*5}")
        probs_chunks.append(probs)

    elapsed = time.time() - t0
    all_probs = np.concatenate(probs_chunks, axis=0)
    sub = pd.DataFrame(all_probs, columns=CLASSES)
    sub.insert(0, "row_id", row_ids)
    sub.to_csv(args.out, index=False, float_format="%.5f")

    # Sanity asserts.
    assert len(sub) == len(files) * 12, f"expected {len(files)*12} rows, got {len(sub)}"
    assert sub.shape[1] == NC + 1, f"expected {NC+1} cols, got {sub.shape[1]}"
    assert sub.drop(columns=["row_id"]).columns.tolist() == CLASSES

    projected = elapsed / len(files) * args.target_total_files
    print(f"OK: {len(sub)} rows x {sub.shape[1]} cols -> {args.out}")
    print(f"elapsed {elapsed:.1f}s on {len(files)} files")
    print(f"projected at {args.target_total_files} files: {projected/60:.1f} min")
    print("PASS (under 80 min budget)" if projected < 80 * 60 else "WARN: exceeds 80 min budget")


if __name__ == "__main__":
    main()
