"""Extract frozen embeddings from an external audio encoder (Perch v2 / BirdNET)
for every 5-sec segment referenced in the mel cache index.

The embedder is loaded from a local path (TensorFlow SavedModel or TFLite) so
the pipeline works offline. Outputs:

  cache/emb/<sample_id>.npy     float16 (D,)
  cache/emb_index.parquet       (sample_id, source, labels, secondary_labels,
                                 weight_group, fold, file, seg_start, seg_end)

Usage:
  python -m data_prep.extract_embeddings --model-path <path-to-perch-or-birdnet> \
      --kind perch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio import iter_soundscape_segments, load_audio, topk_energy_crops
from src.config import (
    CACHE_DIR,
    CLIP_SAMPLES,
    SAMPLE_RATE,
    TRAIN_AUDIO_DIR,
    TRAIN_CSV,
    TRAIN_SOUNDSCAPES_DIR,
    TRAIN_SS_LABELS_CSV,
)

EMB_DIR = CACHE_DIR / "emb"
EMB_INDEX = CACHE_DIR / "emb_index.parquet"


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class PerchEmbedder:
    """Perch v2 TF SavedModel. Expects mono float32 @ 32 kHz, 5-sec window."""

    def __init__(self, model_path: Path):
        import tensorflow as tf
        self.tf = tf
        self.model = tf.saved_model.load(str(model_path))
        # Perch exposes .signatures['serving_default'] with input 'inputs' -> 'output_0'
        self.sig = self.model.signatures["serving_default"]

    def embed(self, wave: np.ndarray) -> np.ndarray:
        x = self.tf.constant(wave[None, :], dtype=self.tf.float32)
        out = self.sig(x)
        # pick the first tensor whose last-dim looks like an embedding vector
        emb = next(iter(out.values())).numpy()
        return emb.reshape(-1).astype(np.float32)


class BirdNetEmbedder:
    """BirdNET TFLite model; pull activations from the second-to-last layer."""

    def __init__(self, model_path: Path):
        import tflite_runtime.interpreter as tflite  # or tensorflow.lite
        self.it = tflite.Interpreter(model_path=str(model_path))
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        self.in_idx = inp["index"]
        # BirdNET convention: classifier is the last op; embedding is the op just before.
        self.emb_idx = self.it.get_output_details()[-1]["index"]

    def embed(self, wave: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.in_idx, wave.astype(np.float32)[None, :])
        self.it.invoke()
        emb = self.it.get_tensor(self.emb_idx)
        return emb.reshape(-1).astype(np.float32)


def build_embedder(kind: str, path: Path):
    if kind == "perch":
        return PerchEmbedder(path)
    if kind == "birdnet":
        return BirdNetEmbedder(path)
    raise ValueError(f"unknown embedder kind: {kind}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _ensure_len(y: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    elif len(y) > n:
        y = y[:n]
    return y


def extract(embedder, max_focal_crops: int = 3) -> pd.DataFrame:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # 1. Focal audio — same crop strategy as the mel cache so sample_ids align.
    train = pd.read_csv(TRAIN_CSV)
    for rec in tqdm(train.to_dict("records"), desc="focal"):
        path = TRAIN_AUDIO_DIR / rec["filename"]
        try:
            y = load_audio(path)
        except Exception:
            continue
        crops = topk_energy_crops(y, k=max_focal_crops, n_samples=CLIP_SAMPLES)
        stem = Path(rec["filename"]).stem.replace("/", "_")
        for i, c in enumerate(crops):
            sid = f"focal_{stem}_{i}"
            out = EMB_DIR / f"{sid}.npy"
            if out.exists():
                continue
            emb = embedder.embed(_ensure_len(c))
            np.save(out, emb.astype(np.float16))
            rows.append(
                {
                    "sample_id": sid, "source": "focal", "file": rec["filename"],
                    "seg_start": -1.0, "seg_end": -1.0,
                    "labels": rec["primary_label"],
                    "secondary_labels": rec.get("secondary_labels", ""),
                    "weight_group": "primary", "rating": rec.get("rating", 0.0), "fold": -1,
                }
            )

    # 2. Labeled soundscapes.
    ss = pd.read_csv(TRAIN_SS_LABELS_CSV)
    folds_path = CACHE_DIR / "folds.csv"
    folds = (
        pd.read_csv(folds_path).set_index("filename")["fold"].to_dict()
        if folds_path.exists() else {}
    )
    for rec in tqdm(ss.to_dict("records"), desc="soundscape"):
        path = TRAIN_SOUNDSCAPES_DIR / rec["filename"]
        try:
            y = load_audio(path)
        except Exception:
            continue
        start = int(float(rec["start"]) * SAMPLE_RATE)
        seg = _ensure_len(y[start : start + CLIP_SAMPLES])
        stem = Path(rec["filename"]).stem
        sid = f"ss_{stem}_{int(rec['start']):03d}"
        out = EMB_DIR / f"{sid}.npy"
        if not out.exists():
            emb = embedder.embed(seg)
            np.save(out, emb.astype(np.float16))
        rows.append(
            {
                "sample_id": sid, "source": "soundscape", "file": rec["filename"],
                "seg_start": float(rec["start"]), "seg_end": float(rec["end"]),
                "labels": rec["primary_label"], "secondary_labels": "",
                "weight_group": "soundscape", "rating": 5.0,
                "fold": folds.get(rec["filename"], 0),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(EMB_INDEX, index=False)
    print(f"wrote {len(df)} embeddings -> {EMB_DIR}")
    print(df.groupby("source").size())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--kind", choices=["perch", "birdnet"], required=True)
    ap.add_argument("--max-focal-crops", type=int, default=3)
    args = ap.parse_args()

    embedder = build_embedder(args.kind, args.model_path)
    extract(embedder, max_focal_crops=args.max_focal_crops)


if __name__ == "__main__":
    main()
