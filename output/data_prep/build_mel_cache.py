"""Offline mel-spec cache builder.

Consumes:
  - train.csv  (focal XC + iNat recordings -> primary/secondary labels)
  - train_soundscapes_labels.csv (strongly labeled 5-sec segments)

Produces per-sample .npy files under cache/mel/ with shape (N_MELS, T) float16,
plus a parquet index (cache/index.parquet) with columns:

    sample_id, source ('focal'|'soundscape'), file, seg_start, seg_end,
    labels (semicolon-separated primary_label codes),
    secondary_labels, weight_group, rating, fold

`weight_group` is used at training time to look up the per-source loss weight
(soundscape=1.0, primary=0.7, secondary=0.3 — secondary is encoded in labels,
not as a separate sample).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio import load_audio, logmel, topk_energy_crops
from src.config import (
    CACHE_DIR,
    CLIP_SAMPLES,
    SAMPLE_RATE,
    TRAIN_AUDIO_DIR,
    TRAIN_CSV,
    TRAIN_SOUNDSCAPES_DIR,
    TRAIN_SS_LABELS_CSV,
)

MEL_DIR = CACHE_DIR / "mel"
INDEX_PATH = CACHE_DIR / "index.parquet"


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def _save_mel(mel: np.ndarray, sample_id: str) -> Path:
    out = MEL_DIR / f"{sample_id}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mel.astype(np.float16))
    return out


def _process_focal(row: dict, max_crops: int) -> list[dict]:
    """Decode a focal training clip, pick top-energy crops, save mel shards."""
    path = TRAIN_AUDIO_DIR / row["filename"]
    try:
        y = load_audio(path)
    except Exception as e:  # corrupt / missing file
        return [{"error": f"{path}: {e}"}]
    crops = topk_energy_crops(y, k=max_crops, n_samples=CLIP_SAMPLES)
    out = []
    stem = Path(row["filename"]).stem.replace("/", "_")
    for i, c in enumerate(crops):
        mel = logmel(c)
        sid = f"focal_{stem}_{i}"
        _save_mel(mel, sid)
        out.append(
            {
                "sample_id": sid,
                "source": "focal",
                "file": row["filename"],
                "seg_start": -1.0,
                "seg_end": -1.0,
                "labels": row["primary_label"],
                "secondary_labels": row.get("secondary_labels", ""),
                "weight_group": "primary",
                "rating": row.get("rating", 0.0),
                "fold": -1,
            }
        )
    return out


def _process_soundscape_segment(row: dict, fold: int) -> list[dict]:
    """One labeled 5-sec segment from train_soundscapes_labels.csv."""
    path = TRAIN_SOUNDSCAPES_DIR / row["filename"]
    try:
        y = load_audio(path)
    except Exception as e:
        return [{"error": f"{path}: {e}"}]
    start = int(float(row["start"]) * SAMPLE_RATE)
    end = start + CLIP_SAMPLES
    seg = y[start:end]
    if len(seg) < CLIP_SAMPLES:
        seg = np.pad(seg, (0, CLIP_SAMPLES - len(seg)))
    mel = logmel(seg)
    stem = Path(row["filename"]).stem
    sid = f"ss_{stem}_{int(row['start']):03d}"
    _save_mel(mel, sid)
    return [
        {
            "sample_id": sid,
            "source": "soundscape",
            "file": row["filename"],
            "seg_start": float(row["start"]),
            "seg_end": float(row["end"]),
            "labels": row["primary_label"],
            "secondary_labels": "",
            "weight_group": "soundscape",
            "rating": 5.0,
            "fold": fold,
        }
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build(max_focal_crops: int = 3, workers: int = 8, limit: int | None = None) -> pd.DataFrame:
    MEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Focal audio
    train = pd.read_csv(TRAIN_CSV)
    if limit:
        train = train.head(limit)
    focal_rows = train.to_dict("records")

    # 2. Labeled soundscapes — with fold lookup.
    ss = pd.read_csv(TRAIN_SS_LABELS_CSV)
    folds_path = CACHE_DIR / "folds.csv"
    if folds_path.exists():
        folds = pd.read_csv(folds_path).set_index("filename")["fold"].to_dict()
    else:
        folds = {}
    ss_rows = [(r, folds.get(r["filename"], 0)) for r in ss.to_dict("records")]
    if limit:
        ss_rows = ss_rows[:limit]

    out: list[dict] = []
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for r in focal_rows:
            futs.append(ex.submit(_process_focal, r, max_focal_crops))
        for r, f in ss_rows:
            futs.append(ex.submit(_process_soundscape_segment, r, f))

        for fut in tqdm(as_completed(futs), total=len(futs), desc="mel cache"):
            for rec in fut.result():
                if "error" in rec:
                    errors.append(rec["error"])
                else:
                    out.append(rec)

    df = pd.DataFrame(out)
    df.to_parquet(INDEX_PATH, index=False)
    if errors:
        (CACHE_DIR / "build_errors.log").write_text("\n".join(errors), encoding="utf-8")
    print(f"wrote {len(df)} mel shards -> {MEL_DIR}")
    print(f"index -> {INDEX_PATH}  ({len(errors)} errors)")
    print(df.groupby("source").size())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-focal-crops", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap rows per source")
    args = ap.parse_args()
    build(max_focal_crops=args.max_focal_crops, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
