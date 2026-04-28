"""Pseudo-labeling round: infer on unlabeled train_soundscapes with a trained
Model A, keep per-class top-k high-confidence 5-sec segments as new pseudo
positives, and emit an updated mel cache + index that extends the original one.

Usage:
  python -m data_prep.pseudo_label --ckpt runs/effnet_b0_fold0/best.pt \
      --backbone tf_efficientnet_b0.ns_jft_in1k --top-k 200 --min-prob 0.6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.audio import iter_soundscape_segments, logmel
from src.config import CACHE_DIR, TRAIN_SOUNDSCAPES_DIR, TRAIN_SS_LABELS_CSV
from src.models import build_model
from src.taxonomy import class_list, num_classes

PSEUDO_MEL_DIR = CACHE_DIR / "mel"
INDEX_PATH = CACHE_DIR / "index.parquet"


def unlabeled_soundscapes() -> list[Path]:
    labeled = set(pd.read_csv(TRAIN_SS_LABELS_CSV)["filename"].unique())
    return [p for p in TRAIN_SOUNDSCAPES_DIR.glob("*.ogg") if p.name not in labeled]


def predict_file(model: torch.nn.Module, path: Path, device: torch.device) -> np.ndarray:
    """Return (n_segments, num_classes) probabilities."""
    from src.datasets import SPEC_FRAMES
    segs = list(iter_soundscape_segments(path))
    if not segs:
        return np.zeros((0, num_classes()), dtype=np.float32)
    mels = []
    for s in segs:
        m = logmel(s)
        if m.shape[1] < SPEC_FRAMES:
            m = np.pad(m, ((0, 0), (0, SPEC_FRAMES - m.shape[1])))
        elif m.shape[1] > SPEC_FRAMES:
            m = m[:, :SPEC_FRAMES]
        m = (m - m.mean()) / (m.std() + 1e-6)
        mels.append(m)
    x = torch.from_numpy(np.stack(mels)[:, None]).float().to(device)
    with torch.no_grad():
        logits = model(x)
    return torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--top-k", type=int, default=200, help="per-class top-k positives")
    ap.add_argument("--min-prob", type=float, default=0.6)
    ap.add_argument("--out-index", type=Path, default=CACHE_DIR / "index_pseudo.parquet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model("timm", backbone=args.backbone, num_classes=num_classes(), pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])

    files = unlabeled_soundscapes()
    print(f"scoring {len(files)} unlabeled soundscapes...")

    # (file, seg_idx, probs) accumulator
    all_probs = []
    seg_refs: list[tuple[str, int]] = []
    for path in tqdm(files):
        probs = predict_file(model, path, device)
        for i in range(probs.shape[0]):
            seg_refs.append((path.name, i))
        all_probs.append(probs)
    P = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes()), dtype=np.float32)
    print("prob matrix:", P.shape)

    # Select per-class top-k above min-prob.
    kept: dict[tuple[str, int], np.ndarray] = {}
    classes = class_list()
    for c in range(P.shape[1]):
        col = P[:, c]
        idx = np.argsort(-col)[: args.top_k]
        idx = idx[col[idx] >= args.min_prob]
        for i in idx:
            fn, seg = seg_refs[i]
            key = (fn, seg)
            if key not in kept:
                kept[key] = np.zeros(len(classes), dtype=np.float32)
            kept[key][c] = 1.0

    print(f"pseudo-labeled segments: {len(kept)}")

    # Materialize mels + rows.
    PSEUDO_MEL_DIR.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for (fn, seg), labels_vec in tqdm(kept.items(), desc="saving"):
        segs = list(iter_soundscape_segments(TRAIN_SOUNDSCAPES_DIR / fn))
        if seg >= len(segs):
            continue
        m = logmel(segs[seg])
        sid = f"pl_{Path(fn).stem}_{seg:03d}"
        np.save(PSEUDO_MEL_DIR / f"{sid}.npy", m.astype(np.float16))
        codes = ";".join(c for c, v in zip(classes, labels_vec) if v > 0)
        new_rows.append(
            {
                "sample_id": sid,
                "source": "pseudo",
                "file": fn,
                "seg_start": float(seg * 5),
                "seg_end": float(seg * 5 + 5),
                "labels": codes,
                "secondary_labels": "",
                "weight_group": "primary",  # treat as primary-strength
                "rating": 0.0,
                "fold": -1,
            }
        )

    base = pd.read_parquet(INDEX_PATH) if INDEX_PATH.exists() else pd.DataFrame()
    out = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_parquet(args.out_index, index=False)
    print(f"wrote {args.out_index}  total rows={len(out)}")


if __name__ == "__main__":
    main()
