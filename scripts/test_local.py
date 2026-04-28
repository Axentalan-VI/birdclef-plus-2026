"""Local sanity test: download a few BirdCLEF train_audio samples via Kaggle CLI,
run inference with output/best.pt, print top-5 predictions vs ground-truth species.

Requires: kaggle CLI configured (~/.kaggle/kaggle.json), torch, timm, librosa, soundfile.

Usage:
  python scripts/test_local.py            # 5 random species, 1 file each
  python scripts/test_local.py --n 10     # more samples
  python scripts/test_local.py --species amakin,grbher1
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import timm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
COMP = "birdclef-2026"

# --- audio constants (must match training) ---------------------------------
SR = 32_000
CLIP = 5 * SR
N_FFT, HOP, WIN = 1024, 320, 1024
N_MELS, FMIN, FMAX = 128, 20, 16_000
TOP_DB = 80.0

MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype(np.float32)


def kaggle_get(remote: str, dest_dir: Path = DATA) -> Path:
    """Download a single competition file via Kaggle CLI; handles auto .zip wrap."""
    target = dest_dir / remote
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] fetching {remote}")
    subprocess.check_call(
        ["kaggle", "competitions", "download", "-c", COMP, "-f", remote,
         "-p", str(target.parent), "--quiet"]
    )
    # CLI may save as <name>.zip
    z = target.parent / (Path(remote).name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target.parent)
        z.unlink()
    if not target.exists():
        # some CLI versions drop the file flat in dest_dir
        flat = dest_dir / Path(remote).name
        if flat.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            flat.rename(target)
    if not target.exists():
        raise FileNotFoundError(f"download failed: {remote}")
    return target


def build_model(ckpt: dict, num_classes: int) -> nn.Module:
    cfg = ckpt.get("cfg", {})
    backbone_name = cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")
    drop_rate = float(cfg.get("drop_rate", 0.3))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, in_chans=1,
                num_classes=0, global_pool="avg",
            )
            feat = self.backbone.num_features
            self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(feat, num_classes))

        def forward(self, x):
            return self.head(self.backbone(x))

    m = M()
    missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    m.eval()
    return m


def logmel(seg: np.ndarray) -> np.ndarray:
    S = librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, win_length=WIN, center=True)
    P = (S.real ** 2 + S.imag ** 2).astype(np.float32)
    M = librosa.power_to_db(MEL_FB @ P, top_db=TOP_DB)
    return ((M - M.mean()) / (M.std() + 1e-6)).astype(np.float32)


def predict(model: nn.Module, path: Path, max_seconds: int = 30) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    n_keep = min(len(y), max_seconds * SR)
    y = y[:n_keep]
    if len(y) < CLIP:
        y = np.pad(y, (0, CLIP - len(y)))
    segs = [logmel(y[i:i + CLIP]) for i in range(0, len(y) - CLIP + 1, CLIP)]
    if not segs:
        segs = [logmel(y[:CLIP])]
    x = torch.from_numpy(np.stack(segs))[:, None, :, :]  # (B,1,N_MELS,T)
    with torch.no_grad():
        logits = model(x).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.mean(axis=0)


def resolve_train_path(rel: str) -> str:
    """train.csv 'filename' col may be 'amakin/XC123.ogg' or just 'XC123.ogg'."""
    rel = rel.replace("\\", "/")
    if rel.startswith("train_audio/"):
        return rel
    return f"train_audio/{rel}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "output" / "best.pt"))
    ap.add_argument("--n", type=int, default=5, help="number of random species")
    ap.add_argument("--per-species", type=int, default=1, help="files per species")
    ap.add_argument("--species", default=None, help="comma-separated primary_label codes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    # 1. Pull metadata
    tax_path = kaggle_get("taxonomy.csv")
    train_path = kaggle_get("train.csv")
    tax = pd.read_csv(tax_path)
    train = pd.read_csv(train_path)
    classes = tax["primary_label"].tolist()
    NC = len(classes)
    print(f"taxonomy: {NC} classes   train.csv: {len(train)} rows  cols={list(train.columns)[:8]}")

    # 2. Pick species + audio files
    if args.species:
        species_pick = [s.strip() for s in args.species.split(",") if s.strip()]
    else:
        avail = sorted(set(train["primary_label"]) & set(classes))
        species_pick = random.sample(avail, min(args.n, len(avail)))

    file_col = "filename" if "filename" in train.columns else train.columns[1]
    samples: list[tuple[str, Path]] = []
    for sp in species_pick:
        rows = train[train["primary_label"] == sp].head(args.per_species)
        for _, r in rows.iterrows():
            try:
                p = kaggle_get(resolve_train_path(str(r[file_col])))
                samples.append((sp, p))
            except Exception as e:
                print(f"  [skip] {sp}/{r[file_col]}: {e}")
    if not samples:
        print("no samples downloaded")
        sys.exit(1)

    # 3. Load model
    print(f"\nloading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  ckpt auc={ckpt.get('auc', float('nan')):.4f}  epoch={ckpt.get('epoch')}")
    model = build_model(ckpt, NC)

    # 4. Inference + report
    cls_idx = {c: i for i, c in enumerate(classes)}
    print(f"\n{'file':<42} {'true':<10} {'rank':<6} top-5")
    print("-" * 110)
    correct1 = correct5 = 0
    for sp, p in samples:
        probs = predict(model, p)
        order = np.argsort(probs)[::-1]
        true_i = cls_idx.get(sp, -1)
        rank = int(np.where(order == true_i)[0][0]) + 1 if true_i >= 0 else -1
        top5 = ", ".join(f"{classes[i]}({probs[i]:.2f})" for i in order[:5])
        print(f"{p.name:<42} {sp:<10} #{rank:<5} {top5}")
        if rank == 1: correct1 += 1
        if 0 < rank <= 5: correct5 += 1
    n = len(samples)
    print("-" * 110)
    print(f"top-1 acc: {correct1}/{n} ({correct1/n:.1%})   top-5 acc: {correct5}/{n} ({correct5/n:.1%})")


if __name__ == "__main__":
    main()
