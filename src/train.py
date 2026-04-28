"""Training loop for Model A (timm CNN on log-mel).

Usage:
  python -m src.train --config configs/effnet_b0.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DEFAULTS, RUNS_DIR
from .datasets import SpecAugCfg, build_train_val_datasets
from .losses import FocalBCE, WeightedBCEWithLogits, compute_pos_weight
from .metrics import birdclef_roc_auc
from .models import build_model
from .taxonomy import num_classes


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_batch(mel: torch.Tensor, target: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return mel, target
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(mel.size(0), device=mel.device)
    return lam * mel + (1 - lam) * mel[idx], torch.maximum(target, target[idx] * lam)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        logits = model(batch["mel"].to(device, non_blocking=True))
        preds.append(torch.sigmoid(logits).cpu().numpy())
        tgts.append(batch["target"].numpy())
    preds = np.concatenate(preds, axis=0)
    tgts = np.concatenate(tgts, axis=0)
    # binarize soundscape val (soundscape targets are 0/1 already)
    return birdclef_roc_auc((tgts > 0.5).astype(np.float32), preds)


def train(cfg: dict) -> None:
    set_seed(cfg.get("seed", DEFAULTS.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec_aug = SpecAugCfg(**cfg.get("spec_aug", {})) if cfg.get("spec_aug_enabled", True) else None
    train_ds, val_ds = build_train_val_datasets(
        val_fold=cfg["val_fold"],
        spec_aug=spec_aug,
        bg_mix_prob=cfg.get("bg_mix_prob", DEFAULTS.bg_mix_prob),
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", DEFAULTS.batch_size),
        shuffle=True, num_workers=cfg.get("num_workers", DEFAULTS.num_workers),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", DEFAULTS.batch_size) * 2,
        shuffle=False, num_workers=cfg.get("num_workers", DEFAULTS.num_workers),
        pin_memory=True,
    )

    model = build_model(
        cfg.get("model_kind", "timm"),
        backbone=cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k"),
        num_classes=num_classes(),
        pretrained=cfg.get("pretrained", True),
        drop_rate=cfg.get("drop_rate", 0.2),
    ).to(device)

    # pos_weight from training targets (approximate by class frequencies in df)
    targets_sum = torch.zeros(num_classes())
    for s in train_ds.df["labels"].fillna("").tolist():
        for code in str(s).replace(",", ";").split(";"):
            code = code.strip()
            from .taxonomy import class_to_idx
            if code in class_to_idx():
                targets_sum[class_to_idx()[code]] += 1
    pos_weight = compute_pos_weight(targets_sum, n_samples=len(train_ds), cap=cfg.get("pos_weight_cap", 50.0))

    if cfg.get("loss", "bce") == "focal":
        criterion = FocalBCE(alpha=cfg.get("focal_alpha", 0.25), gamma=cfg.get("focal_gamma", 2.0))
    else:
        criterion = WeightedBCEWithLogits(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", DEFAULTS.lr), weight_decay=cfg.get("weight_decay", DEFAULTS.weight_decay))
    epochs = cfg.get("epochs", DEFAULTS.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_name = cfg.get("run_name") or f"{cfg.get('backbone','model').replace('/', '_')}_fold{cfg['val_fold']}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"ep{ep+1}/{epochs}", leave=False)
        for batch in pbar:
            mel = batch["mel"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            mel, tgt = mixup_batch(mel, tgt, cfg.get("mixup_alpha", DEFAULTS.mixup_alpha))
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(mel)
                loss = criterion(logits, tgt, batch["weight"].to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        auc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"[{run_name}] epoch {ep+1}: val_auc={auc:.4f}  ({dt:.1f}s)")
        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "cfg": cfg, "auc": auc}, run_dir / "best.pt")
            print(f"  -> new best {auc:.4f}, saved.")

    print(f"done. best_auc={best_auc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train(cfg)


if __name__ == "__main__":
    main()
