"""Train Model B (MLP head) on cached external embeddings."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DEFAULTS, RUNS_DIR
from .emb_dataset import build_emb_datasets, embedding_dim
from .losses import FocalBCE, WeightedBCEWithLogits, compute_pos_weight
from .metrics import birdclef_roc_auc
from .models import build_model
from .taxonomy import class_to_idx, num_classes


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    preds, tgts = [], []
    for b in loader:
        p = torch.sigmoid(model(b["x"].to(device))).cpu().numpy()
        preds.append(p); tgts.append(b["target"].numpy())
    return birdclef_roc_auc(
        (np.concatenate(tgts) > 0.5).astype(np.float32),
        np.concatenate(preds),
    )


def train(cfg: dict) -> None:
    set_seed(cfg.get("seed", DEFAULTS.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_emb_datasets(val_fold=cfg["val_fold"])
    print(f"train={len(train_ds)}  val={len(val_ds)}  emb_dim={embedding_dim()}")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 256), shuffle=True,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 256) * 2, shuffle=False,
                            num_workers=cfg.get("num_workers", 2), pin_memory=True)

    model = build_model(
        "embhead",
        emb_dim=embedding_dim(),
        num_classes=num_classes(),
        hidden=cfg.get("hidden", 512),
        drop=cfg.get("drop", 0.3),
    ).to(device)

    # pos_weight from training labels.
    targets_sum = torch.zeros(num_classes())
    c2i = class_to_idx()
    for s in train_ds.df["labels"].fillna("").tolist():
        for code in str(s).replace(",", ";").split(";"):
            code = code.strip()
            if code in c2i:
                targets_sum[c2i[code]] += 1
    pos_weight = compute_pos_weight(targets_sum, n_samples=len(train_ds), cap=cfg.get("pos_weight_cap", 50.0))

    criterion = (
        FocalBCE(alpha=cfg.get("focal_alpha", 0.25), gamma=cfg.get("focal_gamma", 2.0))
        if cfg.get("loss", "bce") == "focal"
        else WeightedBCEWithLogits(pos_weight=pos_weight)
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 30)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * max(1, len(train_loader)))

    run_dir = RUNS_DIR / (cfg.get("run_name") or f"embhead_fold{cfg['val_fold']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for b in tqdm(train_loader, desc=f"ep{ep+1}/{epochs}", leave=False):
            x = b["x"].to(device, non_blocking=True)
            t = b["target"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), t, b["weight"].to(device, non_blocking=True))
            loss.backward(); optim.step(); sched.step()

        auc = evaluate(model, val_loader, device)
        print(f"ep {ep+1}: val_auc={auc:.4f}  ({time.time()-t0:.1f}s)")
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "cfg": cfg, "auc": auc, "emb_dim": embedding_dim()},
                       run_dir / "best.pt")
    print(f"done. best={best:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    train(yaml.safe_load(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
