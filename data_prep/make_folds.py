"""Build the held-out soundscape validation split.

Validation is the hardest decision in this competition: focal-audio val wildly
over-estimates generalization. We split `train_soundscapes_labels.csv` by
*recording site* (extracted from the filename: `BC2026_Test_<id>_<site>_...`).
Soundscape files are the unit of splitting — never individual 5-sec segments —
so a file ends up entirely in train or val for a given fold.

Outputs:
  cache/folds.csv  columns = [filename, site, fold]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, DEFAULTS, TRAIN_SS_LABELS_CSV

SITE_RE = re.compile(r"_S(\d+)_")


def site_from_filename(name: str) -> str:
    m = SITE_RE.search(name)
    return m.group(0).strip("_") if m else "UNK"


def build_folds(n_folds: int = DEFAULTS.num_folds, seed: int = DEFAULTS.seed) -> pd.DataFrame:
    """Site-grouped K-fold that balances by *labeled segment count*, not file count.

    Old version balanced files; with one site contributing ~10x more segments than
    others the result was 7/6/6/7/40 files -> a useless 168-segment val set.
    New version: weighted greedy by `n_segments` per site so val folds are even.
    """
    labels = pd.read_csv(TRAIN_SS_LABELS_CSV)
    seg_per_file = labels.groupby("filename").size().rename("n_seg")
    files = (
        labels[["filename"]].drop_duplicates()
        .merge(seg_per_file, left_on="filename", right_index=True)
    )
    files["site"] = files["filename"].map(site_from_filename)

    # Aggregate by site so a single site cannot be split across folds.
    site_seg = files.groupby("site")["n_seg"].sum().sort_values(ascending=False)

    fold_seg = np.zeros(n_folds, dtype=float)
    site_to_fold: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for site, n in site_seg.items():
        # Greedy: assign to currently smallest fold; jitter for tie-break.
        order = np.argsort(fold_seg + rng.random(n_folds) * 1e-3)
        f = int(order[0])
        site_to_fold[site] = f
        fold_seg[f] += float(n)

    files["fold"] = files["site"].map(site_to_fold)
    files = files.reset_index(drop=True)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=DEFAULTS.num_folds)
    ap.add_argument("--seed", type=int, default=DEFAULTS.seed)
    ap.add_argument("--out", type=Path, default=CACHE_DIR / "folds.csv")
    args = ap.parse_args()

    df = build_folds(n_folds=args.n_folds, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} file rows across {df['site'].nunique()} sites -> {args.out}")
    print(df.groupby("fold").size())


if __name__ == "__main__":
    main()
