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
    labels = pd.read_csv(TRAIN_SS_LABELS_CSV)
    files = labels["filename"].drop_duplicates().to_frame()
    files["site"] = files["filename"].map(site_from_filename)

    # Group files by site, distribute sites across folds by size (GroupKFold-ish).
    site_counts = files.groupby("site").size().sort_values(ascending=False)
    fold_sizes = np.zeros(n_folds, dtype=int)
    site_to_fold: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for site, n in site_counts.items():
        # assign to fold with smallest current size; break ties randomly
        order = np.argsort(fold_sizes + rng.random(n_folds) * 1e-6)
        f = int(order[0])
        site_to_fold[site] = f
        fold_sizes[f] += int(n)

    files["fold"] = files["site"].map(site_to_fold)
    return files.reset_index(drop=True)


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
