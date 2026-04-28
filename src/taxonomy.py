"""Helpers around `taxonomy.csv` — the 234 submission columns."""
from __future__ import annotations

from functools import lru_cache
from typing import List

import pandas as pd

from .config import TAXONOMY_CSV


@lru_cache(maxsize=1)
def load_taxonomy() -> pd.DataFrame:
    df = pd.read_csv(TAXONOMY_CSV)
    if "primary_label" not in df.columns:
        raise ValueError(f"taxonomy.csv missing 'primary_label' column: {df.columns}")
    return df


@lru_cache(maxsize=1)
def class_list() -> List[str]:
    """Ordered list of the 234 species column names (== submission column order)."""
    return load_taxonomy()["primary_label"].tolist()


@lru_cache(maxsize=1)
def class_to_idx() -> dict[str, int]:
    return {c: i for i, c in enumerate(class_list())}


def num_classes() -> int:
    return len(class_list())
