"""Chronological train/test split on a ratings DataFrame."""

from __future__ import annotations

import pandas as pd


def time_split(ratings: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest `ratio` of ratings (by timestamp) -> train, rest -> test.

    Avoids future leakage that would otherwise make recommender / popularity
    evaluation trivially optimistic.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be in (0, 1)")
    df = ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(df) * ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
