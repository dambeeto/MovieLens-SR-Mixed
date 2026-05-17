"""Raw loaders for MovieLens 20M (CSV)."""

from __future__ import annotations

import pandas as pd

from src.config import ML_20M_DIR


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        ML_20M_DIR / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})


def load_tags() -> pd.DataFrame:
    return pd.read_csv(
        ML_20M_DIR / "tags.csv",
        dtype={"userId": "int32", "movieId": "int32", "tag": "string", "timestamp": "int64"},
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})


def integrity_check_genome_scores() -> dict:
    """Stream genome-scores.csv to confirm it parses; return row count + dtype summary.

    Not loaded into a DataFrame -- the file is ~300 MB and unused in this iteration."""
    path = ML_20M_DIR / "genome-scores.csv"
    total = 0
    for chunk in pd.read_csv(
        path,
        chunksize=1_000_000,
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    ):
        total += len(chunk)
    return {"file": str(path), "rows": total}
