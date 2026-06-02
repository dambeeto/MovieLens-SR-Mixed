"""Lazy-load a trained recommender and serve top-N for a single user."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODELS_RECSYS_DIR, PROCESSED_DIR, RECSYS_TOP_N

_MODEL_FILES = {
    "svd": MODELS_RECSYS_DIR / "svd.joblib",
    "knn_cosine": MODELS_RECSYS_DIR / "item_knn_cosine.joblib",
    "knn_pearson": MODELS_RECSYS_DIR / "item_knn_pearson.joblib",
}


@lru_cache(maxsize=4)
def _load_model(name: str):
    path = _MODEL_FILES.get(name)
    if path is None:
        raise ValueError(f"unknown model {name!r}; choose from {list(_MODEL_FILES)}")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run scripts/07_recsys.py first to train and persist models."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_movies() -> pd.DataFrame:
    path = PROCESSED_DIR / "movies_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/02_clean.py first.")
    return pd.read_parquet(path, columns=["movie_id", "title_clean", "year"])


def top_n_for_user(user_id: int, n: int = RECSYS_TOP_N, model_name: str = "svd") -> list[dict]:
    """Return [{movie_id, title, year, predicted_rating}, ...] for a user."""
    model = _load_model(model_name)
    movies = _load_movies().set_index("movie_id")
    out: list[dict] = []
    for movie_id, score in model.top_n(int(user_id), n=int(n), exclude_seen=True):
        row = movies.loc[movie_id] if movie_id in movies.index else None
        out.append(
            {
                "movie_id": int(movie_id),
                "title": str(row["title_clean"]) if row is not None else "?",
                "year": int(row["year"]) if row is not None and pd.notna(row["year"]) else None,
                "predicted_rating": float(score),
            }
        )
    return out
