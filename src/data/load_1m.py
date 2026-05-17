"""Raw loaders for MovieLens 1M (.dat files, '::' separator, latin-1)."""

from __future__ import annotations

import pandas as pd

from src.config import ML_1M_DIR

_READ_KW = {"sep": "::", "engine": "python", "encoding": "latin-1", "header": None}


def load_users() -> pd.DataFrame:
    return pd.read_csv(
        ML_1M_DIR / "users.dat",
        names=["user_id", "gender", "age_code", "occupation_code", "zip"],
        dtype={"user_id": "int32", "age_code": "int8", "occupation_code": "int8", "zip": "string"},
        **_READ_KW,
    )


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        ML_1M_DIR / "movies.dat",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
        **_READ_KW,
    )


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        ML_1M_DIR / "ratings.dat",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"},
        **_READ_KW,
    )
