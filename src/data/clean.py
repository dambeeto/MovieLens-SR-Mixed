"""Cleaning + missing-value reporting for MovieLens 1M and 20M."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import (
    AGE_CODE_TO_LABEL,
    AGE_CODE_TO_MIDPOINT,
    GENRES_1M,
    OCCUPATION_CODE_TO_LABEL,
)

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_ZIP_RE = re.compile(r"^(\d{5})")


def report_missing(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per-column missing-value report (count + percentage)."""
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            missing = s.isna().sum() + (s.astype("string").str.len().fillna(0) == 0).sum()
        else:
            missing = s.isna().sum()
        rows.append(
            {
                "dataset": dataset,
                "column": col,
                "n_missing": int(missing),
                "pct_missing": round(100.0 * missing / n, 4) if n else 0.0,
                "n_rows": n,
            }
        )
    return pd.DataFrame(rows)


def clean_users_1m(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out["age_label"] = out["age_code"].map(AGE_CODE_TO_LABEL).astype("string")
    out["age_midpoint"] = out["age_code"].map(AGE_CODE_TO_MIDPOINT).astype("int8")
    out["occupation_label"] = out["occupation_code"].map(OCCUPATION_CODE_TO_LABEL).astype("string")
    out["zip5"] = out["zip"].str.extract(_ZIP_RE, expand=False).astype("string")
    out["is_female"] = (out["gender"] == "F").astype("int8")
    return out[
        [
            "user_id",
            "gender",
            "is_female",
            "age_code",
            "age_label",
            "age_midpoint",
            "occupation_code",
            "occupation_label",
            "zip5",
        ]
    ]


def clean_movies_1m(movies: pd.DataFrame) -> pd.DataFrame:
    out = movies.copy()
    year = out["title"].str.extract(_YEAR_RE, expand=False)
    out["year"] = pd.to_numeric(year, errors="coerce").astype("Int16")
    out["title_clean"] = out["title"].str.replace(_YEAR_RE, "", regex=True).str.strip().astype("string")
    out["genres_list"] = out["genres"].fillna("").apply(
        lambda s: [] if s in ("", "(no genres listed)") else s.split("|")
    )
    out["no_genres"] = out["genres_list"].apply(len).eq(0).astype("int8")
    for g in GENRES_1M:
        out[f"g_{g}"] = out["genres_list"].apply(lambda lst, gg=g: int(gg in lst)).astype("int8")
    return out[
        ["movie_id", "title_clean", "year", "genres", "genres_list", "no_genres"]
        + [f"g_{g}" for g in GENRES_1M]
    ]


def clean_ratings_1m(
    ratings: pd.DataFrame, valid_user_ids: set[int], valid_movie_ids: set[int]
) -> tuple[pd.DataFrame, dict]:
    out = ratings.copy()
    n0 = len(out)
    out = out[out["rating"].between(1, 5)]
    n_rating_ok = len(out)
    user_ok = out["user_id"].isin(valid_user_ids)
    movie_ok = out["movie_id"].isin(valid_movie_ids)
    out = out[user_ok & movie_ok]
    out = out.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    stats = {
        "rows_in": n0,
        "rows_after_rating_filter": n_rating_ok,
        "rows_after_referential_filter": len(out) + int(((~user_ok) | (~movie_ok)).sum()),
        "rows_out": len(out),
        "dropped_bad_rating": n0 - n_rating_ok,
        "dropped_orphans": n_rating_ok - int((user_ok & movie_ok).sum()),
    }
    return out.reset_index(drop=True), stats


def clean_ratings_20m(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = ratings.copy()
    n0 = len(out)
    out = out[out["rating"].between(0.5, 5.0)]
    out = out.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out.reset_index(drop=True), {"rows_in": n0, "rows_out": len(out)}


def clean_tags_20m(tags: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = tags.copy()
    n0 = len(out)
    out["tag"] = out["tag"].astype("string").str.strip().str.lower()
    out = out[out["tag"].notna() & (out["tag"].str.len() > 0)]
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out.reset_index(drop=True), {"rows_in": n0, "rows_out": len(out)}
