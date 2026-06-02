"""Load ml-20m movies + aggregate (genre, quarter) popularity series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ML_20M_DIR


def load_movies_20m(path: Path | None = None) -> pd.DataFrame:
    """ml-20m movies.csv -> (movie_id, title, year, genres_list[]).

    Genres are pipe-separated; movies with '(no genres listed)' get an empty list.
    """
    p = path or (ML_20M_DIR / "movies.csv")
    df = pd.read_csv(p, dtype={"movieId": "int32", "title": "string", "genres": "string"})
    df = df.rename(columns={"movieId": "movie_id"})
    year = df["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    df["year"] = pd.to_numeric(year, errors="coerce").astype("Int16")
    df["genres_list"] = df["genres"].fillna("").apply(
        lambda s: [] if s in ("", "(no genres listed)") else s.split("|")
    )
    return df[["movie_id", "title", "year", "genres_list"]]


def aggregate_by_genre_quarter(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Returns long-format DataFrame: (genre, period, n_ratings, mean_rating).

    `period` is a quarter end Timestamp (e.g. 1999Q4 -> 1999-12-31). Series is dense:
    every (genre, quarter) within the observed range gets a row, missing pairs filled with 0.
    """
    # Explode genres so each (movie, genre) gets its own row.
    mg = movies[["movie_id", "genres_list"]].explode("genres_list")
    mg = mg.rename(columns={"genres_list": "genre"}).dropna(subset=["genre"])
    mg = mg[mg["genre"].str.len() > 0]

    df = ratings[["movie_id", "rating", "timestamp"]].merge(mg, on="movie_id", how="inner")
    df["period"] = pd.to_datetime(df["timestamp"], utc=True).dt.to_period("Q").dt.to_timestamp(how="end").dt.floor("D")

    agg = (
        df.groupby(["genre", "period"], observed=True)
        .agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"))
        .reset_index()
    )

    # Dense grid (genre x quarter) so lag features have stable indexing.
    all_periods = pd.date_range(agg["period"].min(), agg["period"].max(), freq="QE-DEC")
    all_periods = all_periods.normalize()
    genres = agg["genre"].unique()
    grid = pd.MultiIndex.from_product([genres, all_periods], names=["genre", "period"]).to_frame(index=False)
    out = grid.merge(agg, on=["genre", "period"], how="left")
    out["n_ratings"] = out["n_ratings"].fillna(0).astype("int64")
    out["mean_rating"] = out["mean_rating"].astype("float32")
    return out.sort_values(["genre", "period"]).reset_index(drop=True)
