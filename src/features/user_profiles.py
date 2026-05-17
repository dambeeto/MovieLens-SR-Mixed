"""Build user feature matrix: per-genre mean rating + per-genre rating count + demographics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GENRES_1M, OCCUPATION_CODE_TO_LABEL


def _genre_mean_and_count(
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (user, genre): mean rating and count of ratings touching that genre."""
    genre_cols = [f"g_{g}" for g in GENRES_1M]
    merged = ratings.merge(movies[["movie_id"] + genre_cols], on="movie_id", how="inner")

    # Weighted sums per user x genre. Multiply rating by genre-indicator (0/1).
    long_frames_sum = {}
    long_frames_cnt = {}
    for g in GENRES_1M:
        mask = merged[f"g_{g}"] == 1
        sub = merged.loc[mask, ["user_id", "rating"]]
        if sub.empty:
            long_frames_sum[g] = pd.Series(dtype="float64", name=f"rmean_{g}")
            long_frames_cnt[g] = pd.Series(dtype="int64", name=f"rcount_{g}")
            continue
        grp = sub.groupby("user_id")["rating"]
        long_frames_sum[g] = grp.mean().rename(f"rmean_{g}")
        long_frames_cnt[g] = grp.size().rename(f"rcount_{g}")

    mean_df = pd.concat(long_frames_sum.values(), axis=1).fillna(0.0).astype("float32")
    count_df = pd.concat(long_frames_cnt.values(), axis=1).fillna(0).astype("int32")
    mean_df.index.name = "user_id"
    count_df.index.name = "user_id"
    return mean_df, count_df


def build_user_features(
    users: pd.DataFrame, ratings: pd.DataFrame, movies: pd.DataFrame
) -> pd.DataFrame:
    """Return DataFrame indexed by user_id with ~59 feature columns ready for scaling + KMeans."""
    mean_df, count_df = _genre_mean_and_count(ratings, movies)

    # Normalize the rating-count signal per user so it expresses *proportion of attention*.
    totals = count_df.sum(axis=1).replace(0, 1)
    cnt_norm = count_df.div(totals, axis=0).astype("float32")
    cnt_norm.columns = [c.replace("rcount_", "rshare_") for c in cnt_norm.columns]

    demo = users.set_index("user_id")[
        ["age_midpoint", "is_female", "occupation_code", "occupation_label"]
    ].copy()
    demo["age_midpoint"] = demo["age_midpoint"].astype("float32")
    demo["is_female"] = demo["is_female"].astype("int8")

    occ_dummies = pd.get_dummies(
        demo["occupation_code"].map(OCCUPATION_CODE_TO_LABEL),
        prefix="occ",
    ).astype("int8")
    # Guarantee every known occupation column exists (in case some are absent in this slice).
    for label in OCCUPATION_CODE_TO_LABEL.values():
        col = f"occ_{label}"
        if col not in occ_dummies.columns:
            occ_dummies[col] = 0
    occ_dummies = occ_dummies.sort_index(axis=1)

    features = (
        mean_df.join(cnt_norm, how="outer")
        .join(demo[["age_midpoint", "is_female"]], how="left")
        .join(occ_dummies, how="left")
    )
    # Users with zero ratings in the training slice -> fill numeric/genre cols with 0.
    features = features.fillna(0)
    features.index = features.index.astype("int32")
    features = features.sort_index()
    return features
