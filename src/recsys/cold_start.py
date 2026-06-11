"""Cold-start recommendations: global popularity + cluster-based picks (no model needed)."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import PROCESSED_DIR, RECSYS_TOP_N
from src.recsys.predict import _load_movies


@lru_cache(maxsize=1)
def _ratings() -> pd.DataFrame:
    path = PROCESSED_DIR / "ratings_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/02_clean.py first.")
    return pd.read_parquet(path, columns=["user_id", "movie_id", "rating"])


@lru_cache(maxsize=1)
def _user_clusters() -> pd.DataFrame:
    path = PROCESSED_DIR / "user_clusters.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/05_cluster.py first.")
    return pd.read_parquet(path, columns=["user_id", "cluster_id"])


def _rank(ratings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ratings into a popularity ranking (most-rated first, mean as tie-break)."""
    agg = ratings.groupby("movie_id")["rating"].agg(n_ratings="size", mean_rating="mean")
    return agg.sort_values(["n_ratings", "mean_rating"], ascending=False)


def _to_recs(ranking: pd.DataFrame, n: int) -> list[dict]:
    movies = _load_movies().set_index("movie_id")
    out: list[dict] = []
    for movie_id, r in ranking.head(int(n)).iterrows():
        row = movies.loc[movie_id] if movie_id in movies.index else None
        out.append(
            {
                "movie_id": int(movie_id),
                "title": str(row["title_clean"]) if row is not None else "?",
                "year": int(row["year"]) if row is not None and pd.notna(row["year"]) else None,
                "predicted_rating": round(float(r["mean_rating"]), 2),
            }
        )
    return out


@lru_cache(maxsize=1)
def _popularity_ranking() -> pd.DataFrame:
    return _rank(_ratings())


def popular_top_n(n: int = RECSYS_TOP_N) -> list[dict]:
    """Most popular movies overall -- cold-start fallback for an unknown user."""
    return _to_recs(_popularity_ranking(), n)


@lru_cache(maxsize=32)
def _cluster_ranking(cluster_id: int) -> pd.DataFrame:
    clusters = _user_clusters()
    members = clusters.loc[clusters["cluster_id"] == int(cluster_id), "user_id"]
    member_ratings = _ratings()[_ratings()["user_id"].isin(members)]
    return _rank(member_ratings)


def recommend_for_cluster(cluster_id: int, n: int = RECSYS_TOP_N) -> list[dict]:
    """Top movies among the members of a cluster -- cold-start for a brand-new profile."""
    return _to_recs(_cluster_ranking(int(cluster_id)), n)
