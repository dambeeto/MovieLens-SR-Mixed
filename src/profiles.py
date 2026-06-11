"""Read-only profile lookups used by the API: per-user demographics + cluster descriptions."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR

_TOP_N_GENRES = 3


@lru_cache(maxsize=1)
def _users() -> pd.DataFrame:
    path = PROCESSED_DIR / "users_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/02_clean.py first.")
    return pd.read_parquet(
        path, columns=["user_id", "gender", "age_label", "occupation_label"]
    ).set_index("user_id")


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    path = PROCESSED_DIR / "user_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/04_features.py first.")
    return pd.read_parquet(path).set_index("user_id")


def _top_genres(row: pd.Series, prefix: str) -> list[str]:
    """Top genres by a prefixed feature group (rshare_/rmean_), dropping zeros."""
    cols = [c for c in row.index if c.startswith(prefix)]
    series = row[cols][row[cols] > 0].sort_values(ascending=False)
    return [c[len(prefix):] for c in series.index[:_TOP_N_GENRES]]


def get_user_profile(user_id: int) -> dict | None:
    """Return {user_id, gender, age, occupation, top_watched, top_rated} or None if unknown."""
    users = _users()
    uid = int(user_id)
    if uid not in users.index:
        return None
    u = users.loc[uid]
    profile: dict = {
        "user_id": uid,
        "gender": "Female" if str(u["gender"]).upper().startswith("F") else "Male",
        "age": str(u["age_label"]),
        "occupation": str(u["occupation_label"]),
        "top_watched": [],
        "top_rated": [],
    }
    feats = _features()
    if uid in feats.index:
        row = feats.loc[uid]
        profile["top_watched"] = _top_genres(row, "rshare_")
        profile["top_rated"] = _top_genres(row, "rmean_")
    return profile


@lru_cache(maxsize=1)
def _cluster_profiles() -> pd.DataFrame:
    path = REPORTS_DIR / "cluster_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/05_cluster.py first.")
    return pd.read_csv(path).set_index("cluster")


def get_cluster_profile(cluster_id: int) -> dict | None:
    """Return a human-readable description for a KMeans cluster, or None if unknown."""
    profiles = _cluster_profiles()
    cid = int(cluster_id)
    if cid not in profiles.index:
        return None
    p = profiles.loc[cid]
    female_share = float(p["female_share"])
    return {
        "cluster_id": cid,
        "size": int(p["size"]),
        "top_genres": [g.strip() for g in str(p["top5_genres_by_mean_rating"]).split(",")],
        "modal_age_bucket": str(p["modal_age_bucket"]),
        "modal_occupation": str(p["modal_occupation"]),
        "female_share": female_share,
        "typical_gender": "Female" if female_share > 0.5 else "Male",
    }
