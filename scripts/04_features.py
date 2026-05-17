"""Build the user-feature matrix used for KMeans clustering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PROCESSED_DIR, ensure_dirs
from src.features.user_profiles import build_user_features


def main() -> None:
    ensure_dirs()
    users = pd.read_parquet(PROCESSED_DIR / "users_1m.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies_1m.parquet")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    print(f"[features] users={len(users)} movies={len(movies)} train_ratings={len(train)}")
    feats = build_user_features(users, train, movies)
    feats.reset_index().to_parquet(PROCESSED_DIR / "user_features.parquet", index=False)
    print(f"[features] wrote user_features.parquet shape={feats.shape}")


if __name__ == "__main__":
    main()
