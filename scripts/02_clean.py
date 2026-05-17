"""Load raw MovieLens 1M + 20M, clean, report missing values, write parquet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, TRAIN_RATIO, ensure_dirs
from src.data import load_1m, load_20m
from src.data.clean import (
    clean_movies_1m,
    clean_ratings_1m,
    clean_ratings_20m,
    clean_tags_20m,
    clean_users_1m,
    report_missing,
)
from src.data.split import time_split


def main() -> None:
    ensure_dirs()
    missing_reports: list[pd.DataFrame] = []

    print("[ml-1m] loading raw .dat files")
    users_raw = load_1m.load_users()
    movies_raw = load_1m.load_movies()
    ratings_raw = load_1m.load_ratings()
    print(f"        users={len(users_raw)} movies={len(movies_raw)} ratings={len(ratings_raw)}")

    missing_reports += [
        report_missing(users_raw, "ml-1m/users.raw"),
        report_missing(movies_raw, "ml-1m/movies.raw"),
        report_missing(ratings_raw, "ml-1m/ratings.raw"),
    ]

    users = clean_users_1m(users_raw)
    movies = clean_movies_1m(movies_raw)
    ratings, rstats = clean_ratings_1m(
        ratings_raw, set(users["user_id"]), set(movies["movie_id"])
    )
    print(f"[ml-1m] ratings clean stats: {rstats}")

    missing_reports += [
        report_missing(users, "ml-1m/users.clean"),
        report_missing(movies, "ml-1m/movies.clean"),
        report_missing(ratings, "ml-1m/ratings.clean"),
    ]

    train, test = time_split(ratings, ratio=TRAIN_RATIO)
    print(f"[ml-1m] split train={len(train)} test={len(test)}  (ratio={TRAIN_RATIO})")

    users.to_parquet(PROCESSED_DIR / "users_1m.parquet", index=False)
    # genres_list is list[str] -> ok for parquet via pyarrow.
    movies.to_parquet(PROCESSED_DIR / "movies_1m.parquet", index=False)
    ratings.to_parquet(PROCESSED_DIR / "ratings_1m.parquet", index=False)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print("[ml-20m] loading ratings.csv (this takes a moment)")
    ratings20_raw = load_20m.load_ratings()
    missing_reports.append(report_missing(ratings20_raw, "ml-20m/ratings.raw"))
    ratings20, r20stats = clean_ratings_20m(ratings20_raw)
    print(f"[ml-20m] ratings clean stats: {r20stats}")
    missing_reports.append(report_missing(ratings20, "ml-20m/ratings.clean"))
    ratings20.to_parquet(PROCESSED_DIR / "ratings_20m.parquet", index=False)
    del ratings20_raw, ratings20

    print("[ml-20m] loading tags.csv")
    tags20_raw = load_20m.load_tags()
    missing_reports.append(report_missing(tags20_raw, "ml-20m/tags.raw"))
    tags20, tstats = clean_tags_20m(tags20_raw)
    print(f"[ml-20m] tags clean stats: {tstats}")
    missing_reports.append(report_missing(tags20, "ml-20m/tags.clean"))
    tags20.to_parquet(PROCESSED_DIR / "tags_20m.parquet", index=False)

    miss = pd.concat(missing_reports, ignore_index=True)
    miss.to_csv(REPORTS_DIR / "missing_report.csv", index=False)
    print(f"[done] wrote parquet to {PROCESSED_DIR} and missing report to {REPORTS_DIR}/missing_report.csv")


if __name__ == "__main__":
    main()
