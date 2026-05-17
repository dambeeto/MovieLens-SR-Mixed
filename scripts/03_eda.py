"""Exploratory plots: ratings distribution, user demographics, genres, time series."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import GENRES_1M, PROCESSED_DIR, REPORTS_DIR, ensure_dirs


def plot_ratings_distribution(ratings_1m: pd.DataFrame, ratings_20m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x="rating", data=ratings_1m, ax=axes[0], color="steelblue")
    axes[0].set_title(f"MovieLens 1M -- ratings (n={len(ratings_1m):,})")
    axes[0].set_xlabel("rating")
    sns.countplot(x="rating", data=ratings_20m, ax=axes[1], color="darkorange")
    axes[1].set_title(f"MovieLens 20M -- ratings (n={len(ratings_20m):,})")
    axes[1].set_xlabel("rating")
    for label in axes[1].get_xticklabels():
        label.set_rotation(45)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ratings_distribution.png", dpi=110)
    plt.close(fig)


def plot_users_demographics(users: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    order_age = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    sns.countplot(x="age_label", data=users, order=order_age, ax=axes[0], color="steelblue")
    axes[0].set_title("Age groups")
    axes[0].set_xlabel("")
    for label in axes[0].get_xticklabels():
        label.set_rotation(30)

    sns.countplot(x="gender", data=users, ax=axes[1], color="steelblue")
    axes[1].set_title("Gender")
    axes[1].set_xlabel("")

    occ_order = users["occupation_label"].value_counts().index.tolist()
    sns.countplot(y="occupation_label", data=users, order=occ_order, ax=axes[2], color="steelblue")
    axes[2].set_title("Occupation")
    axes[2].set_ylabel("")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "users_demographics.png", dpi=110)
    plt.close(fig)


def plot_genres_popularity(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    counts = {g: int(movies[f"g_{g}"].sum()) for g in GENRES_1M}
    mr = ratings.merge(movies[["movie_id"] + [f"g_{g}" for g in GENRES_1M]], on="movie_id")
    mean_rating = {}
    for g in GENRES_1M:
        mask = mr[f"g_{g}"] == 1
        mean_rating[g] = float(mr.loc[mask, "rating"].mean()) if mask.any() else float("nan")

    df = (
        pd.DataFrame({"genre": GENRES_1M})
        .assign(n_movies=lambda d: d["genre"].map(counts), mean_rating=lambda d: d["genre"].map(mean_rating))
        .sort_values("n_movies", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x="n_movies", y="genre", data=df, ax=axes[0], color="steelblue")
    axes[0].set_title("Number of movies per genre (ml-1m)")
    sns.barplot(x="mean_rating", y="genre", data=df, ax=axes[1], color="darkorange")
    axes[1].set_title("Mean rating per genre (ml-1m)")
    axes[1].set_xlim(2.5, 5)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "genres_popularity.png", dpi=110)
    plt.close(fig)


def plot_ratings_over_time(ratings: pd.DataFrame) -> None:
    by_month = (
        ratings.assign(month=ratings["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    by_month.plot(ax=ax, color="steelblue")
    ax.set_title("MovieLens 1M -- ratings per month")
    ax.set_ylabel("ratings count")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ratings_over_time.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    users = pd.read_parquet(PROCESSED_DIR / "users_1m.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies_1m.parquet")
    ratings_1m = pd.read_parquet(PROCESSED_DIR / "ratings_1m.parquet")
    ratings_20m = pd.read_parquet(PROCESSED_DIR / "ratings_20m.parquet")

    print("[eda] ratings distribution")
    plot_ratings_distribution(ratings_1m, ratings_20m)
    print("[eda] user demographics")
    plot_users_demographics(users)
    print("[eda] genres popularity")
    plot_genres_popularity(movies, ratings_1m)
    print("[eda] ratings over time")
    plot_ratings_over_time(ratings_1m)
    print(f"[done] plots in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
