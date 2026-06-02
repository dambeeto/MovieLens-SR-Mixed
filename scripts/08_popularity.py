"""End-to-end genre popularity forecasting (ml-20m, quarterly)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    MODELS_POPULARITY_DIR,
    PROCESSED_DIR,
    REPORTS_POPULARITY_DIR,
    ensure_dirs,
)
from src.popularity.aggregate import aggregate_by_genre_quarter, load_movies_20m
from src.popularity.features import TARGET, build_features, feature_columns
from src.popularity.predict import forecast_all_genres
from src.popularity.train import train_and_evaluate


def _plot_top_genres(history: pd.DataFrame, forecast_df: pd.DataFrame, top_k: int = 8) -> None:
    totals = history.groupby("genre", observed=True)["n_ratings"].sum().sort_values(ascending=False)
    top = list(totals.head(top_k).index)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for ax, genre in zip(axes.ravel(), top):
        h = history[history["genre"] == genre]
        ax.plot(h["period"], h["n_ratings"], color="steelblue", linewidth=1.2, label="actual")
        if genre in forecast_df["genre"].values:
            fr = forecast_df[forecast_df["genre"] == genre].iloc[0]
            ax.scatter([pd.Timestamp(fr["next_period"])], [fr["predicted_n_ratings"]],
                       color="crimson", s=40, zorder=3, label="forecast")
        ax.set_title(genre)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Genre popularity per quarter -- actual + next quarter forecast", y=1.01)
    fig.tight_layout()
    fig.savefig(REPORTS_POPULARITY_DIR / "time_series_per_genre.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    ratings_path = PROCESSED_DIR / "ratings_20m.parquet"
    if not ratings_path.exists():
        raise FileNotFoundError(f"{ratings_path} missing -- run scripts/02_clean.py first")
    print(f"[pop] loading {ratings_path}")
    ratings = pd.read_parquet(ratings_path, columns=["movie_id", "rating", "timestamp"])

    print("[pop] loading ml-20m movies.csv")
    movies = load_movies_20m()
    print(f"[pop] ratings={len(ratings):,} movies={len(movies):,}")

    print("[pop] aggregating by (genre, quarter)")
    history = aggregate_by_genre_quarter(ratings, movies)
    print(f"[pop] aggregated rows={len(history)}  quarters={history['period'].nunique()}  genres={history['genre'].nunique()}")

    # Drop the most recent quarter if its rating count is suspiciously low -- ml-20m ends
    # mid-quarter (March 2015) so the final point is an apparent crash, not a real trend.
    max_period = history["period"].max()
    final = history[history["period"] == max_period]
    prior_avg = history[history["period"] < max_period].groupby("genre", observed=True)["n_ratings"].mean()
    if (final.set_index("genre")["n_ratings"] < 0.5 * prior_avg).mean() > 0.5:
        print(f"[pop] dropping partial final quarter {max_period.date()} (low completeness)")
        history = history[history["period"] < max_period].copy()

    features_df = build_features(history)
    print(f"[pop] feature rows after lag drop={len(features_df)}")

    result = train_and_evaluate(features_df)
    result.metrics.to_csv(REPORTS_POPULARITY_DIR / "model_comparison.csv", index=False)
    print(f"[pop] wrote {REPORTS_POPULARITY_DIR / 'model_comparison.csv'}")

    # Persist artefacts.
    history.rename(columns={})  # noop, but keep original column names
    history_for_predict = history[["genre", "period", "n_ratings"]].copy()
    history_for_predict["n_ratings_log"] = np.log1p(history_for_predict["n_ratings"]).astype("float32")
    history_for_predict.to_parquet(MODELS_POPULARITY_DIR / "history.parquet", index=False)
    joblib.dump(result.best_model, MODELS_POPULARITY_DIR / "best_model.joblib")
    joblib.dump(
        {
            "model_name": result.best_model_name,
            "first_year": int(history["period"].dt.year.min()),
            "feature_columns": feature_columns(),
        },
        MODELS_POPULARITY_DIR / "meta.joblib",
    )
    print(f"[pop] persisted model + history + meta to {MODELS_POPULARITY_DIR}")

    # Forecast next quarter per genre.
    forecasts = forecast_all_genres()
    fc_df = pd.DataFrame(forecasts).sort_values("predicted_n_ratings", ascending=False)
    fc_df.to_csv(REPORTS_POPULARITY_DIR / "forecast_next.csv", index=False)
    print(f"[pop] forecast for next quarter ({fc_df.iloc[0]['next_period']}):")
    print(fc_df.head(8).to_string(index=False))

    _plot_top_genres(history, fc_df, top_k=8)
    print(f"[pop] wrote time_series_per_genre.png")


if __name__ == "__main__":
    main()
