"""Forecast next-quarter popularity per genre using the persisted regressor."""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_POPULARITY_DIR, POPULARITY_LAGS
from src.popularity.features import feature_columns

MODEL_PATH = MODELS_POPULARITY_DIR / "best_model.joblib"
HISTORY_PATH = MODELS_POPULARITY_DIR / "history.parquet"
META_PATH = MODELS_POPULARITY_DIR / "meta.joblib"


@lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.exists() or not HISTORY_PATH.exists():
        raise FileNotFoundError(
            "Popularity model missing -- run scripts/08_popularity.py first."
        )
    return {
        "model": joblib.load(MODEL_PATH),
        "history": pd.read_parquet(HISTORY_PATH),
        "meta": joblib.load(META_PATH),
    }


def _build_next_row(history_g: pd.DataFrame, next_period: pd.Timestamp, first_year: int) -> dict:
    last = history_g.iloc[-1]
    row = {
        "year_idx": int(next_period.year - first_year),
        "q_sin": float(np.sin(2 * np.pi * next_period.quarter / 4)),
        "q_cos": float(np.cos(2 * np.pi * next_period.quarter / 4)),
        "roll_mean_4": float(history_g["n_ratings_log"].iloc[-4:].mean()),
    }
    for lag in POPULARITY_LAGS:
        if len(history_g) >= lag:
            row[f"lag_{lag}"] = float(history_g["n_ratings_log"].iloc[-lag])
        else:
            row[f"lag_{lag}"] = float(history_g["n_ratings_log"].mean())
    return row


def forecast_next_quarter(genre: str) -> dict:
    """Return {genre, next_period, predicted_n_ratings, predicted_log} for a single genre."""
    state = _load()
    history = state["history"]
    if genre not in history["genre"].unique():
        raise KeyError(f"genre {genre!r} not in history; known: {sorted(history['genre'].unique())}")
    history_g = history[history["genre"] == genre].sort_values("period").reset_index(drop=True)
    last_period = history_g["period"].iloc[-1]
    last_n = int(history_g["n_ratings"].iloc[-1])
    next_period = (last_period + pd.tseries.offsets.QuarterEnd()).normalize()
    row = _build_next_row(history_g, next_period, state["meta"]["first_year"])
    X = np.array([[row[c] for c in feature_columns()]])
    pred_log = float(state["model"].predict(X)[0])
    pred_n = int(round(max(np.expm1(pred_log), 0.0)))
    trend_pct = round((pred_n - last_n) / last_n * 100, 1) if last_n else None
    return {
        "genre": genre,
        "next_period": next_period.date().isoformat(),
        "predicted_log": round(pred_log, 4),
        "predicted_n_ratings": pred_n,
        "last_period": last_period.date().isoformat(),
        "last_quarter_n_ratings": last_n,
        "trend_pct": trend_pct,
        "based_on_model": state["meta"].get("model_name", "?"),
    }


def forecast_all_genres() -> list[dict]:
    state = _load()
    return [forecast_next_quarter(g) for g in sorted(state["history"]["genre"].unique())]
