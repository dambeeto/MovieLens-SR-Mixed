"""Time-series feature engineering for per-genre quarterly popularity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import POPULARITY_LAGS

TARGET = "n_ratings_log"
SEASONAL_FEATURES = ("year_idx", "q_sin", "q_cos")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """log1p of n_ratings stabilises variance across genres (3 orders of magnitude span)."""
    out = df.copy()
    out[TARGET] = np.log1p(out["n_ratings"]).astype("float32")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-genre features ready for regression. Drops rows with missing lag values."""
    out = add_target(df)
    out = out.sort_values(["genre", "period"]).reset_index(drop=True)

    # Calendar features. year_idx is relative to the first observed quarter; q in {1..4}.
    first_year = int(out["period"].dt.year.min())
    out["year_idx"] = (out["period"].dt.year - first_year).astype("int16")
    q = out["period"].dt.quarter.astype("int8")
    out["q_sin"] = np.sin(2 * np.pi * q / 4).astype("float32")
    out["q_cos"] = np.cos(2 * np.pi * q / 4).astype("float32")

    # Autoregressive lags per genre.
    for lag in POPULARITY_LAGS:
        out[f"lag_{lag}"] = out.groupby("genre", observed=True)[TARGET].shift(lag).astype("float32")
    out["roll_mean_4"] = (
        out.groupby("genre", observed=True)[TARGET]
        .shift(1)
        .rolling(window=4, min_periods=2)
        .mean()
        .reset_index(drop=True)
        .astype("float32")
    )

    feature_cols = list(SEASONAL_FEATURES) + [f"lag_{l}" for l in POPULARITY_LAGS] + ["roll_mean_4"]
    out["_keep"] = out[feature_cols].notna().all(axis=1)
    return out[out["_keep"]].drop(columns="_keep").reset_index(drop=True)


def feature_columns() -> list[str]:
    return list(SEASONAL_FEATURES) + [f"lag_{l}" for l in POPULARITY_LAGS] + ["roll_mean_4"]
