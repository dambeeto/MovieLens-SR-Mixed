"""Train + walk-forward evaluate quarterly popularity regressors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import POPULARITY_TEST_QUARTERS, RANDOM_STATE
from src.popularity.features import TARGET, feature_columns


def _model_zoo() -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


@dataclass
class PopularityResult:
    best_model_name: str
    best_model: object
    metrics: pd.DataFrame
    last_train_period: pd.Timestamp


def _eval(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    # MAPE on log-scale -> compute on original scale (expm1) for interpretability.
    actual = np.expm1(y)
    pred_orig = np.maximum(np.expm1(pred), 0)
    mape = float(np.mean(np.abs(actual - pred_orig) / np.maximum(actual, 1)))
    return {"rmse_log": rmse, "mae_log": mae, "mape_original": mape}


def train_and_evaluate(features_df: pd.DataFrame) -> PopularityResult:
    """Hold out the last POPULARITY_TEST_QUARTERS distinct quarters as a test set.

    All genres share the same model (genre is implicitly encoded through lag features
    and the seasonal cycle). This is the simplest setup that still produces per-genre
    forecasts because predictions are conditional on each genre's own lags.
    """
    all_periods = np.sort(features_df["period"].unique())
    if len(all_periods) <= POPULARITY_TEST_QUARTERS + 4:
        raise ValueError(
            f"need >{POPULARITY_TEST_QUARTERS + 4} distinct quarters; got {len(all_periods)}"
        )
    cut = all_periods[-POPULARITY_TEST_QUARTERS]
    train_mask = features_df["period"] < cut
    test_mask = features_df["period"] >= cut
    cols = feature_columns()
    X_train = features_df.loc[train_mask, cols].to_numpy()
    y_train = features_df.loc[train_mask, TARGET].to_numpy()
    X_test = features_df.loc[test_mask, cols].to_numpy()
    y_test = features_df.loc[test_mask, TARGET].to_numpy()

    rows = []
    fitted: dict[str, object] = {}
    print(f"[pop] train n={len(X_train)}  test n={len(X_test)}  (cut={pd.Timestamp(cut).date()})")
    for name, model in _model_zoo().items():
        model.fit(X_train, y_train)
        train_score = _eval(model, X_train, y_train)
        test_score = _eval(model, X_test, y_test)
        rows.append(
            {
                "model": name,
                "rmse_log_train": train_score["rmse_log"],
                "rmse_log_test": test_score["rmse_log"],
                "mae_log_test": test_score["mae_log"],
                "mape_test": test_score["mape_original"],
            }
        )
        fitted[name] = model
        print(f"  {name:14s} rmse_log_test={test_score['rmse_log']:.4f}  mape={test_score['mape_original']:.3f}")

    metrics = pd.DataFrame(rows).sort_values("rmse_log_test").reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    print(f"[pop] selected: {best_name}")
    return PopularityResult(
        best_model_name=best_name,
        best_model=fitted[best_name],
        metrics=metrics,
        last_train_period=pd.Timestamp(all_periods[-1]),
    )
