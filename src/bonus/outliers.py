"""Outlier detection on user features via IsolationForest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import OUTLIER_CONTAMINATION, RANDOM_STATE


def detect_outliers(features: pd.DataFrame, user_id_col: str = "user_id") -> pd.DataFrame:
    """Add `is_outlier` (0/1) and `anomaly_score` (lower = more anomalous) columns.

    Features are standardised before fitting -- otherwise high-variance columns (e.g.
    rating counts) dominate the split criteria.
    """
    if user_id_col not in features.columns:
        raise KeyError(f"missing user id column {user_id_col!r}")
    X = features.drop(columns=[user_id_col]).select_dtypes(include="number").to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=OUTLIER_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preds = iso.fit_predict(Xs)            # 1=inlier, -1=outlier
    scores = iso.score_samples(Xs)         # higher = more normal

    out = features[[user_id_col]].copy()
    out["is_outlier"] = (preds == -1).astype("int8")
    out["anomaly_score"] = scores.astype("float32")
    return out, iso, scaler
