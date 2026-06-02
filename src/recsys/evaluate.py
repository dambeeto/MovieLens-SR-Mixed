"""Evaluation metrics for collaborative-filtering recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RECSYS_LIKE_THRESHOLD, RECSYS_TOP_N


def _known_pairs(model, test: pd.DataFrame) -> pd.DataFrame:
    """Filter to (user, item) pairs where both id sides are known to the model."""
    mask = (
        test["user_id"].isin(model.user_to_idx.keys())
        & test["movie_id"].isin(model.item_to_idx.keys())
    )
    return test.loc[mask].copy()


def rmse_mae(model, test: pd.DataFrame) -> dict[str, float]:
    """Rating-prediction error on the held-out test set (cold pairs dropped)."""
    pairs = _known_pairs(model, test)
    if pairs.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "n_pairs": 0}
    y = pairs["rating"].to_numpy(dtype=np.float32)
    y_hat = model.predict_batch(pairs[["user_id", "movie_id"]])
    err = y - y_hat
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "mae": mae, "n_pairs": int(len(pairs))}


def hit_rate_at_k(
    model,
    test: pd.DataFrame,
    k: int = RECSYS_TOP_N,
    like_threshold: float = RECSYS_LIKE_THRESHOLD,
    max_users: int | None = None,
) -> dict[str, float]:
    """Fraction of evaluable users for whom at least one liked test item appears in top-K.

    Only users with >=1 "liked" test item (rating >= ``like_threshold``) are considered, since
    users without any positives in the test set have no possible hit by construction.
    Items already seen in the training matrix are excluded from candidate lists.
    """
    pairs = _known_pairs(model, test)
    liked = pairs[pairs["rating"] >= like_threshold]
    if liked.empty:
        return {"hit_rate_at_k": float("nan"), "n_users_eval": 0}

    grouped = liked.groupby("user_id")["movie_id"].apply(set)
    user_ids = list(grouped.index)
    if max_users is not None and max_users < len(user_ids):
        rng = np.random.default_rng(0)
        user_ids = list(rng.choice(user_ids, size=max_users, replace=False))

    hits = 0
    for uid in user_ids:
        top = model.top_n(int(uid), n=k, exclude_seen=True)
        if not top:
            continue
        recs = {m_id for m_id, _ in top}
        if recs & grouped[uid]:
            hits += 1
    return {"hit_rate_at_k": hits / max(len(user_ids), 1), "n_users_eval": len(user_ids)}
