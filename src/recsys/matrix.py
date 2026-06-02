"""Build sparse user x item rating matrix from a ratings DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    rating_col: str = "rating",
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    """Return (csr_matrix shape=(n_users, n_items), user_id->row, item_id->col).

    Rows / columns are dense indices over the unique IDs present in ``ratings``,
    sorted ascending so the indexing is deterministic across runs.
    """
    required = {user_col, item_col, rating_col}
    if not required.issubset(ratings.columns):
        missing = required - set(ratings.columns)
        raise KeyError(f"missing columns: {missing}")

    users = np.sort(ratings[user_col].unique())
    items = np.sort(ratings[item_col].unique())
    user_to_idx = {int(u): i for i, u in enumerate(users)}
    item_to_idx = {int(m): i for i, m in enumerate(items)}

    row = ratings[user_col].map(user_to_idx).to_numpy()
    col = ratings[item_col].map(item_to_idx).to_numpy()
    val = ratings[rating_col].to_numpy(dtype="float32")

    mat = csr_matrix((val, (row, col)), shape=(len(users), len(items)))
    return mat, user_to_idx, item_to_idx
