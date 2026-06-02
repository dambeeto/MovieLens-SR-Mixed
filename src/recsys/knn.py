"""Item-based KNN collaborative filtering with selectable similarity measure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RECSYS_KNN_NEIGHBORS
from src.recsys.matrix import build_user_item_matrix

ALLOWED_SIM = ("cosine", "pearson")


class ItemKNNRecommender:
    """Item-item KNN with cosine or Pearson similarity.

    - cosine: similarity on raw rating columns (only co-rated entries contribute via dot product)
    - pearson: similarity on user-mean-centered columns; algebraically this is centered cosine,
      i.e. closer to classical Pearson correlation among users who rated both items.

    Prediction for (u, i): weighted average of u's known ratings on neighbours of i, plus the
    user bias to bring it back into the rating scale.
    """

    def __init__(self, similarity: str = "cosine", k: int = RECSYS_KNN_NEIGHBORS) -> None:
        if similarity not in ALLOWED_SIM:
            raise ValueError(f"similarity must be one of {ALLOWED_SIM}, got {similarity!r}")
        self.similarity = similarity
        self.k = int(k)
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: np.ndarray | None = None
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.matrix: csr_matrix | None = None         # raw ratings as csr (rows=users, cols=items)
        self.matrix_centered: csr_matrix | None = None  # user-mean-centered (for pearson)
        self.sim: np.ndarray | None = None            # dense (n_items, n_items) top-k pruned
        self.seen: dict[int, set[int]] = {}

    # --- Fit ---

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        mat, user_to_idx, item_to_idx = build_user_item_matrix(train)
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_item = np.array(sorted(item_to_idx, key=item_to_idx.get))
        self.matrix = mat

        nnz_mask = mat.copy()
        nnz_mask.data = np.ones_like(nnz_mask.data)
        ratings_sum = np.asarray(mat.sum(axis=1)).ravel()
        ratings_cnt = np.asarray(nnz_mask.sum(axis=1)).ravel()
        self.global_mean = float(mat.data.mean())
        user_means = ratings_sum / np.maximum(ratings_cnt, 1)
        self.user_bias = user_means - self.global_mean

        coo = mat.tocoo()
        centered = csr_matrix(
            (coo.data - user_means[coo.row], (coo.row, coo.col)), shape=mat.shape
        )
        self.matrix_centered = centered

        # Similarity is item-item -> work on columns.
        if self.similarity == "cosine":
            items = mat.T.tocsr()
        else:  # pearson via centered cosine
            items = centered.T.tocsr()

        print(f"[knn-{self.similarity}] computing item-item similarity on {items.shape[0]} items")
        sim = cosine_similarity(items, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)

        # Keep only top-k neighbours per row; everything else -> 0.
        k = min(self.k, sim.shape[1] - 1)
        if k < sim.shape[1] - 1:
            kth_idx = np.argpartition(-sim, kth=k, axis=1)[:, k:]
            for i in range(sim.shape[0]):
                sim[i, kth_idx[i]] = 0.0
        self.sim = sim

        self.seen = {}
        train_rows = train[["user_id", "movie_id"]].itertuples(index=False)
        for u_id, m_id in train_rows:
            u_idx = self.user_to_idx.get(int(u_id))
            m_idx = self.item_to_idx.get(int(m_id))
            if u_idx is None or m_idx is None:
                continue
            self.seen.setdefault(u_idx, set()).add(m_idx)
        return self

    # --- Prediction ---

    def _check_fit(self) -> None:
        if self.sim is None or self.matrix_centered is None:
            raise RuntimeError("ItemKNNRecommender is not fitted")

    def _user_centered_row(self, u_idx: int) -> np.ndarray:
        # Dense row of mean-centered ratings for u (zeros where unobserved -> no contribution).
        row = self.matrix_centered.getrow(u_idx).toarray().ravel()
        return row

    def predict(self, user_id: int, item_id: int) -> float:
        self._check_fit()
        u_idx = self.user_to_idx.get(int(user_id))
        m_idx = self.item_to_idx.get(int(item_id))
        if u_idx is None or m_idx is None:
            return float(self.global_mean)
        sims = self.sim[m_idx]
        user_row = self._user_centered_row(u_idx)
        mask = user_row != 0
        if not mask.any():
            return float(self.global_mean + self.user_bias[u_idx])
        weights = sims[mask]
        if not np.any(weights != 0):
            return float(self.global_mean + self.user_bias[u_idx])
        num = float(np.dot(weights, user_row[mask]))
        denom = float(np.sum(np.abs(weights))) or 1.0
        offset = num / denom
        return float(np.clip(self.global_mean + self.user_bias[u_idx] + offset, 0.5, 5.0))

    def predict_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        self._check_fit()
        out = np.empty(len(pairs), dtype=np.float32)
        users = pairs["user_id"].to_numpy()
        items = pairs["movie_id"].to_numpy()
        for i, (u, m) in enumerate(zip(users, items)):
            out[i] = self.predict(int(u), int(m))
        return out

    def top_n(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        self._check_fit()
        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is None:
            return []
        user_row = self._user_centered_row(u_idx)
        # scores[i] = sum_j sim[i, j] * user_row[j] / sum_j |sim[i, j]| over j where user_row[j] != 0
        mask = user_row != 0
        if not mask.any():
            return []
        sim_masked = self.sim[:, mask]  # (n_items, n_rated_by_u)
        num = sim_masked @ user_row[mask]
        denom = np.abs(sim_masked).sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        offset = num / denom
        scores = self.global_mean + self.user_bias[u_idx] + offset
        if exclude_seen and u_idx in self.seen:
            scores = scores.copy()
            scores[list(self.seen[u_idx])] = -np.inf
        top_idx = np.argpartition(-scores, range(min(n, len(scores))))[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(self.idx_to_item[i]), float(np.clip(scores[i], 0.5, 5.0))) for i in top_idx]
