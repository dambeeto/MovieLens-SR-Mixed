"""SVD-based collaborative filtering recommender (scipy.sparse.linalg.svds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.config import RECSYS_K_FACTORS
from src.recsys.matrix import build_user_item_matrix


class SVDRecommender:
    """Truncated SVD on user-mean-centered ratings.

    Prediction: r_hat(u, i) = global_mean + user_bias[u] + item_bias[i] + p_u . q_i
    where p_u = U @ diag(sqrt(sigma)) and q_i = (diag(sqrt(sigma)) @ Vt)^T.
    """

    def __init__(self, k: int = RECSYS_K_FACTORS) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        # filled in fit()
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: np.ndarray | None = None
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None   # (n_users, k)
        self.item_factors: np.ndarray | None = None   # (n_items, k)
        self.seen: dict[int, set[int]] = {}           # train interactions per user (internal idx -> item internal idx set)

    def fit(self, train: pd.DataFrame) -> "SVDRecommender":
        mat, user_to_idx, item_to_idx = build_user_item_matrix(train)
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_item = np.array(sorted(item_to_idx, key=item_to_idx.get))

        # Compute biases on the raw rating matrix (sparse means).
        n_users, n_items = mat.shape
        nnz_mask = mat.copy()
        nnz_mask.data = np.ones_like(nnz_mask.data)

        ratings_sum = np.asarray(mat.sum(axis=1)).ravel()
        ratings_cnt = np.asarray(nnz_mask.sum(axis=1)).ravel()
        items_sum = np.asarray(mat.sum(axis=0)).ravel()
        items_cnt = np.asarray(nnz_mask.sum(axis=0)).ravel()

        self.global_mean = float(mat.data.mean())
        self.user_bias = (ratings_sum / np.maximum(ratings_cnt, 1)) - self.global_mean
        self.item_bias = (items_sum / np.maximum(items_cnt, 1)) - self.global_mean

        # Center the matrix: subtract user mean from observed entries only.
        # Convert to COO for elementwise broadcasting, then back to CSR.
        coo = mat.tocoo()
        centered_vals = coo.data - (self.global_mean + self.user_bias[coo.row])
        centered = csr_matrix((centered_vals, (coo.row, coo.col)), shape=mat.shape)

        # svds requires k < min(shape).
        k_eff = min(self.k, min(centered.shape) - 1)
        if k_eff < self.k:
            print(f"[svd] requested k={self.k} too large; using k={k_eff}")
        U, sigma, Vt = svds(centered.astype(np.float32), k=k_eff)
        # svds returns ascending singular values; reorder to descending for clarity.
        order = np.argsort(-sigma)
        U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

        sqrt_sigma = np.sqrt(sigma)
        self.user_factors = (U * sqrt_sigma).astype(np.float32)
        self.item_factors = (Vt.T * sqrt_sigma).astype(np.float32)

        # Track seen interactions for exclude_seen at top-N.
        self.seen = {}
        train_rows = train[["user_id", "movie_id"]].itertuples(index=False)
        for u_id, m_id in train_rows:
            u_idx = self.user_to_idx.get(int(u_id))
            m_idx = self.item_to_idx.get(int(m_id))
            if u_idx is None or m_idx is None:
                continue
            self.seen.setdefault(u_idx, set()).add(m_idx)
        return self

    # --- Prediction helpers ---

    def _check_fit(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("SVDRecommender is not fitted")

    def predict(self, user_id: int, item_id: int) -> float:
        self._check_fit()
        u_idx = self.user_to_idx.get(int(user_id))
        m_idx = self.item_to_idx.get(int(item_id))
        if u_idx is None or m_idx is None:
            # Unknown user or item -> fall back to global mean.
            return float(self.global_mean)
        bias = self.global_mean + float(self.user_bias[u_idx]) + float(self.item_bias[m_idx])
        dot = float(self.user_factors[u_idx] @ self.item_factors[m_idx])
        return float(np.clip(bias + dot, 0.5, 5.0))

    def predict_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        self._check_fit()
        u = pairs["user_id"].map(self.user_to_idx).to_numpy()
        m = pairs["movie_id"].map(self.item_to_idx).to_numpy()
        out = np.full(len(pairs), self.global_mean, dtype=np.float32)
        known = (~pd.isna(u)) & (~pd.isna(m))
        if known.any():
            ui = u[known].astype(int)
            mi = m[known].astype(int)
            bias = self.global_mean + self.user_bias[ui] + self.item_bias[mi]
            dot = np.einsum("ij,ij->i", self.user_factors[ui], self.item_factors[mi])
            out[known] = np.clip(bias + dot, 0.5, 5.0)
        return out

    def top_n(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        self._check_fit()
        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is None:
            return []
        bias = self.global_mean + self.user_bias[u_idx] + self.item_bias
        scores = bias + self.item_factors @ self.user_factors[u_idx]
        if exclude_seen and u_idx in self.seen:
            scores = scores.copy()
            scores[list(self.seen[u_idx])] = -np.inf
        top_idx = np.argpartition(-scores, range(min(n, len(scores))))[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(self.idx_to_item[i]), float(np.clip(scores[i], 0.5, 5.0))) for i in top_idx]
