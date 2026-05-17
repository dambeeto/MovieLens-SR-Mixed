"""KMeans on user features: scale -> sweep k -> pick by silhouette -> persist."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE


@dataclass
class ClusteringResult:
    scaler: StandardScaler
    model: KMeans
    feature_cols: list[str]
    selection_report: pd.DataFrame
    best_k: int
    labels: np.ndarray


def _sweep_k(
    X: np.ndarray, k_values: range, silhouette_sample: int = 2000
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    n_samples = X.shape[0]
    sample_idx = rng.choice(n_samples, size=min(silhouette_sample, n_samples), replace=False)
    Xs = X[sample_idx]

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        labels_sample = labels[sample_idx]
        sil = float(silhouette_score(Xs, labels_sample)) if len(set(labels_sample)) > 1 else float("nan")
        db = float(davies_bouldin_score(X, labels))
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "inertia": float(km.inertia_)})
        print(f"  k={k:2d}  silhouette={sil:6.4f}  DB={db:6.4f}  inertia={km.inertia_:.1f}")
    return pd.DataFrame(rows)


def _pick_best_k(report: pd.DataFrame) -> int:
    # Maximize silhouette; break ties by minimum Davies-Bouldin (lower is better).
    sorted_df = report.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    return int(sorted_df.iloc[0]["k"])


def fit_kmeans(
    features: pd.DataFrame, k_values: range = range(2, 16)
) -> ClusteringResult:
    """Fit scaler, sweep k, refit the best model, return everything (no plotting here)."""
    feature_cols = list(features.columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    print(f"[kmeans] sweeping k in {list(k_values)} on shape={X.shape}")
    report = _sweep_k(X, k_values)
    best_k = _pick_best_k(report)
    print(f"[kmeans] selected k*={best_k}")

    final = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = final.fit_predict(X)
    return ClusteringResult(
        scaler=scaler,
        model=final,
        feature_cols=feature_cols,
        selection_report=report,
        best_k=best_k,
        labels=labels,
    )


def persist(result: ClusteringResult) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(result.model, MODELS_DIR / "kmeans.joblib")
    joblib.dump(result.feature_cols, MODELS_DIR / "feature_cols.joblib")
    print(f"[kmeans] saved scaler.joblib, kmeans.joblib, feature_cols.joblib in {MODELS_DIR}")
