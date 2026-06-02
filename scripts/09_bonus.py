"""Bonus tasks: outlier detection + feature selection diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.bonus.feature_selection import mutual_info_report, variance_report
from src.bonus.outliers import detect_outliers
from src.config import (
    MODELS_BONUS_DIR,
    PROCESSED_DIR,
    REPORTS_BONUS_DIR,
    ensure_dirs,
)


def _outliers_section() -> pd.DataFrame:
    path = PROCESSED_DIR / "user_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/04_features.py first")
    feats = pd.read_parquet(path)
    print(f"[bonus] loaded {len(feats)} users from {path}")

    flagged, model, scaler = detect_outliers(feats)
    n_out = int(flagged["is_outlier"].sum())
    print(f"[bonus] outliers detected: {n_out}/{len(flagged)} ({100*n_out/len(flagged):.2f}%)")

    flagged.to_parquet(PROCESSED_DIR / "user_outliers.parquet", index=False)
    joblib.dump(model, MODELS_BONUS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODELS_BONUS_DIR / "isolation_scaler.joblib")

    # Scatter on first two PCA components, outliers in red.
    X = feats.drop(columns=["user_id"]).select_dtypes(include="number").to_numpy()
    coords = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(8, 6))
    inl = flagged["is_outlier"] == 0
    ax.scatter(coords[inl, 0], coords[inl, 1], s=6, alpha=0.3, color="steelblue", label="inlier")
    ax.scatter(coords[~inl, 0], coords[~inl, 1], s=14, alpha=0.8, color="crimson", label=f"outlier (n={n_out})")
    ax.set_title("Users in PCA space -- IsolationForest outliers highlighted")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_BONUS_DIR / "outliers_pca.png", dpi=110)
    plt.close(fig)

    # Top 10 most anomalous users for the CSV.
    top_anom = flagged.sort_values("anomaly_score").head(10)
    top_anom.to_csv(REPORTS_BONUS_DIR / "top_anomalous_users.csv", index=False)
    return flagged


def _feature_selection_section() -> None:
    feats_path = PROCESSED_DIR / "user_features.parquet"
    clust_path = PROCESSED_DIR / "user_clusters.parquet"
    if not (feats_path.exists() and clust_path.exists()):
        print("[bonus] skipping feature selection (need user_features + user_clusters)")
        return
    feats = pd.read_parquet(feats_path)
    clusters = pd.read_parquet(clust_path)
    joined = feats.merge(clusters, on="user_id", how="inner")
    labels = joined["cluster_id"].to_numpy()
    X = joined.drop(columns=["user_id", "cluster_id"])

    var_df, _ = variance_report(X, threshold=0.01)
    var_df.to_csv(REPORTS_BONUS_DIR / "variance_report.csv", index=False)
    n_low = int(var_df["low_variance"].sum())
    print(f"[bonus] variance: {n_low}/{len(var_df)} cech ponizej progu 0.01")

    mi_df = mutual_info_report(X, labels, top_k=20)
    mi_df.to_csv(REPORTS_BONUS_DIR / "mutual_info_top20.csv", index=False)
    print("[bonus] top-10 mutual_info vs cluster_id:")
    print(mi_df.head(10).to_string(index=False))


def main() -> None:
    ensure_dirs()
    print("== Bonus: outliers ==")
    _outliers_section()
    print("\n== Bonus: feature selection ==")
    _feature_selection_section()
    print("\n[done] bonus reports in", REPORTS_BONUS_DIR)


if __name__ == "__main__":
    main()
