"""Train KMeans on user features, persist artifacts, write interpretation reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.clustering.interpret import cluster_profiles, plot_pca, print_summary
from src.clustering.kmeans import fit_kmeans, persist
from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    users = pd.read_parquet(PROCESSED_DIR / "users_1m.parquet")
    feats = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")

    user_ids = feats["user_id"].astype("int32")
    X_df = feats.drop(columns=["user_id"])

    result = fit_kmeans(X_df)
    persist(result)

    result.selection_report.to_csv(REPORTS_DIR / "k_selection.csv", index=False)
    print(f"[cluster] wrote k_selection.csv")

    X_scaled = result.scaler.transform(X_df.values)
    plot_pca(X_scaled, result.labels, result.best_k)
    print(f"[cluster] wrote clusters_pca.png")

    assignments = pd.DataFrame({"user_id": user_ids.values, "cluster_id": result.labels})
    assignments.to_parquet(PROCESSED_DIR / "user_clusters.parquet", index=False)

    feats_for_profiles = X_df.copy()
    feats_for_profiles.insert(0, "user_id", user_ids.values)
    profiles = cluster_profiles(feats_for_profiles.set_index("user_id"), result.labels, users)
    profiles.to_csv(REPORTS_DIR / "cluster_profiles.csv", index=False)
    print(f"[cluster] wrote cluster_profiles.csv (k*={result.best_k})")

    print("\n== Cluster summaries ==")
    print_summary(profiles)


if __name__ == "__main__":
    main()
