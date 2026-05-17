"""Cluster interpretation: PCA scatter + per-cluster profile table + console summary."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import GENRES_1M, REPORTS_DIR


def plot_pca(X_scaled: np.ndarray, labels: np.ndarray, k: int) -> None:
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=k)
    for c in range(k):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5, color=palette[c], label=f"c{c} (n={int(m.sum())})")
    ax.set_title(f"User clusters (PCA 2D, k={k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "clusters_pca.png", dpi=110)
    plt.close(fig)


def cluster_profiles(features: pd.DataFrame, labels: np.ndarray, users: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster summary: size, top 5 genres by mean rating, demographic snapshot."""
    df = features.copy()
    df["cluster"] = labels
    df = df.reset_index().rename(columns={"index": "user_id"}) if "user_id" not in df.columns else df

    rmean_cols = [f"rmean_{g}" for g in GENRES_1M]
    rows = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        # Drop zeros when picking top genres -- 0 means "user has no ratings in that genre".
        genre_means = sub[rmean_cols].replace(0, np.nan).mean().sort_values(ascending=False)
        top5 = ", ".join(g.replace("rmean_", "") for g in genre_means.head(5).index)
        u_sub = users.merge(sub[["user_id"]], on="user_id", how="inner")
        female_share = float(u_sub["is_female"].mean())
        age_mean = float(u_sub["age_midpoint"].mean())
        top_age = u_sub["age_label"].mode().iloc[0] if not u_sub.empty else ""
        top_occ = u_sub["occupation_label"].mode().iloc[0] if not u_sub.empty else ""
        rows.append(
            {
                "cluster": c,
                "size": int(len(sub)),
                "top5_genres_by_mean_rating": top5,
                "female_share": round(female_share, 3),
                "age_mean": round(age_mean, 1),
                "modal_age_bucket": top_age,
                "modal_occupation": top_occ,
            }
        )
    return pd.DataFrame(rows)


def print_summary(profiles: pd.DataFrame) -> None:
    for _, r in profiles.iterrows():
        gender_word = "kobiety" if r["female_share"] > 0.55 else ("mezczyzni" if r["female_share"] < 0.45 else "mieszane")
        print(
            f"  cluster {int(r['cluster'])} (n={r['size']}): {gender_word}, "
            f"sredni wiek {r['age_mean']} ({r['modal_age_bucket']}), "
            f"glownie {r['modal_occupation']}, top gatunki: {r['top5_genres_by_mean_rating']}"
        )
