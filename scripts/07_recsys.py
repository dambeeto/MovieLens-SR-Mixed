"""End-to-end CF recommender training, evaluation and persistence (ml-1m)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    MODELS_RECSYS_DIR,
    PROCESSED_DIR,
    RECSYS_K_FACTORS,
    RECSYS_KNN_NEIGHBORS,
    RECSYS_LIKE_THRESHOLD,
    RECSYS_TOP_N,
    REPORTS_RECSYS_DIR,
    ensure_dirs,
)
from src.recsys.evaluate import hit_rate_at_k, rmse_mae
from src.recsys.knn import ItemKNNRecommender
from src.recsys.svd import SVDRecommender


def _require_parquet(name: str) -> Path:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run scripts/02_clean.py first to generate the parquet files."
        )
    return path


def _fit_models(train: pd.DataFrame) -> dict[str, object]:
    print(f"[recsys] fitting SVD (k={RECSYS_K_FACTORS})")
    svd = SVDRecommender(k=RECSYS_K_FACTORS).fit(train)
    print(f"[recsys] fitting ItemKNN cosine (k={RECSYS_KNN_NEIGHBORS})")
    knn_cos = ItemKNNRecommender(similarity="cosine", k=RECSYS_KNN_NEIGHBORS).fit(train)
    print(f"[recsys] fitting ItemKNN pearson (k={RECSYS_KNN_NEIGHBORS})")
    knn_pea = ItemKNNRecommender(similarity="pearson", k=RECSYS_KNN_NEIGHBORS).fit(train)
    return {"svd": svd, "knn_cosine": knn_cos, "knn_pearson": knn_pea}


def _evaluate_all(models: dict, test: pd.DataFrame, hit_rate_max_users: int = 1000) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        print(f"[eval] {name} -- rating prediction")
        rm = rmse_mae(model, test)
        print(f"       rmse={rm['rmse']:.4f}  mae={rm['mae']:.4f}  n_pairs={rm['n_pairs']}")
        print(f"[eval] {name} -- hit-rate@{RECSYS_TOP_N} (sample {hit_rate_max_users} users)")
        hr = hit_rate_at_k(model, test, k=RECSYS_TOP_N,
                           like_threshold=RECSYS_LIKE_THRESHOLD, max_users=hit_rate_max_users)
        print(f"       hit_rate@{RECSYS_TOP_N}={hr['hit_rate_at_k']:.4f}  n_users={hr['n_users_eval']}")
        rows.append(
            {
                "model": name,
                "rmse": rm["rmse"],
                "mae": rm["mae"],
                "hit_rate_at_10": hr["hit_rate_at_k"],
                "n_test_pairs": rm["n_pairs"],
                "n_eval_users": hr["n_users_eval"],
            }
        )
    return pd.DataFrame(rows)


def _persist_models(models: dict) -> None:
    for name, model in models.items():
        joblib.dump(model, MODELS_RECSYS_DIR / f"{name if name == 'svd' else 'item_' + name}.joblib")
    # Indices: pull from any fitted model (they share the same ID mapping by construction
    # only if all were trained on the same train slice, which is our case here).
    any_model = next(iter(models.values()))
    joblib.dump(any_model.user_to_idx, MODELS_RECSYS_DIR / "user_index.joblib")
    joblib.dump(any_model.item_to_idx, MODELS_RECSYS_DIR / "item_index.joblib")
    print(f"[recsys] persisted models + indices to {MODELS_RECSYS_DIR}")


def _write_top10_examples(models: dict, movies: pd.DataFrame, user_ids: list[int]) -> None:
    movies_idx = movies.set_index("movie_id")
    rows = []
    for uid in user_ids:
        for name, model in models.items():
            for rank, (mid, score) in enumerate(model.top_n(uid, n=RECSYS_TOP_N), start=1):
                title = movies_idx.loc[mid, "title_clean"] if mid in movies_idx.index else "?"
                year = movies_idx.loc[mid, "year"] if mid in movies_idx.index else None
                rows.append(
                    {
                        "user_id": uid,
                        "model": name,
                        "rank": rank,
                        "movie_id": mid,
                        "title": str(title),
                        "year": int(year) if pd.notna(year) else None,
                        "predicted_rating": round(float(score), 4),
                    }
                )
    pd.DataFrame(rows).to_csv(REPORTS_RECSYS_DIR / "top10_examples.csv", index=False)
    print(f"[recsys] wrote top10_examples.csv (5 users x 3 models x {RECSYS_TOP_N} items)")


def _write_similarity_comparison(metrics: pd.DataFrame) -> None:
    knn = metrics[metrics["model"].str.startswith("knn_")].copy()
    if knn.empty:
        return
    knn["similarity"] = knn["model"].str.replace("knn_", "", regex=False)
    knn[["similarity", "rmse", "mae", "hit_rate_at_10"]].to_csv(
        REPORTS_RECSYS_DIR / "similarity_comparison.csv", index=False
    )


def _plot_pred_vs_actual(best_model_name: str, model, test: pd.DataFrame, n_sample: int = 5000) -> None:
    # Take a random sample to keep the scatter readable.
    rng = np.random.default_rng(0)
    pairs = test.sample(n=min(n_sample, len(test)), random_state=int(rng.integers(0, 1_000_000)))
    pairs = pairs[
        pairs["user_id"].isin(model.user_to_idx) & pairs["movie_id"].isin(model.item_to_idx)
    ]
    if pairs.empty:
        return
    y = pairs["rating"].to_numpy(dtype=np.float32)
    y_hat = model.predict_batch(pairs[["user_id", "movie_id"]])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_hat + np.random.default_rng(0).normal(0, 0.02, size=len(y_hat)),
               y + np.random.default_rng(1).normal(0, 0.02, size=len(y)),
               s=4, alpha=0.3, color="steelblue")
    ax.plot([0.5, 5.0], [0.5, 5.0], "r--", linewidth=1)
    ax.set_xlim(0.5, 5.0)
    ax.set_ylim(0.5, 5.0)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Predicted vs actual ratings -- {best_model_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_RECSYS_DIR / "rating_predictions_scatter.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    train_path = _require_parquet("train.parquet")
    test_path = _require_parquet("test.parquet")
    movies_path = _require_parquet("movies_1m.parquet")

    print(f"[recsys] loading {train_path}")
    train = pd.read_parquet(train_path, columns=["user_id", "movie_id", "rating", "timestamp"])
    test = pd.read_parquet(test_path, columns=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_parquet(movies_path)
    print(f"[recsys] train={len(train):,}  test={len(test):,}  movies={len(movies):,}")

    models = _fit_models(train)
    _persist_models(models)

    metrics = _evaluate_all(models, test)
    metrics_path = REPORTS_RECSYS_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[recsys] wrote {metrics_path}")
    print(metrics.to_string(index=False))

    _write_similarity_comparison(metrics)

    rng = np.random.default_rng(0)
    sample_users = rng.choice(np.array(list(models["svd"].user_to_idx)), size=5, replace=False)
    _write_top10_examples(models, movies, [int(u) for u in sample_users])

    best_name = metrics.sort_values("rmse").iloc[0]["model"]
    print(f"[recsys] best RMSE -> {best_name}")
    _plot_pred_vs_actual(best_name, models[best_name], test)

    # Console example: top-10 for the first sample user using the best model.
    sample_uid = int(sample_users[0])
    print(f"\n== Top-{RECSYS_TOP_N} for user {sample_uid} using {best_name} ==")
    movies_idx = movies.set_index("movie_id")
    for rank, (mid, score) in enumerate(models[best_name].top_n(sample_uid, n=RECSYS_TOP_N), start=1):
        title = movies_idx.loc[mid, "title_clean"] if mid in movies_idx.index else "?"
        year = movies_idx.loc[mid, "year"] if mid in movies_idx.index else "?"
        print(f"  {rank:2d}. {title} ({year})  -- predicted={score:.3f}")


if __name__ == "__main__":
    main()
