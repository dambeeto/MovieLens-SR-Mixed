"""FastAPI app surfacing all four trained components of MovieLens-SR-Mixed.

Endpoints:
  GET  /                        -> single-page demo UI (static/index.html)
  POST /api/sentiment           -> classify a free-text comment as positive/negative
  POST /api/cluster             -> assign a user profile vector to a KMeans cluster
  GET  /api/cluster/profile/{id}   -> readable description of a cluster
  GET  /api/cluster/recommend/{id} -> cold-start picks for a cluster's profile
  GET  /api/recommend/{uid}     -> top-N CF recommendations (+ user profile, cold-start fallback)
  GET  /api/forecast/{genre}    -> next-quarter popularity forecast for a genre
  GET  /api/genres              -> list of genres available in the popularity model
  GET  /api/health              -> readiness flags for each artefact
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

# Pre-import sklearn estimators that may be present in persisted joblib files.
# Without this, the first lazy joblib.load() inside an endpoint can race with
# sklearn's own (re)initialization under uvicorn --reload and surface as:
#   ImportError: cannot import name 'clone' from partially initialized module 'sklearn.base'
# Importing them once at module load time forces full initialisation up front.
import sklearn.base  # noqa: F401
from sklearn.cluster import KMeans  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: F401
from sklearn.ensemble import IsolationForest, RandomForestRegressor  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: F401

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, PROCESSED_DIR, STATIC_DIR

app = FastAPI(title="MovieLens-SR-Mixed", version="1.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- Pydantic schemas ----------


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class ClusterRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Mapping feature_name -> value; missing features default to 0."
    )


# ---------- Lazy loaders ----------


@lru_cache(maxsize=1)
def _kmeans_artifacts() -> tuple:
    scaler_path = MODELS_DIR / "scaler.joblib"
    kmeans_path = MODELS_DIR / "kmeans.joblib"
    cols_path = MODELS_DIR / "feature_cols.joblib"
    for p in (scaler_path, kmeans_path, cols_path):
        if not p.exists():
            raise HTTPException(status_code=503, detail=f"missing artefact {p.name}; run scripts/05_cluster.py")
    return joblib.load(scaler_path), joblib.load(kmeans_path), joblib.load(cols_path)


# ---------- Helpers ----------


def _is_ready() -> dict[str, bool]:
    """Quick existence check for every artefact the UI relies on."""
    checks = {
        "sentiment": (MODELS_DIR / "sentiment_clf.joblib").exists()
        and (MODELS_DIR / "sentiment_vectorizer.joblib").exists(),
        "cluster": all(
            (MODELS_DIR / fname).exists() for fname in ("scaler.joblib", "kmeans.joblib", "feature_cols.joblib")
        ),
        "recsys": (MODELS_DIR / "recsys" / "svd.joblib").exists(),
        "popularity": (MODELS_DIR / "popularity" / "best_model.joblib").exists(),
    }
    return checks


# ---------- Routes ----------


@app.get("/")
def index() -> Any:
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(str(page))
    return {"message": "MovieLens-SR-Mixed API. UI not built yet. See /docs."}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ready": _is_ready()}


@app.post("/api/sentiment")
def sentiment(req: SentimentRequest) -> dict[str, Any]:
    try:
        # The colleague's binary classifier ships vectorizer + clf as two flat artefacts in models/.
        vec_path = MODELS_DIR / "sentiment_vectorizer.joblib"
        clf_path = MODELS_DIR / "sentiment_clf.joblib"
        if vec_path.exists() and clf_path.exists():
            vectorizer = joblib.load(vec_path)
            clf = joblib.load(clf_path)
            X = vectorizer.transform([req.text.strip().lower()])
            label = str(clf.predict(X)[0])
            payload: dict[str, Any] = {"text": req.text, "label": label}
            if hasattr(clf, "predict_proba"):
                try:
                    probs = clf.predict_proba(X)[0]
                    payload["probabilities"] = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
                except Exception:
                    # A model pickled with a different sklearn can fail predict_proba (version
                    # skew) while predict() still works; return the label without probabilities.
                    pass
            return payload
        # Fallback to the iteration-2 pipeline interface (Pipeline + label_classes in models/sentiment/).
        from src.sentiment.predict import predict_sentiment

        return predict_sentiment([req.text])[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/cluster")
def cluster(req: ClusterRequest) -> dict[str, Any]:
    import numpy as np

    scaler, kmeans, feature_cols = _kmeans_artifacts()
    row = pd.DataFrame([req.features]).reindex(columns=feature_cols, fill_value=0.0)
    # sklearn>=1.4 KMeans.predict needs float64; the scaler returns float32 when input is float32.
    Xs = scaler.transform(row.to_numpy(dtype=np.float64))
    cluster_id = int(kmeans.predict(Xs.astype(np.float64, copy=False))[0])
    return {"cluster_id": cluster_id, "n_clusters": int(kmeans.n_clusters)}


@app.get("/api/cluster/profile/{cluster_id}")
def cluster_profile(cluster_id: int) -> dict[str, Any]:
    """Readable description of a KMeans cluster (from reports/cluster_profiles.csv)."""
    try:
        from src.profiles import get_cluster_profile

        prof = get_cluster_profile(cluster_id)
        if prof is None:
            raise HTTPException(status_code=404, detail=f"cluster {cluster_id} not found")
        return prof
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/cluster/recommend/{cluster_id}")
def cluster_recommend(cluster_id: int, n: int = 10) -> dict[str, Any]:
    """Cold-start picks for a brand-new profile: top movies among that cluster's members."""
    try:
        from src.recsys.cold_start import recommend_for_cluster

        return {"cluster_id": cluster_id, "recommendations": recommend_for_cluster(cluster_id, n)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/recommend/{user_id}")
def recommend(user_id: int, n: int = 10, model: str = "svd") -> dict[str, Any]:
    try:
        from src.profiles import get_user_profile
        from src.recsys.cold_start import popular_top_n
        from src.recsys.predict import top_n_for_user

        recs = top_n_for_user(user_id=user_id, n=n, model_name=model)
        base = {"user_id": user_id, "model": model, "n": n}
        if not recs:
            # Unknown user (cold start): serve popular picks instead of an empty table.
            return {**base, "found": False, "fallback": "popular", "profile": None,
                    "recommendations": popular_top_n(n)}
        return {**base, "found": True, "profile": get_user_profile(user_id), "recommendations": recs}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/forecast/{genre}")
def forecast(genre: str) -> dict[str, Any]:
    try:
        from src.popularity.predict import forecast_next_quarter

        return forecast_next_quarter(genre)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/genres")
def list_genres() -> dict[str, Any]:
    try:
        from src.popularity.predict import _load

        state = _load()
        return {"genres": sorted(state["history"]["genre"].unique().tolist())}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/users/sample")
def sample_users(n: int = 8) -> dict[str, Any]:
    """Pick a few example user_ids that exist in the recommender index (helps the UI demo)."""
    path = MODELS_DIR / "recsys" / "user_index.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail="user_index missing; run scripts/07_recsys.py")
    idx = joblib.load(path)
    sample = sorted(list(idx.keys()))[: max(1, int(n))]
    return {"sample_user_ids": [int(u) for u in sample]}


@app.get("/api/cluster/template")
def cluster_template() -> dict[str, Any]:
    """Feature names + per-feature training means (StandardScaler.mean_) for the UI form.

    The means let the UI seed unselected genre features with a neutral baseline instead of 0,
    so a sparse profile is not read as "rated almost nothing" (which collapsed every profile
    into the low-activity cluster).
    """
    scaler, _, feature_cols = _kmeans_artifacts()
    baseline = {c: float(m) for c, m in zip(feature_cols, scaler.mean_)}
    return {"feature_cols": list(feature_cols), "baseline": baseline}
