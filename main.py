"""FastAPI app surfacing all four trained components of MovieLens-SR-Mixed.

Endpoints:
  GET  /                        -> single-page demo UI (static/index.html)
  POST /api/sentiment           -> classify a free-text comment as positive/negative
  POST /api/cluster             -> assign a user profile vector to a KMeans cluster
  GET  /api/recommend/{uid}     -> top-N CF recommendations for an existing user
  GET  /api/forecast/{genre}    -> next-quarter popularity forecast for a genre
  GET  /api/genres              -> list of genres available in the popularity model
  GET  /api/health              -> readiness flags for each artefact
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

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
                probs = clf.predict_proba(X)[0]
                payload["probabilities"] = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
            return payload
        # Fallback to the iteration-2 pipeline interface (Pipeline + label_classes in models/sentiment/).
        from src.sentiment.predict import predict_sentiment

        return predict_sentiment([req.text])[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/cluster")
def cluster(req: ClusterRequest) -> dict[str, Any]:
    scaler, kmeans, feature_cols = _kmeans_artifacts()
    row = pd.DataFrame([req.features]).reindex(columns=feature_cols, fill_value=0.0)
    cluster_id = int(kmeans.predict(scaler.transform(row.to_numpy()))[0])
    return {"cluster_id": cluster_id, "n_clusters": int(kmeans.n_clusters)}


@app.get("/api/recommend/{user_id}")
def recommend(user_id: int, n: int = 10, model: str = "svd") -> dict[str, Any]:
    try:
        from src.recsys.predict import top_n_for_user

        recs = top_n_for_user(user_id=user_id, n=n, model_name=model)
        return {"user_id": user_id, "model": model, "n": n, "recommendations": recs}
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
    """Return the list of feature names expected by /api/cluster so the UI can build the form."""
    _, _, feature_cols = _kmeans_artifacts()
    return {"feature_cols": list(feature_cols)}
