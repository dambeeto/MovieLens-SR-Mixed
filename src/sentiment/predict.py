"""Classify new comments using the trained sentiment pipeline."""

from __future__ import annotations

from functools import lru_cache

import joblib

from src.config import MODELS_SENTIMENT_DIR

PIPELINE_PATH = MODELS_SENTIMENT_DIR / "pipeline.joblib"


@lru_cache(maxsize=1)
def _load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Sentiment pipeline not found at {PIPELINE_PATH}. "
            "Run scripts/06_sentiment.py first."
        )
    return joblib.load(PIPELINE_PATH)


def predict_sentiment(texts: list[str]) -> list[dict]:
    """Classify each text as positive/negative/neutral with per-class probabilities."""
    pipe = _load_pipeline()
    if not texts:
        return []
    # Pipeline expects iterable of strings.
    cleaned = [str(t).strip().lower() for t in texts]
    preds = pipe.predict(cleaned)
    out: list[dict] = []
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(cleaned)
        classes = list(pipe.classes_)
        for txt, label, row in zip(texts, preds, probs):
            out.append(
                {
                    "text": txt,
                    "label": str(label),
                    "probabilities": {c: float(p) for c, p in zip(classes, row)},
                }
            )
    else:
        for txt, label in zip(texts, preds):
            out.append({"text": txt, "label": str(label), "probabilities": None})
    return out
