"""Weak sentiment labelling via VADER on short tag texts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import SENTIMENT_CLASSES, VADER_NEG_THRESHOLD, VADER_POS_THRESHOLD


def compound_scores(texts: pd.Series) -> np.ndarray:
    """Return VADER compound score in [-1, 1] for each input text."""
    analyzer = SentimentIntensityAnalyzer()
    out = np.empty(len(texts), dtype="float32")
    # VADER has no native batching; the per-text call is ~1us-1ms, totally fine
    # for the ~465k tags we have.
    for i, t in enumerate(texts.astype("string").fillna("")):
        out[i] = analyzer.polarity_scores(str(t))["compound"]
    return out


def compound_to_label(compound: np.ndarray) -> np.ndarray:
    """Map VADER compound score to {'negative','neutral','positive'} using thresholds."""
    labels = np.full(compound.shape, "neutral", dtype=object)
    labels[compound >= VADER_POS_THRESHOLD] = "positive"
    labels[compound <= VADER_NEG_THRESHOLD] = "negative"
    return labels


def label_with_vader(df: pd.DataFrame, text_col: str = "tag") -> pd.DataFrame:
    """Add 'vader_compound' (float32) and 'sentiment' (categorical) columns to a copy of df."""
    if text_col not in df.columns:
        raise KeyError(f"column {text_col!r} missing in DataFrame")
    out = df.copy()
    compound = compound_scores(out[text_col])
    labels = compound_to_label(compound)
    out["vader_compound"] = compound
    out["sentiment"] = pd.Categorical(labels, categories=list(SENTIMENT_CLASSES), ordered=False)
    return out
