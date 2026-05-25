"""TF-IDF feature extraction for short-text sentiment classification."""

from __future__ import annotations

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer() -> FeatureUnion:
    """Word (1-2) + char_wb (3-5) TF-IDF union.

    Word features cover normal vocabulary, char_wb robust to typos / morphology
    which are common in user-generated tags (e.g. 'funy', 'predicatable').
    """
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,  # tags already lowercased in clean_tags_20m
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )
    return FeatureUnion([("word", word), ("char", char)])
