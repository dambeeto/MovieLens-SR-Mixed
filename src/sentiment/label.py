"""Filtering and sentiment labelling for MovieLens tags using NLTK and VADER."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk

# Tokenizer splits text into individual words
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
# POS tagger model - recognizes parts of speech
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

SENTIMENT_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}


def _has_sentiment_word(text: str) -> bool:
    """Returns True if the tag contains at least one adjective or adverb."""

    # Split tag into individual tokens
    tokens = nltk.word_tokenize(text.lower())
    # Assign POS tag to each token
    tags = nltk.pos_tag(tokens)

    return any(pos in SENTIMENT_POS for _, pos in tags)


def filter_sentiment_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Keeps only tags containing adjectives or adverbs."""

    mask = tags["tag"].apply(_has_sentiment_word)

    return tags[mask].reset_index(drop=True)


def label_sentiment(tags: pd.DataFrame) -> pd.DataFrame:
    """Labels each tag with a sentiment using VADER compound score."""
    analyzer = SentimentIntensityAnalyzer()

    def _score(text: str) -> str:
        # VADER returns scores like:
        # {"neg": 0.6, "neu": 0.3, "pos": 0.0, "compound": -0.54}
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    out = tags.copy()
    # Add new column with sentiment label for each tag
    out["sentiment"] = out["tag"].apply(_score)
    return out[out["sentiment"] != "neutral"].reset_index(drop=True)
