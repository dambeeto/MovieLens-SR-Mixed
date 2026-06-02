from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd

from src.config import MODELS_DIR


def train_sentiment_classifier(tags: pd.DataFrame) -> None:
    """Trains TF-IDF + Logistic Regression on labelled sentiment data."""

    X = tags["tag"]
    y = tags["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build TF-IDF vocab from 5000 most frequent words
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Testing performance
    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    joblib.dump(vectorizer, MODELS_DIR / "sentiment_vectorizer.joblib")
    joblib.dump(clf, MODELS_DIR / "sentiment_clf.joblib")
    print("[sentiment] saved vectorizer and classifier")
