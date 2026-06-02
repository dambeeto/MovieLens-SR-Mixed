import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import PROCESSED_DIR, ensure_dirs
from src.sentiment.label import filter_sentiment_tags, label_sentiment
from src.sentiment.train import train_sentiment_classifier


def main() -> None:
    ensure_dirs()

    tags = pd.read_parquet(PROCESSED_DIR / "tags_20m.parquet")
    print(f"[sentiment] loaded {len(tags)} tags")

    # 1. Filtering - leaves tags that are adjectives or adverbs
    filtered = filter_sentiment_tags(tags)
    print(f"[sentiment] after filter: {len(filtered)} tags")

    # 2. VADER labeling
    labelled = label_sentiment(filtered)
    print(
        f"[sentiment] after labelling: {len(labelled)} tags (neutral deleted)")

    # 3. Classifier training
    train_sentiment_classifier(labelled)


if __name__ == "__main__":
    main()
