"""Download MovieLens 1M + 20M into data/raw/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import download_all

if __name__ == "__main__":
    download_all()
