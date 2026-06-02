"""Paths, dataset URLs and code mappings for the MovieLens project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_SENTIMENT_DIR = MODELS_DIR / "sentiment"
REPORTS_SENTIMENT_DIR = REPORTS_DIR / "sentiment"

MODELS_RECSYS_DIR = MODELS_DIR / "recsys"
REPORTS_RECSYS_DIR = REPORTS_DIR / "recsys"

MODELS_POPULARITY_DIR = MODELS_DIR / "popularity"
REPORTS_POPULARITY_DIR = REPORTS_DIR / "popularity"

REPORTS_BONUS_DIR = REPORTS_DIR / "bonus"
MODELS_BONUS_DIR = MODELS_DIR / "bonus"

STATIC_DIR = PROJECT_ROOT / "static"

ML_1M_DIR = RAW_DIR / "ml-1m"
ML_20M_DIR = RAW_DIR / "ml-20m"

DATASET_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}

# MovieLens 1M genres (canonical 18 from the README).
GENRES_1M = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Age codes -> human-readable bucket + numeric midpoint used as a feature.
AGE_CODE_TO_LABEL = {
    1:  "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}
AGE_CODE_TO_MIDPOINT = {1: 15, 18: 21, 25: 30, 35: 40, 45: 47, 50: 53, 56: 60}

# Occupation codes from the MovieLens 1M README.
OCCUPATION_CODE_TO_LABEL = {
    0:  "other",
    1:  "academic/educator",
    2:  "artist",
    3:  "clerical/admin",
    4:  "college/grad student",
    5:  "customer service",
    6:  "doctor/health care",
    7:  "executive/managerial",
    8:  "farmer",
    9:  "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# Sentiment classification (iteration 2): VADER compound thresholds for weak labels.
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05
SENTIMENT_CLASSES = ("negative", "neutral", "positive")

# Recommender system (iteration 3): hyperparameters shared by SVD and item-KNN models.
RECSYS_K_FACTORS = 50          # SVD latent factors
RECSYS_KNN_NEIGHBORS = 30      # item-based KNN neighbours
RECSYS_TOP_N = 10              # default top-N recommendation length
RECSYS_LIKE_THRESHOLD = 4.0    # rating threshold for hit-rate@K "relevant" definition

# Popularity regression (iteration 4): time series aggregation + walk-forward forecasting.
POPULARITY_TEST_QUARTERS = 4   # how many of the latest quarters are held out for evaluation
POPULARITY_LAGS = (1, 2, 4)    # quarter offsets used as autoregressive features

# Bonus tasks: outlier detection + feature selection.
OUTLIER_CONTAMINATION = 0.05   # IsolationForest expected fraction of outliers


def ensure_dirs() -> None:
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        MODELS_SENTIMENT_DIR,
        REPORTS_SENTIMENT_DIR,
        MODELS_RECSYS_DIR,
        REPORTS_RECSYS_DIR,
        MODELS_POPULARITY_DIR,
        REPORTS_POPULARITY_DIR,
        REPORTS_BONUS_DIR,
        MODELS_BONUS_DIR,
        STATIC_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
