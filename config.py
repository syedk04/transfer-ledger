"""Single source of truth for paths, seasons, and IDs used across the pipeline.

Rationale: scattering "GB1" or a season list across multiple files means that
changing scope later (e.g. to the Championship, or to 6 seasons) requires
hunting through every module. Centralizing it here means each module imports
what it needs and the actual logic files stay free of magic numbers.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

PLAYER_SEASONS_CSV = PROCESSED_DATA_DIR / "player_seasons.csv"

MODEL_PATHS = {
    "linear_regression": MODELS_DIR / "linear_regression.joblib",
    "ridge": MODELS_DIR / "ridge.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}
METRICS_JSON = REPORTS_DIR / "metrics.json"
PREDICTIONS_CSV = REPORTS_DIR / "test_predictions.csv"

# The model predict.py loads by default. Ridge, not plain LinearRegression:
# it gets nearly identical accuracy but with stable, trustworthy coefficients
# (see train.py docstring on why OLS coefficients are unreliable here).
DEFAULT_MODEL = "ridge"

# The model the upcoming SHAP-explainability / agent pipeline will use.
# Deliberately separate from DEFAULT_MODEL: SHAP's fast TreeExplainer needs
# a tree-based model, and Ridge's own coefficients already serve as its
# "explanation" - so the interpretable linear baseline (predict.py's
# default) and the model the agent explains are allowed to be different
# models on purpose, not a change of mind about which is "best."
SCOUTING_MODEL = "xgboost"

# --- XGBoost hyperparameters ------------------------------------------------
# This is not a hyperparameter-search project - these are sane, modestly
# tuned defaults, not a tuned optimum. n_estimators/max_depth/learning_rate
# are kept conservative (shallow trees, slow learning rate) because the
# training set is small (~4,258 rows): a deep, fast-learning boosted model
# would overfit noise in individual players' stats rather than the general
# stats-to-value relationship, the same overfitting risk flagged for
# RandomForest above.
XGBOOST_N_ESTIMATORS = 300
XGBOOST_MAX_DEPTH = 3
XGBOOST_LEARNING_RATE = 0.05

# --- Remote data source --------------------------------------------------
# transfermarkt-datasets (CC0) publishes gzipped CSV exports behind this
# Cloudflare R2 bucket. There is no versioned API here, just a flat file per
# table, so we cache whatever we download and only re-fetch on request.

DATA_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

RAW_FILES = {
    "players": "players.csv.gz",
    "appearances": "appearances.csv.gz",
    "player_valuations": "player_valuations.csv.gz",
    "games": "games.csv.gz",
    "competitions": "competitions.csv.gz",
}

# --- Scope ---------------------------------------------------------------

COMPETITION_ID = "GB1"  # Premier League, per competitions.csv

# Season labels are the year the season started, e.g. 2024 == the 2024-25
# season. Verified against games.csv on 2026-09-03: GB1 season 2025 (i.e.
# 2025-26) already has all 380 matches played, so it is the most recently
# *completed* season as of today, not 2024.
LAST_COMPLETED_SEASON = 2025
N_SEASONS = 10  # "the last decade" -> 2016-17 through 2025-26
SEASONS = list(range(LAST_COMPLETED_SEASON - N_SEASONS + 1, LAST_COMPLETED_SEASON + 1))

# Train/test split is by season, not random — see build_dataset.py / train.py
# docstrings for why a random split leaks information here.
TEST_SEASON = LAST_COMPLETED_SEASON
TRAIN_SEASONS = [s for s in SEASONS if s != TEST_SEASON]

# A Premier League season runs roughly Aug (year N) to May (year N+1). We
# treat a season as "closed" for valuation purposes on June 30 of year N+1 -
# late enough that the season's final matches have been played and reflected
# in valuations, early enough that it's before the next season's transfer
# business and pre-season hype could move the price.
SEASON_END_MONTH = 6
SEASON_END_DAY = 30

RANDOM_STATE = 42

# --- Feature engineering ---------------------------------------------------

# A player with a single 3-minute cameo can have a goals_per_90 of 30 - the
# ratio is technically correct but statistically meaningless with so little
# playing time behind it, and it would just inject noise into training. We
# require at least one full match-equivalent of minutes before a
# player-season is used at all.
MIN_MINUTES_PLAYED = 90

TARGET_COL = "market_value_eur"
LOG_TARGET_COL = "log_market_value_eur"

# Plain per-season counting stats.
COUNT_FEATURES = [
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
]

# Engineered on top of the counting stats - see features.py for the "why"
# behind each one (per-90 rates control for playing time, age_squared lets a
# linear model represent a rise-then-decline career curve).
ENGINEERED_NUMERIC_FEATURES = [
    "goals_per_90",
    "assists_per_90",
    "age_at_season_end",
    "age_squared",
    "height_in_cm",
]

NUMERIC_FEATURES = COUNT_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["position"]

ID_COLUMNS = ["player_id", "name", "season"]
