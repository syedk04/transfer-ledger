"""Train, evaluate, and persist the four regression models.

Why log1p(market_value) instead of raw euros
----------------------------------------------
Premier League market values are heavily right-skewed: most rotation
players sit under EUR 15M while a handful of superstars sit at
EUR 150-200M. A linear model minimizes squared error, so a few outliers
at 150M+ dominate the loss and the model effectively becomes a predictor
of "is this a top-10 player or not," fitting everyone else poorly. Taking
log1p(value) compresses that range (log1p(15_000_000) ~= 16.5,
log1p(200_000_000) ~= 19.1) so a proportional error - being off by 20% -
costs roughly the same everywhere on the scale, which matches how we
actually think about valuation error ("20% off" is a meaningful, comparable
mistake whether the player is worth 5M or 100M). We train and pick alphas
on this log scale, then invert with expm1 only when reporting euro-scale
metrics. log1p (not log) is used purely so a value of exactly 0 doesn't
produce -inf, though it does not occur in this dataset.

Why the train/test split is by season, not random
----------------------------------------------------
The same player appears in multiple seasons, and a player's stats and value
are highly autocorrelated year over year (a player worth 80M in 2023 is
very likely still worth roughly that in 2024). A random row-level split
would put, e.g., Saka-2023 in train and Saka-2024 in test - the model
would then be half-memorizing a specific player's value rather than
learning the general relationship between stats and value, and test
accuracy would look far better than it would on a genuinely new season.
Splitting by season (train on 2022-2024, test on 2025) mimics the actual
deployment scenario: predicting a season you have not seen the outcome of
yet.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    METRICS_JSON,
    MODEL_PATHS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PREDICTIONS_CSV,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SEASON,
    TRAIN_SEASONS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_N_ESTIMATORS,
)
from src.features import build_feature_matrix


def build_preprocessor() -> ColumnTransformer:
    """The shared preprocessing step for every model.

    Both branches impute before transforming (median for numeric, most
    frequent for the one categorical column) so a stray missing height
    doesn't crash the whole pipeline. OneHotEncoder(drop="first") drops one
    category as the implicit baseline - without this, a full one-hot
    encoding plus the model's intercept term are perfectly collinear (the
    "dummy variable trap"): the four position columns always sum to 1, which
    is a linear combination of the intercept's constant 1. LinearRegression
    won't error on this (it falls back to a minimum-norm least-squares
    solution) but the individual coefficients become arbitrary and
    uninterpretable, which matters here because we specifically want to read
    the coefficients off the linear model.
    """
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """One Pipeline per model, each wrapping the *same* preprocessing step.

    Wrapping StandardScaler/OneHotEncoder inside the Pipeline (rather than
    transforming X once outside and reusing the array) guarantees that
    predict.py applies the exact fitted transform - same means/std, same
    known categories - to a single new player row, instead of a hand-rolled
    copy of the transform that could quietly drift out of sync.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return {
        "linear_regression": Pipeline(
            [("pre", build_preprocessor()), ("model", LinearRegression())]
        ),
        "ridge": Pipeline(
            [
                ("pre", build_preprocessor()),
                (
                    "model",
                    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        # Shallow trees (max_depth=3) + a slow learning rate + 300 rounds is
        # the standard "don't overfit a small dataset" boosting recipe -
        # many weak trees correcting each other's residuals generalizes
        # better here than a few deep, high-variance trees would.
        "xgboost": Pipeline(
            [
                ("pre", build_preprocessor()),
                (
                    "model",
                    xgb.XGBRegressor(
                        n_estimators=XGBOOST_N_ESTIMATORS,
                        max_depth=XGBOOST_MAX_DEPTH,
                        learning_rate=XGBOOST_LEARNING_RATE,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _position_median_baseline(
    train_positions: pd.Series, train_values_eur: pd.Series, test_positions: pd.Series
) -> np.ndarray:
    """Predict the training-set median euro value for a row's position.

    This is the sanity-check baseline: if a model with real stats can't beat
    "just guess the typical value for a goalkeeper/defender/midfielder/
    attacker," it isn't learning anything from the season's performance.
    Medians (not means) come from TRAIN seasons only, so this baseline is
    exactly as blind to the test season as the real models are.
    """
    medians = train_values_eur.groupby(train_positions).median()
    overall_median = train_values_eur.median()
    return test_positions.map(medians).fillna(overall_median).to_numpy()


def _regression_metrics(y_true_eur: np.ndarray, y_pred_eur: np.ndarray) -> dict:
    return {
        "mae_eur": float(mean_absolute_error(y_true_eur, y_pred_eur)),
        "rmse_eur": float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))),
    }


def train_and_evaluate() -> dict:
    """Fit all three models on TRAIN_SEASONS, evaluate on TEST_SEASON.

    Returns the full metrics dict and, as a side effect, writes:
    - a fitted Pipeline per model to MODEL_PATHS (joblib)
    - reports/metrics.json (all metrics, for the README/dashboard)
    - reports/test_predictions.csv (every test-season row's predictions
      from every model, for the overvalued/undervalued chart and predict.py)
    """
    X, y, meta = build_feature_matrix()

    train_mask = meta["season"].isin(TRAIN_SEASONS)
    test_mask = meta["season"] == TEST_SEASON

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_test = meta[test_mask].reset_index(drop=True)

    print(f"[train] train rows: {len(X_train)} (seasons {TRAIN_SEASONS})")
    print(f"[train] test rows:  {len(X_test)} (season {TEST_SEASON})")

    y_test_eur = meta_test[TARGET_COL].to_numpy()

    metrics: dict = {}
    predictions = pd.DataFrame(meta_test)

    models = build_models()
    fitted = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

        y_pred_log = pipeline.predict(X_test)
        y_pred_eur = np.expm1(y_pred_log)
        # Never predict a negative transfer value from an expm1'd near-zero.
        y_pred_eur = np.clip(y_pred_eur, a_min=0, a_max=None)

        model_metrics = _regression_metrics(y_test_eur, y_pred_eur)
        model_metrics["r2_log_scale"] = float(r2_score(y_test, y_pred_log))
        metrics[name] = model_metrics
        predictions[f"predicted_eur_{name}"] = y_pred_eur

        print(f"[train] {name}: {model_metrics}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATHS[name])

    ridge_alpha = fitted["ridge"].named_steps["model"].alpha_
    metrics["ridge"]["selected_alpha"] = float(ridge_alpha)
    print(f"[train] ridge selected alpha (5-fold CV on train seasons): {ridge_alpha:.4g}")

    baseline_pred_eur = _position_median_baseline(
        X_train["position"], meta[train_mask][TARGET_COL], X_test["position"]
    )
    metrics["position_median_baseline"] = _regression_metrics(y_test_eur, baseline_pred_eur)
    metrics["position_median_baseline"]["r2_log_scale"] = float(
        r2_score(y_test, np.log1p(baseline_pred_eur))
    )
    predictions["predicted_eur_position_median_baseline"] = baseline_pred_eur
    print(f"[train] position_median_baseline: {metrics['position_median_baseline']}")

    metrics["train_seasons"] = TRAIN_SEASONS
    metrics["test_season"] = TEST_SEASON
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[train] wrote {METRICS_JSON} and {PREDICTIONS_CSV}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
