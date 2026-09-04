"""Generate the four evaluation charts into reports/, from the artifacts
train.py already wrote (reports/test_predictions.csv, a saved model). Kept
separate from train.py so re-running charts doesn't require re-fitting
models, and so train.py's core logic stays plotting-free and easy to test.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL,
    MODEL_PATHS,
    NUMERIC_FEATURES,
    PREDICTIONS_CSV,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SEASON,
)

PRED_COL = f"predicted_eur_{DEFAULT_MODEL}"


def _load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_CSV)


def plot_predicted_vs_actual(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    actual = df[TARGET_COL] / 1e6
    predicted = df[PRED_COL] / 1e6

    ax.scatter(actual, predicted, alpha=0.5, edgecolor="none")
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="crimson", linestyle="--", label="y = x (perfect prediction)")

    ax.set_xlabel("Actual market value (EUR millions)")
    ax.set_ylabel("Predicted market value (EUR millions)")
    ax.set_title(f"Predicted vs actual - {DEFAULT_MODEL}, {TEST_SEASON} season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame) -> None:
    predicted = df[PRED_COL] / 1e6
    residual = (df[PRED_COL] - df[TARGET_COL]) / 1e6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residual, alpha=0.5, edgecolor="none")
    ax.axhline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Predicted market value (EUR millions)")
    ax.set_ylabel("Residual: predicted - actual (EUR millions)")
    ax.set_title(f"Residuals vs predicted - {DEFAULT_MODEL}, {TEST_SEASON} season")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "residuals_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_coefficients() -> None:
    """Ridge's coefficients, not plain LinearRegression's.

    Several features here are correlated (goals and goals_per_90 both track
    playing time, age and age_squared are related by construction), and
    under collinearity OLS coefficients become unstable - small changes in
    the data can swing them wildly, even flip their sign, while the model's
    overall predictions barely change. Ridge's L2 penalty shrinks correlated
    coefficients toward each other rather than letting them fight for credit,
    so its coefficients are the more trustworthy ones to read for "what
    drives value."
    """
    pipeline = joblib.load(MODEL_PATHS["ridge"])
    preprocessor = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    cat_names = list(
        preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )
    feature_names = NUMERIC_FEATURES + cat_names
    coefs = pd.Series(model.coef_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["crimson" if c < 0 else "steelblue" for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ridge coefficient (on log1p(market value), standardized numeric features)")
    ax.set_title("What drives predicted value - Ridge coefficients")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ridge_coefficients.png", dpi=150)
    plt.close(fig)


def plot_over_under_valued(df: pd.DataFrame, n: int = 15) -> None:
    """The n most 'the market pays more than stats justify' and n most
    'the market pays less than stats justify' players in the test season.

    gap = actual - predicted. Positive gap: actual value exceeds what the
    model's stats-only view predicts -> market thinks this player is worth
    more than raw output suggests (reputation, hype, potential, marketing
    value...). Negative gap: model predicts more than the market currently
    pays -> "undervalued" by this narrow stats-based view.
    """
    gap = df[TARGET_COL] - df[PRED_COL]
    ranked = df.assign(gap_eur=gap).sort_values("gap_eur")

    undervalued = ranked.head(n)  # most negative gap: predicted >> actual
    overvalued = ranked.tail(n).sort_values("gap_eur", ascending=False)  # most positive gap

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].barh(overvalued["name"][::-1], overvalued["gap_eur"][::-1] / 1e6, color="crimson")
    axes[0].set_title(f"Most overvalued by the market ({TEST_SEASON})")
    axes[0].set_xlabel("Actual - predicted (EUR millions)")

    axes[1].barh(undervalued["name"][::-1], -undervalued["gap_eur"][::-1] / 1e6, color="steelblue")
    axes[1].set_title(f"Most undervalued by the market ({TEST_SEASON})")
    axes[1].set_xlabel("Predicted - actual (EUR millions)")

    fig.suptitle(f"Where the model and the market disagree most - {DEFAULT_MODEL}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "over_under_valued.png", dpi=150)
    plt.close(fig)


def generate_all_charts() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_predictions()
    plot_predicted_vs_actual(df)
    plot_residuals(df)
    plot_coefficients()
    plot_over_under_valued(df)
    print(f"[visualize] wrote 4 charts to {REPORTS_DIR}")


if __name__ == "__main__":
    generate_all_charts()
