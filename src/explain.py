"""CLI: explain a single player's predicted market value with SHAP.

    python -m src.explain "Bukayo Saka" --season 2024

SHAP explanations are only wired up for SCOUTING_MODEL (xgboost). The tree
structure of a boosted model has no closed-form coefficients the way Ridge
does, so it needs SHAP's TreeExplainer to attribute a prediction back to
individual features. Ridge/LinearRegression don't need this module at all -
their own fitted coefficients (times the scaled feature value) already are
the exact, additive contribution per feature, no approximation required.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import shap

from config import MODEL_PATHS, SCOUTING_MODEL
from src.features import build_feature_matrix, find_player_row


def _humanize_feature_name(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to something a person (or an
    LLM agent narrating a scouting report) can read directly.

    ColumnTransformer prefixes every output column with its transformer's
    name, e.g. "num__goals_per_90" or "cat__position_Midfield" (see
    get_feature_names_out()). For the numeric branch this prefix is the only
    difference from the original column, so we just strip it. For the
    one-hot categorical branch, the column also encodes which category value
    it represents (e.g. "position_Midfield"), which we render as
    "position: Midfield" - a value it can take -in favor of leaving the
    raw one-hot column name, which nobody feeding this into a sentence wants
    to see.
    """
    if transformed_name.startswith("num__"):
        return transformed_name[len("num__") :]
    if transformed_name.startswith("cat__"):
        remainder = transformed_name[len("cat__") :]
        # remainder looks like "<original_column>_<category_value>", e.g.
        # "position_Midfield". Original categorical columns are known ahead
        # of time (CATEGORICAL_FEATURES), but this function is deliberately
        # kept generic/testable without importing config, so we just split
        # on the first underscore - true for every categorical feature this
        # project currently has (all single-word column names).
        if "_" in remainder:
            column, _, value = remainder.partition("_")
            return f"{column}: {value}"
        return remainder
    # Unrecognized prefix (e.g. a future transformer branch) - fall back to
    # returning it unchanged rather than raising, so this stays additive.
    return transformed_name


def explain_prediction(name: str, season: int, model_name: str = SCOUTING_MODEL) -> dict:
    """Return a SHAP-based breakdown of one player-season's predicted value.

    model_name defaults to (and, in practice, must be) SCOUTING_MODEL -
    xgboost. SHAP's TreeExplainer only supports tree-based models, so
    calling this with e.g. "ridge" or "linear_regression" is a usage error:
    it's rejected up front with a clear ValueError rather than letting shap
    raise its own (much less legible) exception deep inside TreeExplainer.
    The parameter exists mainly so callers/tests can demonstrate that
    rejection explicitly; there is no supported way to get a real SHAP
    explanation for a non-tree model here.

    Returns
    -------
    dict with keys:
        player, season, model : identifying info, echoed back.
        predicted_eur : model's predicted value in euros (expm1 of the
            model's log-scale output).
        base_value_log : the TreeExplainer's expected value on the log
            scale - roughly "what the model would predict with no
            information", the baseline SHAP contributions are added to.
        contributions : list of dicts, one per transformed feature, each
            with:
                feature : human-readable name (see _humanize_feature_name).
                raw_shap_log : the feature's SHAP value, in log-target
                    space (same units the model was trained in - see
                    features.py). This is the additive, exact
                    contribution: base_value_log + sum(raw_shap_log
                    across all features) == the model's raw log-scale
                    prediction (up to floating point error).
                direction : "positive" or "negative", i.e. did this
                    feature push the prediction up or down.
            Sorted by abs(raw_shap_log) descending, so the biggest drivers
            of the prediction (in either direction) come first.

    Deliberately NOT included: a per-feature EUR-scale swing (e.g. "removing
    this feature would change the prediction by X euros"). SHAP values are
    additive in log-space by construction (that's what makes TreeExplainer
    exact and fast), but expm1() is nonlinear, so "zero out this feature's
    log-contribution and re-expm1" is not a real counterfactual - it
    silently mixes in every other feature's log-scale value too, and the
    resulting euro delta would not sum across features to the actual
    prediction-vs-baseline euro gap. Presenting a precise-looking euro
    number that doesn't obey additivity risks being read as more exact than
    it is. Callers (dashboard, LLM agent) should describe raw_shap_log
    qualitatively - direction and relative magnitude ("age was the biggest
    negative driver of this valuation") - rather than as an exact euro
    figure.
    """
    if model_name != SCOUTING_MODEL:
        raise ValueError(
            f"SHAP explanations are only supported for the '{SCOUTING_MODEL}' model "
            f"(got '{model_name}'). SHAP's TreeExplainer requires a tree-based model; "
            "Ridge/LinearRegression should be explained via their own fitted "
            "coefficients instead."
        )

    if SCOUTING_MODEL not in MODEL_PATHS:
        raise ValueError(f"SCOUTING_MODEL '{SCOUTING_MODEL}' has no entry in MODEL_PATHS.")

    model_path = MODEL_PATHS[SCOUTING_MODEL]
    if not model_path.exists():
        raise ValueError(
            f"No trained model found for '{SCOUTING_MODEL}' at {model_path}. "
            "Run `python -m src.train` first."
        )

    pipeline = joblib.load(model_path)

    X, y, meta = build_feature_matrix()
    idx = find_player_row(meta, name, season)
    x_row = X.loc[[idx]]

    return _explain_row(pipeline, x_row, meta.loc[idx, "name"], int(meta.loc[idx, "season"]))


def _explain_row(pipeline, x_row: pd.DataFrame, resolved_name: str, resolved_season: int) -> dict:
    """Do the actual SHAP work for one already-resolved row. Split out from
    explain_prediction() so tests can exercise it against a small synthetic
    pipeline/model without needing the real dataset or trained model.
    """
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    x_transformed = pre.transform(x_row)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_transformed)
    # shap_values is (n_rows, n_features) for a single-output regressor;
    # we only ever pass one row in.
    row_shap = np.asarray(shap_values)[0]
    base_value_log = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    predicted_log = pipeline.predict(x_row)[0]
    predicted_eur = max(float(np.expm1(predicted_log)), 0.0)

    contributions = [
        {
            "feature": _humanize_feature_name(feature_names[i]),
            "raw_shap_log": float(row_shap[i]),
            "direction": "positive" if row_shap[i] >= 0 else "negative",
        }
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda c: abs(c["raw_shap_log"]), reverse=True)

    return {
        "player": resolved_name,
        "season": resolved_season,
        "model": SCOUTING_MODEL,
        "predicted_eur": predicted_eur,
        "base_value_log": base_value_log,
        "contributions": contributions,
    }


def _format_eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Player name (or substring), e.g. 'Bukayo Saka'")
    parser.add_argument("--season", type=int, required=True, help="Season start year, e.g. 2024")
    args = parser.parse_args(argv)

    try:
        result = explain_prediction(args.name, args.season)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"{result['player']} - {result['season']} season ({result['model']})")
    print(f"  Predicted value: {_format_eur(result['predicted_eur'])}")
    print(f"  Base value (log-scale): {result['base_value_log']:.3f}")
    print("  Top contributions (log-scale, additive, sorted by magnitude):")
    for c in result["contributions"]:
        sign = "+" if c["direction"] == "positive" else "-"
        print(f"    {sign} {c['feature']:<22} {c['raw_shap_log']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
