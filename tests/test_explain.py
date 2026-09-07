"""Tests for the SHAP explainability module (src/explain.py).

Test strategy: rather than depending on the real, ~4,700-row production
dataset and the previously-committed models/xgboost.joblib being present
(which would make this test slow and fragile in CI - it wouldn't run at all
on a fresh checkout before `python -m src.build_dataset` / `train` have been
run), the end-to-end test below fits a tiny XGBRegressor inside a real
sklearn Pipeline (same "pre" + "model" shape as train.py produces) on a
handful of synthetic rows. This exercises the *actual* code path -
ColumnTransformer.transform(), get_feature_names_out() naming, shap.
TreeExplainer against a real fitted xgboost model, name remapping - without
requiring any project data files or a pre-trained model artifact on disk.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.explain import _explain_row, _humanize_feature_name, explain_prediction


# --- _humanize_feature_name ------------------------------------------------


def test_humanize_numeric_feature_strips_prefix():
    assert _humanize_feature_name("num__goals_per_90") == "goals_per_90"
    assert _humanize_feature_name("num__age_at_season_end") == "age_at_season_end"


def test_humanize_categorical_feature_maps_to_readable_label():
    assert _humanize_feature_name("cat__position_Midfield") == "position: Midfield"
    assert _humanize_feature_name("cat__position_Goalkeeper") == "position: Goalkeeper"


def test_humanize_unrecognized_prefix_falls_back_unchanged():
    # Defensive: a future transformer branch shouldn't raise, just pass through.
    assert _humanize_feature_name("weird__thing") == "weird__thing"


# --- explain_prediction model-name guard ------------------------------------


def test_explain_prediction_rejects_non_scouting_model():
    with pytest.raises(ValueError, match="only supported for the 'xgboost' model"):
        explain_prediction("Anyone", 2024, model_name="ridge")


# --- end-to-end against a tiny synthetic pipeline ---------------------------


def _tiny_pipeline_and_data():
    """Build a Pipeline with the same "pre" + "model" shape train.py
    produces (ColumnTransformer -> imputer/scaler/one-hot, then an
    XGBRegressor), fit on a small synthetic frame.
    """
    numeric_features = ["goals_per_90", "age_at_season_end"]
    categorical_features = ["position"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    pipeline = Pipeline([("pre", pre), ("model", model)])

    rng = np.random.default_rng(42)
    n = 30
    X = pd.DataFrame(
        {
            "goals_per_90": rng.uniform(0, 1, n),
            "age_at_season_end": rng.uniform(18, 34, n),
            "position": rng.choice(["Attack", "Midfield", "Defender"], n),
        }
    )
    # Made-up but directionally sensible target: more goals -> more log-value.
    y = 15.0 + 0.5 * X["goals_per_90"] + rng.normal(0, 0.05, n)

    pipeline.fit(X, y)
    return pipeline, X


def test_explain_row_end_to_end_returns_additive_shap_contributions():
    pipeline, X = _tiny_pipeline_and_data()
    x_row = X.iloc[[0]]

    result = _explain_row(pipeline, x_row, resolved_name="Synthetic Player", resolved_season=2024)

    assert result["player"] == "Synthetic Player"
    assert result["season"] == 2024
    assert result["model"] == "xgboost"
    assert result["predicted_eur"] >= 0.0
    assert isinstance(result["base_value_log"], float)

    contributions = result["contributions"]
    assert len(contributions) > 0
    for c in contributions:
        assert set(c.keys()) == {"feature", "raw_shap_log", "direction"}
        assert c["direction"] in ("positive", "negative")

    # Sorted by |raw_shap_log| descending.
    magnitudes = [abs(c["raw_shap_log"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)

    # SHAP additivity: base_value + sum(shap values) == raw model output
    # (on the log scale, before expm1) up to floating point tolerance.
    predicted_log = pipeline.predict(x_row)[0]
    reconstructed = result["base_value_log"] + sum(c["raw_shap_log"] for c in contributions)
    assert reconstructed == pytest.approx(predicted_log, abs=1e-3)

    # Human-readable names, not raw ColumnTransformer output names.
    feature_names = {c["feature"] for c in contributions}
    assert "goals_per_90" in feature_names
    assert "age_at_season_end" in feature_names
    assert any(f.startswith("position:") for f in feature_names)
