import pandas as pd
import pytest

from src.errors import PredictionError
from src.modeling import TrainConfig, train_model
from src.predictions import convert_inputs_to_model_units, predict_frame, validate_prediction_inputs
from src.synthetic_data import generate_synthetic_dataset


def _trained_payload():
    frame = generate_synthetic_dataset().rename(columns={"age_years": "age", "synthetic_metabolic_score": "outcome"})
    config = TrainConfig(
        task="regression",
        model_name="Linear Regression",
        target="outcome",
        features=["age", "bmi"],
        seed=1,
        test_size=0.25,
        dataset_type="synthetic",
        column_units={},
        group_column="participant_id",
    )
    result = train_model(frame, config)
    return {
        "app": "Insulin Matrix",
        "task": "regression",
        "features": ["age", "bmi"],
        "pipeline": result.pipeline,
        "classes": [],
        "dataset_type": "synthetic",
        "preprocessing": result.preprocessing,
        "units": {},
    }


def test_prediction_requires_the_trained_columns_and_does_not_refit():
    artifact = _trained_payload()
    before = artifact["pipeline"].named_steps["model"].coef_.copy()
    report = validate_prediction_inputs(pd.DataFrame({"age": [30]}), artifact, units_confirmed=True)
    assert report.has_errors()
    with pytest.raises(PredictionError, match="Missing required columns"):
        predict_frame(artifact, pd.DataFrame({"age": [30]}))
    scored = predict_frame(artifact, pd.DataFrame({"bmi": [27.0, 31.0], "age": [30, 55], "extra": [1, 2]}))
    after = artifact["pipeline"].named_steps["model"].coef_.copy()
    assert scored["predicted_value"].notna().all()
    assert list(scored["predicted_value"]) != []
    assert (before == after).all()
    assert scored.loc[0, "training_data"] == "synthetic_demonstration"


def test_unit_mismatch_is_rejected_until_conversion_is_requested():
    artifact = {
        "features": ["fasting_glucose"],
        "units": {"fasting_glucose": "mmol/L"},
        "preprocessing": {"numeric_features": ["fasting_glucose"], "categorical_features": []},
        "task": "regression",
        "pipeline": None,
    }
    frame = pd.DataFrame({"fasting_glucose": [90.0]})
    report = validate_prediction_inputs(
        frame,
        artifact,
        units_confirmed=True,
        provided_units={"fasting_glucose": "mg/dL"},
    )
    assert report.has_errors()
    assert "UNSUPPORTED_UNIT" in set(report.to_frame()["code"])
    converted, notes = convert_inputs_to_model_units(frame, artifact, {"fasting_glucose": "mg/dL"})
    assert converted.loc[0, "fasting_glucose"] == pytest.approx(5.0)
    assert notes
