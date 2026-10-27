import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_synthetic
from src.database import Database
from src.errors import ModelingError
from src.modeling import TrainConfig, available_models, load_artifact, train_model, xgboost_available
from src.predictions import predict_frame
from src.services import prepare_training_frame, remember_frame, train_and_record
from src.synthetic_data import SYNTHETIC_FEATURES


def test_model_catalog_lists_required_estimators_and_optional_xgboost():
    regression = {model["name"] for model in available_models("regression")}
    classification = {model["name"] for model in available_models("classification")}
    assert {"Linear Regression", "Ridge Regression", "Random Forest Regressor", "Gradient Boosting Regressor"} <= regression
    assert {"Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"} <= classification
    assert ("XGBoost Regressor" in regression) is xgboost_available()


def test_too_few_participants_is_explained():
    frame = pd.DataFrame(
        {
            "participant_id": [f"P{i}" for i in range(5)],
            "age": [40, 41, 42, 43, 44],
            "outcome": [1.0, 1.2, 1.1, 1.4, 1.3],
        }
    )
    config = TrainConfig(
        task="regression",
        model_name="Linear Regression",
        target="outcome",
        features=["age"],
        group_column="participant_id",
    )
    with pytest.raises(ModelingError, match="reliable evaluation"):
        train_model(frame, config)


def test_synthetic_regression_and_classification_round_trip(home):
    database = Database()
    active = load_synthetic(database)
    frame, messages, units = prepare_training_frame(active, [], False)
    features = [feature for feature in SYNTHETIC_FEATURES if feature in frame.columns]
    regression = TrainConfig(
        task="regression",
        model_name="Ridge Regression",
        target="synthetic_metabolic_score",
        features=features,
        seed=42,
        test_size=0.25,
        hyperparameters={"alpha": 1.0},
        scale_numeric=True,
        dataset_type="synthetic",
        units=active["units"],
        column_units=units,
        group_column="participant_id",
    )
    remember_frame(regression, frame)
    result, experiment_id = train_and_record(active, regression, messages)
    assert result.split_mode == "participant_grouped"
    assert result.n_train_participants + result.n_test_participants == 80
    assert result.metrics["mae"] >= 0
    assert result.metrics["rmse"] >= result.metrics["mae"]
    assert result.metrics["r2"] is not None
    assert result.metrics["metric_set"] == "held_out_test_partition"
    assert any("not clinical evidence" in message for message in result.messages)

    stored = database.get_experiment(experiment_id)
    artifact_row = database.get_artifact(stored["artifact_id"])
    payload = load_artifact(home / artifact_row["relpath"])
    assert payload["app"] == "Insulin Matrix"
    coefficient_before = np.array(payload["pipeline"].named_steps["model"].coef_, copy=True)
    sample = frame.loc[:, features].head(3)
    predictions = predict_frame(payload, sample.iloc[:, ::-1])
    coefficient_after = np.array(payload["pipeline"].named_steps["model"].coef_, copy=True)
    assert np.allclose(coefficient_before, coefficient_after)
    assert len(predictions) == 3
    assert "predicted_value" in predictions.columns

    classification = TrainConfig(
        task="classification",
        model_name="Logistic Regression",
        target="synthetic_risk_group",
        features=features,
        seed=42,
        test_size=0.25,
        hyperparameters={"C": 1.0},
        scale_numeric=True,
        dataset_type="synthetic",
        positive_class="higher",
        group_column="participant_id",
    )
    class_result = train_model(frame, classification)
    assert class_result.metrics["accuracy"] is not None
    assert class_result.metrics["f1"] is not None
    assert class_result.positive_class == "higher"
    assert class_result.n_test_participants >= 2
