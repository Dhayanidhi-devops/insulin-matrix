import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.errors import ModelingError
from src.feature_engineering import add_derived_features, missing_summary
from src.preprocessing import build_preprocessor, incomplete_feature_rows, split_participants
import pytest


def _panel():
    rows = []
    for participant in range(10):
        for visit in (1, 2):
            rows.append(
                {
                    "participant_id": f"P{participant}",
                    "observation_id": f"V{visit}",
                    "x": 1.0 if participant < 6 else 100.0,
                    "age": 30 + participant,
                }
            )
    return pd.DataFrame(rows)


def test_participant_split_keeps_visits_together():
    frame = _panel()
    first_train, first_test, first_info = split_participants(frame, "participant_id", 0.3, 7)
    second_train, second_test, _info = split_participants(frame, "participant_id", 0.3, 7)
    assert list(first_train) == list(second_train)
    assert list(first_test) == list(second_test)
    train_ids = set(frame.loc[first_train, "participant_id"])
    test_ids = set(frame.loc[first_test, "participant_id"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(frame["participant_id"])
    assert first_info["n_train_participants"] + first_info["n_test_participants"] == 10
    assert first_info["split_mode"] == "participant_grouped"
    for participant in frame["participant_id"].unique():
        visits = set(frame.loc[frame["participant_id"] == participant].index)
        assert visits <= set(first_train) or visits <= set(first_test)


def test_split_rejects_too_few_participants():
    frame = pd.DataFrame({"participant_id": ["A", "B", "C"], "x": [1, 2, 3]})
    with pytest.raises(ModelingError, match="At least 4"):
        split_participants(frame, "participant_id", 0.25, 1)


def test_imputer_learns_training_rows_only():
    train = pd.DataFrame({"x": [1.0, 1.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [100.0, 100.0, 100.0]})
    preprocessor = build_preprocessor(["x"], [], scale_numeric=False, numeric_impute="median")
    pipeline = Pipeline([("preprocess", preprocessor)])
    pipeline.fit(train)
    statistic = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
    assert statistic == pytest.approx(1.0)
    assert statistic != pytest.approx(pd.concat([train, test])["x"].median())
    transformed = pipeline.transform(test)
    assert transformed.shape == (3, 1)


def test_incomplete_row_count_does_not_drop_anything():
    frame = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, None]})
    assert incomplete_feature_rows(frame, ["a", "b"]) == 2
    assert len(frame) == 3
    summary = missing_summary(frame, ["a"])
    assert summary.loc[0, "missing"] == 1


def test_derived_homa_is_rowwise():
    frame = pd.DataFrame({"fasting_glucose": [5.0, 6.0], "fasting_insulin": [10.0, 8.0]})
    derived, messages = add_derived_features(frame, {"glucose": "mmol/L", "insulin": "µU/mL"}, ["homa_ir"])
    assert derived.loc[0, "homa_ir"] == pytest.approx(50 / 22.5)
    assert "homa_ir" in derived.columns
    assert messages == [] or isinstance(messages, list)
