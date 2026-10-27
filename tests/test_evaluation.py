import numpy as np
import pytest

from src.evaluation import classification_metrics, confusion_frame, regression_metrics


def test_regression_metrics_match_known_values():
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(0.5)


def test_regression_r2_is_absent_for_a_constant_target():
    metrics = regression_metrics([5, 5, 5], [4, 5, 6])
    assert metrics["r2"] is None
    assert "undefined" in metrics["r2_note"]


def test_classification_metrics_and_confusion_matrix():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], positive_class="1", class_labels=["0", "1"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1)
    assert metrics["f1"] == pytest.approx(0.8)
    matrix = confusion_frame(["0", "0", "1", "1"], ["0", "1", "1", "1"], ["0", "1"])
    assert matrix.to_numpy().tolist() == [[1, 1], [0, 2]]


def test_roc_auc_is_reported_only_when_probabilities_exist():
    proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])
    metrics = classification_metrics(["0", "0", "1", "1"], ["0", "1", "1", "1"], proba, positive_class="1", class_labels=["0", "1"])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    missing = classification_metrics(["0", "0", "1", "1"], ["0", "1", "1", "1"], None, positive_class="1", class_labels=["0", "1"])
    assert missing["roc_auc"] is None
    assert missing["roc_auc_note"]
