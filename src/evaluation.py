"""Held-out metrics. These functions score existing predictions; they do not fit models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def regression_metrics(y_true, y_pred) -> dict:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": None,
        "r2_note": "",
    }
    if len(actual) < 2 or float(np.nanstd(actual)) == 0.0:
        metrics["r2_note"] = "R² is undefined because the test target is constant or has fewer than two rows."
        return metrics
    metrics["r2"] = float(r2_score(actual, predicted))
    return metrics


def classification_metrics(y_true, y_pred, y_proba=None, positive_class: str | None = None, class_labels: list[str] | None = None) -> dict:
    actual = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    labels = class_labels or sorted(set(actual) | set(predicted))
    present = sorted(set(actual))
    metrics: dict = {
        "accuracy": float(accuracy_score(actual, predicted)) if len(actual) else None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "average": "binary" if len(present) == 2 else "weighted",
        "positive_class": positive_class,
        "roc_auc_note": "",
        "class_labels": labels,
    }
    if len(present) == 0:
        metrics["roc_auc_note"] = "No test labels were available."
        return metrics
    if len(present) == 1:
        metrics["roc_auc_note"] = (
            "Precision, recall, F1, and ROC-AUC need more than one class in the test partition. "
            "Only accuracy is reported."
        )
        return metrics
    average = "binary" if len(labels) == 2 and positive_class else "weighted"
    kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = positive_class
    metrics["precision"] = float(precision_score(actual, predicted, **kwargs))
    metrics["recall"] = float(recall_score(actual, predicted, **kwargs))
    metrics["f1"] = float(f1_score(actual, predicted, **kwargs))
    metrics["average"] = average
    if y_proba is None:
        metrics["roc_auc_note"] = "ROC-AUC was not calculated because the model did not supply class probabilities."
        return metrics
    proba = np.asarray(y_proba, dtype=float)
    try:
        if len(labels) == 2 and positive_class in labels:
            score = proba[:, labels.index(positive_class)]
            binary_actual = (actual == positive_class).astype(int)
            if len(np.unique(binary_actual)) < 2:
                metrics["roc_auc_note"] = "ROC-AUC was not calculated because the test partition contains one class."
            else:
                metrics["roc_auc"] = float(roc_auc_score(binary_actual, score))
        elif proba.ndim == 2 and proba.shape[1] == len(labels):
            metrics["roc_auc"] = float(
                roc_auc_score(actual, proba, multi_class="ovr", average="weighted", labels=labels)
            )
        else:
            metrics["roc_auc_note"] = "ROC-AUC was not calculated because the probability columns do not match the classes."
    except ValueError as exc:
        metrics["roc_auc_note"] = f"ROC-AUC was not calculated: {exc}"
    return metrics


def confusion_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual: {label}" for label in labels],
        columns=[f"predicted: {label}" for label in labels],
    )


def metrics_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, value in metrics.items():
        if key in {"class_labels"}:
            value = ", ".join(value) if isinstance(value, list) else value
        rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)
