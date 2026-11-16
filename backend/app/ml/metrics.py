"""Classification metrics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_classification_metrics(
    y_true,
    y_pred,
    *,
    labels: list[str],
    y_proba=None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    present = sorted(set(y_true) | set(y_pred))
    ordered = [lab for lab in labels if lab in present] or present

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": ordered,
            "matrix": confusion_matrix(y_true, y_pred, labels=ordered).tolist(),
        },
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=ordered,
            output_dict=True,
            zero_division=0,
        ),
        "support": {str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
    }

    if y_proba is not None:
        try:
            proba = np.asarray(y_proba, dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(
                        y_true,
                        proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=ordered,
                    )
                )
        except Exception:
            metrics["roc_auc_ovr_weighted"] = None

    return metrics
