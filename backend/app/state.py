"""In-memory runtime state shared across API routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.database import Database
from app.ml.ann import TrainedANNBundle
from app.ml.models import TrainedModelBundle


@dataclass
class AppState:
    db: Database = field(default_factory=Database)
    frame: pd.DataFrame | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    is_synthetic: bool = True
    preprocess_config: dict[str, Any] = field(default_factory=dict)
    preprocess_report: dict[str, Any] | None = None
    trained_models: dict[str, TrainedModelBundle | TrainedANNBundle] = field(
        default_factory=dict
    )
    active_model_name: str | None = None
    settings: dict[str, Any] = field(
        default_factory=lambda: {
            "default_test_size": 0.2,
            "default_seed": 42,
            "scale_numeric": True,
            "numeric_impute": "median",
            "auto_load_sample": True,
        }
    )

    def has_dataset(self) -> bool:
        return self.frame is not None and len(self.frame) > 0

    def require_frame(self) -> pd.DataFrame:
        if self.frame is None or self.frame.empty:
            raise ValueError("No dataset loaded. Load the sample dataset or upload one.")
        return self.frame

    def get_model(self, name: str | None = None):
        key = name or self.active_model_name
        if not key or key not in self.trained_models:
            raise ValueError("No trained model selected. Train a model first.")
        return self.trained_models[key]

    def comparison_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name, bundle in self.trained_models.items():
            metrics = bundle.metrics
            rows.append(
                {
                    "model_name": name,
                    "family": getattr(bundle, "family", "classical"),
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_weighted": metrics.get("f1_weighted"),
                    "roc_auc_ovr_weighted": metrics.get("roc_auc_ovr_weighted"),
                    "n_train": bundle.n_train,
                    "n_test": bundle.n_test,
                    "class_labels": bundle.class_labels,
                }
            )
        rows.sort(key=lambda r: (r.get("f1_macro") or 0), reverse=True)
        return rows


STATE = AppState()
