"""Classical scikit-learn model registry and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.ml.metrics import safe_classification_metrics
from app.ml.pipeline import (
    build_preprocessor,
    fit_label_encoder,
    split_feature_types,
    stratified_split,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGBOOST_AVAILABLE = False


CLASSICAL_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Logistic Regression",
        "key": "logistic_regression",
        "family": "classical",
        "params": [
            {"key": "C", "default": 1.0, "type": "float", "min": 0.01, "max": 100.0},
        ],
    },
    {
        "name": "Decision Tree",
        "key": "decision_tree",
        "family": "classical",
        "params": [
            {"key": "max_depth", "default": 8, "type": "int", "min": 1, "max": 30},
        ],
    },
    {
        "name": "Random Forest",
        "key": "random_forest",
        "family": "classical",
        "params": [
            {"key": "n_estimators", "default": 200, "type": "int", "min": 50, "max": 500},
            {"key": "max_depth", "default": 10, "type": "int", "min": 0, "max": 40},
        ],
    },
    {
        "name": "SVM",
        "key": "svm",
        "family": "classical",
        "params": [
            {"key": "C", "default": 1.0, "type": "float", "min": 0.01, "max": 100.0},
        ],
    },
    {
        "name": "KNN",
        "key": "knn",
        "family": "classical",
        "params": [
            {"key": "n_neighbors", "default": 15, "type": "int", "min": 1, "max": 101},
        ],
    },
    {
        "name": "Gradient Boosting",
        "key": "gradient_boosting",
        "family": "classical",
        "params": [
            {"key": "n_estimators", "default": 150, "type": "int", "min": 50, "max": 400},
            {"key": "learning_rate", "default": 0.1, "type": "float", "min": 0.01, "max": 0.5},
            {"key": "max_depth", "default": 3, "type": "int", "min": 1, "max": 8},
        ],
    },
]


def xgboost_available() -> bool:
    return XGBOOST_AVAILABLE


def available_classical_models() -> list[dict[str, Any]]:
    models = list(CLASSICAL_MODEL_SPECS)
    if xgboost_available():
        models.append(
            {
                "name": "XGBoost",
                "key": "xgboost",
                "family": "classical",
                "params": [
                    {
                        "key": "n_estimators",
                        "default": 150,
                        "type": "int",
                        "min": 50,
                        "max": 400,
                    },
                    {
                        "key": "learning_rate",
                        "default": 0.1,
                        "type": "float",
                        "min": 0.01,
                        "max": 0.5,
                    },
                    {
                        "key": "max_depth",
                        "default": 4,
                        "type": "int",
                        "min": 1,
                        "max": 10,
                    },
                ],
            }
        )
    return models


def _depth(value) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def build_estimator(name: str, hyperparameters: dict | None, seed: int):
    params = hyperparameters or {}
    if name == "Logistic Regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=_depth(params.get("max_depth", 8)),
            class_weight="balanced",
            random_state=seed,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=_depth(params.get("max_depth", 10)),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if name == "SVM":
        return SVC(
            C=float(params.get("C", 1.0)),
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 15)),
            weights="distance",
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=seed,
        )
    if name == "XGBoost":
        if not xgboost_available():
            raise ValueError("XGBoost is not installed.")
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 4)),
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


@dataclass
class TrainedModelBundle:
    model_name: str
    pipeline: Pipeline
    label_encoder: Any
    features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    class_labels: list[str]
    metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    n_train: int
    n_test: int
    seed: int
    test_size: float
    family: str = "classical"
    extras: dict[str, Any] = field(default_factory=dict)

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        X = frame[self.features]
        pred_idx = self.pipeline.predict(X)
        # Some estimators already emit string labels via pipeline without encoder.
        if getattr(self.label_encoder, "classes_", None) is not None:
            # XGBoost path encodes integers; sklearn classifiers may use strings.
            classes = list(self.label_encoder.classes_)
            if np.issubdtype(np.asarray(pred_idx).dtype, np.integer):
                labels = self.label_encoder.inverse_transform(np.asarray(pred_idx))
            else:
                labels = np.asarray(pred_idx).astype(str)
        else:
            classes = self.class_labels
            labels = np.asarray(pred_idx).astype(str)

        probabilities = None
        estimator = self.pipeline.named_steps["model"]
        if hasattr(estimator, "predict_proba"):
            proba = self.pipeline.predict_proba(X)
            # Align probability columns to class_labels order when possible.
            model_classes = [str(c) for c in getattr(estimator, "classes_", classes)]
            if np.issubdtype(np.asarray(model_classes).dtype, np.number) or (
                model_classes and str(model_classes[0]).isdigit()
            ):
                # Integer-encoded classes (XGBoost)
                model_classes = [
                    str(self.label_encoder.inverse_transform([int(c)])[0])
                    for c in model_classes
                ]
            probabilities = []
            for row in proba:
                probabilities.append(
                    {model_classes[i]: float(row[i]) for i in range(len(model_classes))}
                )
        return {
            "predictions": [str(x) for x in labels],
            "probabilities": probabilities,
            "class_labels": self.class_labels,
        }


def train_classical_model(
    frame: pd.DataFrame,
    *,
    model_name: str,
    features: list[str],
    hyperparameters: dict | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    scale_numeric: bool = True,
    numeric_impute: str = "median",
) -> TrainedModelBundle:
    numeric, categorical = split_feature_types(frame, features)
    split = stratified_split(frame, features, test_size=test_size, seed=seed)
    preprocessor = build_preprocessor(
        numeric, categorical, scale_numeric=scale_numeric, numeric_impute=numeric_impute
    )
    label_encoder = fit_label_encoder(split["y_train"])
    y_train = split["y_train"].astype(str)
    y_test = split["y_test"].astype(str)

    estimator = build_estimator(model_name, hyperparameters, seed)
    # XGBoost needs integer labels.
    if model_name == "XGBoost":
        y_train_fit = label_encoder.transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
    else:
        y_train_fit = y_train
        y_test_enc = None

    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    pipe.fit(split["X_train"], y_train_fit)

    raw_pred = pipe.predict(split["X_test"])
    if model_name == "XGBoost":
        y_pred = label_encoder.inverse_transform(np.asarray(raw_pred).astype(int))
    else:
        y_pred = np.asarray(raw_pred).astype(str)

    y_proba = None
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_proba = pipe.predict_proba(split["X_test"])

    class_labels = [str(c) for c in label_encoder.classes_]
    metrics = safe_classification_metrics(
        y_test, y_pred, labels=class_labels, y_proba=y_proba
    )
    metrics["split_stratified"] = split["stratified"]
    metrics["split_warnings"] = split["warnings"]

    # Lightweight permutation importance on a sample of the test set.
    try:
        sample_n = min(400, len(split["X_test"]))
        X_sample = split["X_test"].iloc[:sample_n]
        y_sample = y_test.iloc[:sample_n]
        if model_name == "XGBoost":
            y_sample_fit = label_encoder.transform(y_sample)
            scoring = "accuracy"
            result = permutation_importance(
                pipe,
                X_sample,
                y_sample_fit,
                n_repeats=3,
                random_state=seed,
                scoring=scoring,
                n_jobs=1,
            )
        else:
            result = permutation_importance(
                pipe,
                X_sample,
                y_sample,
                n_repeats=3,
                random_state=seed,
                scoring="accuracy",
                n_jobs=1,
            )
        importance = sorted(
            [
                {"feature": features[i], "importance": float(result.importances_mean[i])}
                for i in range(len(features))
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )
        metrics["feature_importance"] = importance
    except Exception as exc:  # pragma: no cover
        metrics["feature_importance"] = []
        metrics["importance_error"] = str(exc)

    return TrainedModelBundle(
        model_name=model_name,
        pipeline=pipe,
        label_encoder=label_encoder,
        features=features,
        numeric_features=numeric,
        categorical_features=categorical,
        class_labels=class_labels,
        metrics=metrics,
        hyperparameters=hyperparameters or {},
        n_train=int(len(split["X_train"])),
        n_test=int(len(split["X_test"])),
        seed=seed,
        test_size=test_size,
        family="classical",
        extras={"warnings": split["warnings"]},
    )


def save_bundle(bundle: TrainedModelBundle, path) -> None:
    joblib.dump(bundle, path)


def load_bundle(path) -> TrainedModelBundle:
    return joblib.load(path)


def clone_estimator_from_bundle(bundle: TrainedModelBundle):
    return clone(bundle.pipeline)
