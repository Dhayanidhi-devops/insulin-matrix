"""Train, save, and reload scikit-learn models for Insulin Matrix.

The saved object includes the preprocessing pipeline. Prediction reuses that
pipeline and does not fit it again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.config import APP_NAME, MIN_PARTICIPANTS_FOR_TRAINING, VERSION
from src.errors import ModelingError
from src.evaluation import classification_metrics, regression_metrics
from src.preprocessing import (
    build_preprocessor,
    coerce_model_frame,
    constant_features,
    incomplete_feature_rows,
    rows_with_missing_target,
    split_feature_types,
    split_independent_rows,
    split_participants,
)
from src.validation import leakage_issues

MODEL_SPECS = [
    {"name": "Linear Regression", "task": "regression", "params": []},
    {"name": "Ridge Regression", "task": "regression", "params": [{"key": "alpha", "default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.1, "kind": "float"}]},
    {"name": "Random Forest Regressor", "task": "regression", "params": [
        {"key": "n_estimators", "default": 200, "min": 10, "max": 500, "step": 10, "kind": "int"},
        {"key": "max_depth", "default": 6, "min": 0, "max": 30, "step": 1, "kind": "int"},
    ]},
    {"name": "Gradient Boosting Regressor", "task": "regression", "params": [
        {"key": "n_estimators", "default": 150, "min": 10, "max": 400, "step": 10, "kind": "int"},
        {"key": "learning_rate", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "kind": "float"},
        {"key": "max_depth", "default": 3, "min": 1, "max": 8, "step": 1, "kind": "int"},
    ]},
    {"name": "Logistic Regression", "task": "classification", "params": [{"key": "C", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1, "kind": "float"}]},
    {"name": "Random Forest Classifier", "task": "classification", "params": [
        {"key": "n_estimators", "default": 200, "min": 10, "max": 500, "step": 10, "kind": "int"},
        {"key": "max_depth", "default": 6, "min": 0, "max": 30, "step": 1, "kind": "int"},
    ]},
    {"name": "Gradient Boosting Classifier", "task": "classification", "params": [
        {"key": "n_estimators", "default": 150, "min": 10, "max": 400, "step": 10, "kind": "int"},
        {"key": "learning_rate", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "kind": "float"},
        {"key": "max_depth", "default": 3, "min": 1, "max": 8, "step": 1, "kind": "int"},
    ]},
]


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def available_models(task: str) -> list[dict]:
    models = [spec for spec in MODEL_SPECS if spec["task"] == task]
    if xgboost_available():
        if task == "regression":
            models.append(
                {
                    "name": "XGBoost Regressor",
                    "task": "regression",
                    "params": [
                        {"key": "n_estimators", "default": 150, "min": 10, "max": 400, "step": 10, "kind": "int"},
                        {"key": "learning_rate", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "kind": "float"},
                        {"key": "max_depth", "default": 3, "min": 1, "max": 8, "step": 1, "kind": "int"},
                    ],
                }
            )
        else:
            models.append(
                {
                    "name": "XGBoost Classifier",
                    "task": "classification",
                    "params": [
                        {"key": "n_estimators", "default": 150, "min": 10, "max": 400, "step": 10, "kind": "int"},
                        {"key": "learning_rate", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "kind": "float"},
                        {"key": "max_depth", "default": 3, "min": 1, "max": 8, "step": 1, "kind": "int"},
                    ],
                }
            )
    return models


def _depth(value):
    if value is None or int(value) <= 0:
        return None
    return int(value)


def build_estimator(name: str, hyperparameters: dict, seed: int, class_weight_balanced: bool):
    params = hyperparameters or {}
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge Regression":
        return Ridge(alpha=float(params.get("alpha", 1.0)))
    if name == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=_depth(params.get("max_depth", 6)),
            random_state=seed,
            n_jobs=1,
        )
    if name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=seed,
        )
    if name == "Logistic Regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            class_weight="balanced" if class_weight_balanced else None,
        )
    if name == "Random Forest Classifier":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=_depth(params.get("max_depth", 6)),
            random_state=seed,
            n_jobs=1,
            class_weight="balanced" if class_weight_balanced else None,
        )
    if name == "Gradient Boosting Classifier":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=seed,
        )
    if name in {"XGBoost Regressor", "XGBoost Classifier"}:
        if not xgboost_available():
            raise ModelingError("XGBoost is not installed. Choose another model. The rest of Insulin Matrix does not need it.")
        from xgboost import XGBClassifier, XGBRegressor

        shared = dict(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=seed,
            n_jobs=1,
        )
        if name == "XGBoost Regressor":
            return XGBRegressor(objective="reg:squarederror", **shared)
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **shared)
    raise ModelingError(f"Unknown model '{name}'.")


def jsonable_params(estimator) -> dict:
    clean = {}
    for key, value in estimator.get_params(deep=False).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
    return clean


@dataclass
class TrainConfig:
    task: str
    model_name: str
    target: str
    features: list[str]
    seed: int = 42
    test_size: float = 0.25
    hyperparameters: dict = field(default_factory=dict)
    scale_numeric: bool = True
    numeric_impute: str = "median"
    missing_strategy: str = "pipeline_impute"
    drop_exact_duplicates: bool = False
    group_column: str | None = "participant_id"
    allow_row_split: bool = False
    positive_class: str | None = None
    class_weight_balanced: bool = False
    dataset_type: str = "real"
    units: dict = field(default_factory=dict)
    column_units: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    metrics: dict
    messages: list[str]
    pipeline: Pipeline
    evaluation: dict
    importance: pd.DataFrame | None
    permutation: pd.DataFrame | None
    preprocessing: dict
    hyperparameters: dict
    features: list[str]
    task: str
    model_name: str
    target: str
    split_mode: str
    n_train_participants: int
    n_test_participants: int
    n_train_rows: int
    n_test_rows: int
    seed: int
    test_size: float
    class_labels: list[str]
    positive_class: str | None
    column_units: dict
    dataset_type: str


def _normalize_target(series: pd.Series, task: str) -> pd.Series:
    if task == "regression":
        converted = pd.to_numeric(series, errors="coerce")
        failed = int((series.notna() & converted.isna()).sum())
        if failed:
            raise ModelingError(f"The regression target has {failed} values that are not numbers.")
        return converted
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        present = numeric.dropna()
        if len(present) and present.eq(present.round()).all():
            labels = numeric.round().astype("Int64").astype(str)
            labels = labels.where(numeric.notna(), other=pd.NA)
            return labels
    return series.astype("string").astype(object).where(series.notna(), other=np.nan)


def _choose_positive_class(labels: list[str], requested: str | None) -> str | None:
    if len(labels) != 2:
        return None
    if requested and requested in labels:
        return requested
    if "higher" in labels:
        return "higher"
    if "1" in labels:
        return "1"
    return sorted(labels)[-1]


def train_model(frame: pd.DataFrame, config: TrainConfig) -> TrainResult:
    """Fit one model on a participant holdout and score the untouched participants."""
    if config.task not in {"regression", "classification"}:
        raise ModelingError("Choose regression or classification.")
    if not config.target:
        raise ModelingError("Choose a target variable.")
    if config.target not in frame.columns:
        raise ModelingError(f"The target '{config.target}' is not on the analysis table.")
    features = [feature for feature in config.features if feature != config.target]
    missing_features = [feature for feature in features if feature not in frame.columns]
    if missing_features:
        raise ModelingError(
            "These features are not on the training table: "
            + ", ".join(missing_features)
            + ". Add them as derived columns or remove them from the feature list."
        )
    if not features:
        raise ModelingError("Select at least one feature that is not the target.")
    issues = leakage_issues(features, config.target)
    errors = [issue.message for issue in issues if issue.severity == "error"]
    if errors:
        raise ModelingError(" ".join(errors))
    messages = [issue.message for issue in issues if issue.severity == "warning"]
    if config.dataset_type == "synthetic" and config.target in {"synthetic_metabolic_score", "synthetic_risk_group"}:
        if any(feature in {"fasting_glucose", "fasting_insulin", "homa_ir"} for feature in features):
            messages.append(
                "The synthetic target is generated from fasting glucose, fasting insulin, age, and BMI. "
                "A model that uses those inputs can recover the generator. That is not clinical evidence."
            )

    working, coerce_messages = coerce_model_frame(frame, features)
    messages.extend(coerce_messages)
    working[config.target] = _normalize_target(working[config.target], config.task)
    before_target = len(working)
    missing_target = rows_with_missing_target(working, config.target)
    if missing_target:
        messages.append(f"{missing_target} rows were excluded because the target is missing.")
        working = working.loc[working[config.target].notna()].copy()
    if config.drop_exact_duplicates:
        duplicate_count = int(working.duplicated().sum())
        if duplicate_count:
            messages.append(f"{duplicate_count} exact duplicate rows were removed because that option was selected.")
            working = working.drop_duplicates().copy()
    elif int(working.duplicated().sum()) and not config.group_column:
        raise ModelingError(
            "Exact duplicate rows are present. Remove them explicitly on the processing page, or map a participant ID, before splitting rows."
        )
    if config.missing_strategy == "drop_incomplete":
        drop_count = incomplete_feature_rows(working, features)
        if drop_count:
            messages.append(
                f"{drop_count} rows were excluded because a selected feature is missing and the drop-incomplete option was selected."
            )
            working = working.loc[~working.loc[:, features].isna().any(axis=1)].copy()
    elif config.missing_strategy != "pipeline_impute":
        raise ModelingError("Missing-value handling must be pipeline imputation or an explicit drop of incomplete rows.")
    group_column = config.group_column if config.group_column in working.columns else None
    if group_column:
        n_participants = int(working[group_column].astype(str).nunique())
        if n_participants < MIN_PARTICIPANTS_FOR_TRAINING:
            raise ModelingError(
                f"Only {n_participants} independent participants are available. "
                f"At least {MIN_PARTICIPANTS_FOR_TRAINING} are required so a participant holdout can be interpreted. "
                "A reliable evaluation cannot be performed on this sample."
            )
    if len(working) < MIN_PARTICIPANTS_FOR_TRAINING:
        raise ModelingError(
            f"Only {len(working)} rows remain after the reported exclusions (started with {before_target}). "
            "That is not enough for a held-out evaluation."
        )
    constants = constant_features(working, features)
    if constants:
        raise ModelingError(
            "These features have a single value and were not removed automatically: " + ", ".join(constants) + "."
        )

    numeric, categorical = split_feature_types(working, features)
    if config.task == "classification":
        class_counts = working[config.target].astype(str).value_counts()
        if len(class_counts) < 2:
            raise ModelingError("The target has only one class, so classification metrics cannot be estimated.")
        if int(class_counts.min()) < 2:
            raise ModelingError(
                "A class has fewer than two observations. A reliable classification evaluation cannot be performed."
            )
    y = working[config.target]
    if config.task == "regression":
        y = y.astype(float)
    X = working.loc[:, features]
    stratify = y if config.task == "classification" else None
    if group_column:
        train_index, test_index, split_info = split_participants(working, group_column, config.test_size, config.seed, stratify)
    elif config.allow_row_split:
        if len(working) < MIN_PARTICIPANTS_FOR_TRAINING:
            raise ModelingError(
                f"Only {len(working)} rows are available. At least {MIN_PARTICIPANTS_FOR_TRAINING} independent rows are required."
            )
        train_index, test_index, split_info = split_independent_rows(working, config.test_size, config.seed, stratify)
    else:
        raise ModelingError(
            "No participant ID is available. Insulin Matrix will not split repeated measurements by row "
            "unless you confirm that each row is an independent participant."
        )
    messages.extend(split_info["messages"])

    estimator = build_estimator(config.model_name, config.hyperparameters, config.seed, config.class_weight_balanced)
    preprocessor = build_preprocessor(numeric, categorical, scale_numeric=config.scale_numeric, numeric_impute=config.numeric_impute)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    if config.task == "classification" and y_test.astype(str).nunique() < 2:
        raise ModelingError(
            "The test partition contains one class. Change the seed or the test proportion. "
            "A reliable classification evaluation cannot be performed on this split."
        )
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    proba = None
    class_labels: list[str] = []
    positive = None
    if config.task == "regression":
        holdout = regression_metrics(y_test, test_pred)
        training_scores = regression_metrics(y_train, train_pred)
        metrics = {
            "mae": holdout["mae"],
            "rmse": holdout["rmse"],
            "r2": holdout["r2"],
            "r2_note": holdout["r2_note"],
            "train_mae": training_scores["mae"],
            "train_rmse": training_scores["rmse"],
            "train_r2": training_scores["r2"],
        }
    else:
        class_labels = [str(label) for label in pipeline.named_steps["model"].classes_]
        positive = _choose_positive_class(class_labels, config.positive_class)
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
        holdout = classification_metrics(y_test, test_pred, proba, positive, class_labels)
        train_pred_labels = pipeline.predict(X_train)
        training_scores = classification_metrics(y_train, train_pred_labels, None, positive, class_labels)
        metrics = {
            "accuracy": holdout["accuracy"],
            "precision": holdout["precision"],
            "recall": holdout["recall"],
            "f1": holdout["f1"],
            "roc_auc": holdout["roc_auc"],
            "roc_auc_note": holdout["roc_auc_note"],
            "average": holdout["average"],
            "positive_class": positive,
            "train_accuracy": training_scores["accuracy"],
            "train_f1": training_scores["f1"],
        }
    metrics["split_mode"] = split_info["split_mode"]
    metrics["metric_set"] = "held_out_test_partition"

    cv_stats, cv_note = _cross_validate(pipeline, X_train, y_train, working.loc[train_index, group_column] if group_column else None, config.task)
    if cv_note:
        messages.append(cv_note)
    if cv_stats:
        metrics.update(cv_stats)

    importance = _native_importance(pipeline)
    permutation = _permutation(pipeline, X_test, y_test, config)
    if permutation is None and len(X_test) > 2000:
        messages.append("Permutation importance was skipped because the test partition has more than 2000 rows.")

    evaluation = {
        "y_true": [None if pd.isna(value) else value for value in (y_test.astype(float).tolist() if config.task == "regression" else y_test.astype(str).tolist())],
        "y_pred": [None if pd.isna(value) else value for value in (np.asarray(test_pred, dtype=float).tolist() if config.task == "regression" else np.asarray(test_pred).astype(str).tolist())],
        "y_proba": None if proba is None else np.asarray(proba, dtype=float).tolist(),
        "positive_class": positive,
        "classes": class_labels,
        "task": config.task,
    }
    preprocessing = {
        "numeric_impute": config.numeric_impute,
        "scale_numeric": config.scale_numeric,
        "missing_strategy": config.missing_strategy,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "drop_exact_duplicates": config.drop_exact_duplicates,
    }
    return TrainResult(
        metrics=metrics,
        messages=messages,
        pipeline=pipeline,
        evaluation=evaluation,
        importance=importance,
        permutation=permutation,
        preprocessing=preprocessing,
        hyperparameters=jsonable_params(pipeline.named_steps["model"]),
        features=features,
        task=config.task,
        model_name=config.model_name,
        target=config.target,
        split_mode=split_info["split_mode"],
        n_train_participants=int(split_info["n_train_participants"]),
        n_test_participants=int(split_info["n_test_participants"]),
        n_train_rows=int(split_info["n_train_rows"]),
        n_test_rows=int(split_info["n_test_rows"]),
        seed=config.seed,
        test_size=config.test_size,
        class_labels=class_labels,
        positive_class=positive,
        column_units=dict(config.column_units),
        dataset_type=config.dataset_type,
    )


def _cross_validate(fitted_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series | None, task: str):
    if groups is None:
        return None, "Grouped cross-validation was not run because rows were treated as independent participants."
    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)
    group_values = groups.astype(str).reset_index(drop=True).to_numpy()
    n_groups = len(set(group_values))
    n_splits = min(5, n_groups // 2)
    if n_splits < 2:
        return None, "Grouped cross-validation was not run because the training partition has fewer than 4 participants."
    if task == "classification":
        label_counts = pd.Series(y_reset.astype(str).to_numpy()).groupby(group_values).first().value_counts()
        if int(label_counts.min()) < n_splits:
            n_splits = int(label_counts.min())
        if n_splits < 2:
            return None, "Grouped cross-validation was not run because a class has fewer than 2 participants in the training partition."
        splitter = StratifiedGroupKFold(n_splits=n_splits)
    else:
        splitter = GroupKFold(n_splits=n_splits)
    scores = []
    try:
        for train_pos, test_pos in splitter.split(X_reset, y_reset, group_values):
            if set(group_values[train_pos]) & set(group_values[test_pos]):
                raise ModelingError("Cross-validation placed one participant on both sides of a fold.")
            model = clone(fitted_pipeline)
            model.fit(X_reset.iloc[train_pos], y_reset.iloc[train_pos])
            predicted = model.predict(X_reset.iloc[test_pos])
            if task == "regression":
                from sklearn.metrics import r2_score

                scores.append(float(r2_score(y_reset.iloc[test_pos], predicted)))
            else:
                from sklearn.metrics import f1_score

                scores.append(float(f1_score(y_reset.iloc[test_pos].astype(str), np.asarray(predicted).astype(str), average="weighted", zero_division=0)))
    except Exception as exc:
        return None, f"Grouped cross-validation was not run: {exc}"
    name = "r2" if task == "regression" else "f1_weighted"
    return {
        "cv_scope": "training_partition_only",
        "cv_splits": n_splits,
        "cv_metric_name": name,
        "cv_metric_mean": float(np.mean(scores)),
        "cv_metric_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }, None


def _native_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    try:
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "model_importance"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim == 2 else np.abs(coefficients)
        kind = "absolute_coefficient"
    else:
        return None
    if len(names) != len(values):
        return None
    table = pd.DataFrame({"feature": names, "importance": values, "kind": kind})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def _permutation(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: TrainConfig) -> pd.DataFrame | None:
    if len(X_test) > 2000 or X_test.shape[1] > 40:
        return None
    scoring = "r2" if config.task == "regression" else "f1_weighted"
    try:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=config.seed,
            scoring=scoring,
            n_jobs=1,
        )
    except Exception:
        return None
    table = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def artifact_payload(result: TrainResult, *, experiment_id: str, artifact_id: str, created_at: str, dataset_id: str, dataset_label: str) -> dict:
    return {
        "format": "insulin-matrix-model-v1",
        "app": APP_NAME,
        "version": VERSION,
        "experiment_id": experiment_id,
        "artifact_id": artifact_id,
        "created_at": created_at,
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "dataset_type": result.dataset_type,
        "task": result.task,
        "model_name": result.model_name,
        "target": result.target,
        "features": result.features,
        "hyperparameters": result.hyperparameters,
        "preprocessing": result.preprocessing,
        "units": result.column_units,
        "metrics": result.metrics,
        "messages": result.messages,
        "classes": result.class_labels,
        "positive_class": result.positive_class,
        "pipeline": result.pipeline,
        "evaluation": result.evaluation,
        "importance_records": None if result.importance is None else result.importance.to_dict(orient="records"),
        "permutation_records": None if result.permutation is None else result.permutation.to_dict(orient="records"),
        "split_mode": result.split_mode,
        "seed": result.seed,
        "test_size": result.test_size,
        "disclaimer": "Research prototype. Not a diagnostic device. Not for treatment or insulin dosing.",
    }


def save_artifact(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: Path) -> dict:
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ModelingError("The saved model could not be read.") from exc
    if not isinstance(payload, dict) or payload.get("app") != APP_NAME or "pipeline" not in payload:
        raise ModelingError("This file is not an Insulin Matrix model artifact.")
    return payload
