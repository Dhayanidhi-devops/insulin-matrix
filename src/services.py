"""Workflows that connect calculations, files, and the SQLite history."""

from __future__ import annotations

import json

import pandas as pd

from src.config import artifact_dir, safe_under_home
from src.data_loader import apply_validation, load_synthetic, new_id, register_dataset, save_upload
from src.database import Database, dump_json, utc_now
from src.errors import ModelingError, StorageError
from src.feature_engineering import add_derived_features, add_normalized_units, column_units_for_features
from src.indices import INDEX_COLUMNS, calculate_indices
from src.modeling import TrainConfig, TrainResult, artifact_payload, load_artifact, save_artifact, train_model


def database() -> Database:
    return Database()


def prepare_training_frame(active: dict, derived: list[str], normalize_units: bool) -> tuple[pd.DataFrame, list[str], dict]:
    if active.get("validation") is None or active.get("analysis_frame") is None:
        raise ModelingError("Validate the mapped dataset before training.")
    if active["validation"].has_errors():
        raise ModelingError("Resolve validation errors before training. Rows were not deleted automatically.")
    frame = active["analysis_frame"].copy()
    messages: list[str] = []
    column_units: dict[str, str] = {}
    if normalize_units:
        frame, normalized_units, notes = add_normalized_units(frame, active["units"])
        column_units.update(normalized_units)
        messages.extend(notes)
    if derived:
        frame, notes = add_derived_features(frame, active["units"], derived)
        messages.extend(notes)
    return frame, messages, column_units


def train_and_record(active: dict, config: TrainConfig, preparation_messages: list[str] | None = None) -> tuple[TrainResult, str]:
    if active.get("validation") is None:
        raise ModelingError("Validate the mapped dataset before training.")
    if active["validation"].has_errors():
        raise ModelingError("Resolve validation errors before training.")
    result = train_model(config_frame(active, config), config)
    if preparation_messages:
        result.messages = preparation_messages + result.messages
    experiment_id = new_id("ex")
    artifact_id = new_id("md")
    created_at = utc_now()
    payload = artifact_payload(
        result,
        experiment_id=experiment_id,
        artifact_id=artifact_id,
        created_at=created_at,
        dataset_id=active["dataset_id"],
        dataset_label=active["display_name"],
    )
    relative = f"artifacts/models/{artifact_id}.joblib"
    path = artifact_dir() / f"{artifact_id}.joblib"
    save_artifact(payload, path)
    db = database()
    try:
        db.insert_artifact(artifact_id, experiment_id, relative)
        db.insert_experiment(
            {
                "experiment_id": experiment_id,
                "created_at": created_at,
                "dataset_id": active["dataset_id"],
                "dataset_label": active["display_name"],
                "dataset_type": active["dataset_type"],
                "task": result.task,
                "target": result.target,
                "features_json": dump_json(result.features),
                "model_name": result.model_name,
                "hyperparameters_json": dump_json(result.hyperparameters),
                "metrics_json": dump_json(result.metrics),
                "n_train_participants": result.n_train_participants,
                "n_test_participants": result.n_test_participants,
                "n_train_rows": result.n_train_rows,
                "n_test_rows": result.n_test_rows,
                "seed": result.seed,
                "test_size": result.test_size,
                "artifact_id": artifact_id,
                "split_mode": result.split_mode,
                "notes": " ".join(result.messages),
            }
        )
    except Exception:
        if path.is_file():
            path.unlink()
        raise
    return result, experiment_id


def config_frame(active: dict, config: TrainConfig) -> pd.DataFrame:
    """The training frame is supplied by the caller through config only.

    `train_and_record` receives a config whose features already exist on the
    frame stored in the config via a side channel. To keep the call explicit,
    the frame is read from `config` attribute set by the UI helper below.
    """
    frame = getattr(config, "frame", None)
    if frame is None:
        raise ModelingError("The training table is missing.")
    return frame


def remember_frame(config: TrainConfig, frame: pd.DataFrame) -> TrainConfig:
    config.frame = frame
    return config


def calculate_and_store(active: dict) -> dict:
    if active.get("validation") is None or active.get("analysis_frame") is None:
        raise ModelingError("Validate the mapped dataset before calculating indices.")
    if active["validation"].has_errors():
        raise ModelingError("Resolve validation errors before calculating indices.")
    units = active.get("units") or {}
    if not units.get("glucose") or not units.get("insulin"):
        raise ModelingError("Choose glucose and insulin units before calculating indices.")
    indices, summary = calculate_indices(active["analysis_frame"], units)
    count = int(indices.loc[:, list(INDEX_COLUMNS)].notna().to_numpy().sum()) if len(indices) else 0
    database().update_dataset(active["dataset_id"], n_index_values=count)
    updated = dict(active)
    updated["indices"] = indices
    updated["index_summary"] = summary
    updated["index_value_count"] = count
    return updated


def open_artifact(experiment_id: str) -> tuple[dict, dict]:
    db = database()
    experiment = db.get_experiment(experiment_id)
    if experiment is None:
        raise StorageError("That experiment is not in the history.")
    artifact_row = db.get_artifact(experiment["artifact_id"]) if experiment.get("artifact_id") else None
    if artifact_row is None:
        raise StorageError("The model artifact record is missing. Metrics remain in the history, but plots cannot be rebuilt.")
    path = safe_under_home(artifact_row["relpath"])
    if not path.is_file():
        raise StorageError("The saved model file is no longer available. The history row was kept.")
    return experiment, load_artifact(path)


def experiment_comparison(rows: list[dict] | None = None) -> pd.DataFrame:
    records = rows if rows is not None else database().list_experiments()
    table_rows = []
    metric_keys = ["mae", "rmse", "r2", "accuracy", "precision", "recall", "f1", "roc_auc"]
    for row in records:
        metrics = json.loads(row["metrics_json"]) if row.get("metrics_json") else {}
        item = {
            "experiment_id": row["experiment_id"],
            "created_at_utc": row["created_at"],
            "dataset_type": row["dataset_type"],
            "dataset_label": row["dataset_label"],
            "task": row["task"],
            "target": row["target"],
            "model": row["model_name"],
            "split": row.get("split_mode"),
            "train_participants": row["n_train_participants"],
            "test_participants": row["n_test_participants"],
        }
        for key in metric_keys:
            item[key] = metrics.get(key)
        table_rows.append(item)
    return pd.DataFrame(table_rows)


def configuration_from_artifact(artifact: dict) -> dict:
    return {
        "experiment_id": artifact.get("experiment_id"),
        "created_at_utc": artifact.get("created_at"),
        "dataset_id": artifact.get("dataset_id"),
        "dataset_type": artifact.get("dataset_type"),
        "task": artifact.get("task"),
        "target": artifact.get("target"),
        "features": artifact.get("features"),
        "model_name": artifact.get("model_name"),
        "hyperparameters": artifact.get("hyperparameters"),
        "preprocessing": artifact.get("preprocessing"),
        "column_units": artifact.get("units"),
        "seed": artifact.get("seed"),
        "test_size": artifact.get("test_size"),
        "split_mode": artifact.get("split_mode"),
        "metrics": artifact.get("metrics"),
        "disclaimer": artifact.get("disclaimer"),
    }


def attach_column_units(config: TrainConfig, features: list[str], base_units: dict, extra_units: dict) -> None:
    combined = dict(extra_units)
    combined.update(column_units_for_features(features, base_units))
    config.column_units = combined
