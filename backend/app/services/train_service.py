"""Training orchestration for classical models and ANN."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.config import DEFAULT_FEATURES, MEDICAL_DISCLAIMER, artifact_dir
from app.database import utc_now
from app.ml.ann import (
    save_ann_bundle,
    tensorflow_status,
    train_ann,
)
from app.ml.features import validate_features
from app.ml.models import (
    available_classical_models,
    save_bundle,
    train_classical_model,
    xgboost_available,
)
from app.ml.pipeline import preprocessing_report
from app.state import STATE


def list_model_catalog() -> dict:
    return {
        "classical": available_classical_models(),
        "ann": {
            "name": "ANN",
            "key": "ann",
            "family": "ann",
            "tensorflow": tensorflow_status(),
            "params": [
                {"key": "hidden_layers", "default": [64, 32], "type": "int_list"},
                {"key": "dropout", "default": 0.2, "type": "float"},
                {"key": "learning_rate", "default": 0.001, "type": "float"},
                {"key": "epochs", "default": 25, "type": "int"},
                {"key": "batch_size", "default": 64, "type": "int"},
            ],
        },
        "xgboost_available": xgboost_available(),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def run_preprocessing(
    features: list[str] | None = None,
    *,
    scale_numeric: bool | None = None,
    numeric_impute: str | None = None,
    test_size: float | None = None,
    seed: int | None = None,
) -> dict:
    frame = STATE.require_frame()
    feats = validate_features(features or DEFAULT_FEATURES, frame)
    scale = (
        STATE.settings["scale_numeric"] if scale_numeric is None else scale_numeric
    )
    impute = (
        STATE.settings["numeric_impute"] if numeric_impute is None else numeric_impute
    )
    ts = STATE.settings["default_test_size"] if test_size is None else test_size
    sd = STATE.settings["default_seed"] if seed is None else seed

    report = preprocessing_report(
        frame,
        feats,
        scale_numeric=scale,
        numeric_impute=impute,
        test_size=ts,
        seed=sd,
    )
    # Drop non-serializable objects for API response / stored report.
    serializable = {
        k: v
        for k, v in report.items()
        if k not in {"preprocessor", "split"}
    }
    STATE.preprocess_config = {
        "features": feats,
        "scale_numeric": scale,
        "numeric_impute": impute,
        "test_size": ts,
        "seed": sd,
    }
    STATE.preprocess_report = serializable
    return {
        **serializable,
        "disclaimer": MEDICAL_DISCLAIMER,
        "is_synthetic": STATE.is_synthetic,
    }


def _persist_run(bundle, *, notes: str = "") -> str:
    run_id = uuid.uuid4().hex[:12]
    artifact_path = artifact_dir() / f"{run_id}_{bundle.model_name.replace(' ', '_')}.joblib"
    if getattr(bundle, "family", "classical") == "ann":
        save_ann_bundle(bundle, artifact_path)
    else:
        save_bundle(bundle, artifact_path)

    try:
        from app.config import storage_home

        relpath = str(Path(artifact_path).relative_to(storage_home()))
    except Exception:
        relpath = str(artifact_path)

    STATE.db.insert_training_run(
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "dataset_id": STATE.dataset_id,
            "model_name": bundle.model_name,
            "model_family": getattr(bundle, "family", "classical"),
            "features_json": json.dumps(bundle.features),
            "hyperparameters_json": json.dumps(bundle.hyperparameters),
            "metrics_json": json.dumps(bundle.metrics),
            "class_labels_json": json.dumps(bundle.class_labels),
            "artifact_relpath": relpath,
            "n_train": bundle.n_train,
            "n_test": bundle.n_test,
            "seed": bundle.seed,
            "test_size": bundle.test_size,
            "notes": notes,
        }
    )
    return run_id


def train_models(
    model_names: list[str],
    *,
    features: list[str] | None = None,
    hyperparameters: dict | None = None,
    test_size: float | None = None,
    seed: int | None = None,
) -> dict:
    frame = STATE.require_frame()
    feats = validate_features(
        features
        or STATE.preprocess_config.get("features")
        or DEFAULT_FEATURES,
        frame,
    )
    ts = (
        test_size
        if test_size is not None
        else STATE.preprocess_config.get("test_size", STATE.settings["default_test_size"])
    )
    sd = (
        seed
        if seed is not None
        else STATE.preprocess_config.get("seed", STATE.settings["default_seed"])
    )
    scale = STATE.preprocess_config.get("scale_numeric", STATE.settings["scale_numeric"])
    impute = STATE.preprocess_config.get(
        "numeric_impute", STATE.settings["numeric_impute"]
    )
    hyperparameters = hyperparameters or {}

    if not model_names:
        raise ValueError("Select at least one model to train.")

    results = []
    for name in model_names:
        hp = hyperparameters.get(name, hyperparameters.get(name.lower(), {}))
        bundle = train_classical_model(
            frame,
            model_name=name,
            features=feats,
            hyperparameters=hp,
            test_size=ts,
            seed=sd,
            scale_numeric=scale,
            numeric_impute=impute,
        )
        run_id = _persist_run(bundle)
        STATE.trained_models[name] = bundle
        STATE.active_model_name = name
        results.append(
            {
                "run_id": run_id,
                "model_name": name,
                "family": "classical",
                "metrics": bundle.metrics,
                "n_train": bundle.n_train,
                "n_test": bundle.n_test,
                "class_labels": bundle.class_labels,
                "hyperparameters": bundle.hyperparameters,
            }
        )

    return {
        "trained": results,
        "active_model": STATE.active_model_name,
        "comparison": STATE.comparison_rows(),
        "disclaimer": MEDICAL_DISCLAIMER,
        "is_synthetic": STATE.is_synthetic,
    }


def train_ann_model(
    *,
    features: list[str] | None = None,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 25,
    batch_size: int = 64,
    test_size: float | None = None,
    seed: int | None = None,
) -> dict:
    frame = STATE.require_frame()
    feats = validate_features(
        features
        or STATE.preprocess_config.get("features")
        or DEFAULT_FEATURES,
        frame,
    )
    ts = (
        test_size
        if test_size is not None
        else STATE.preprocess_config.get("test_size", STATE.settings["default_test_size"])
    )
    sd = (
        seed
        if seed is not None
        else STATE.preprocess_config.get("seed", STATE.settings["default_seed"])
    )
    scale = STATE.preprocess_config.get("scale_numeric", STATE.settings["scale_numeric"])
    impute = STATE.preprocess_config.get(
        "numeric_impute", STATE.settings["numeric_impute"]
    )

    bundle = train_ann(
        frame,
        features=feats,
        hidden_layers=hidden_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        test_size=ts,
        seed=sd,
        scale_numeric=scale,
        numeric_impute=impute,
    )
    run_id = _persist_run(bundle, notes=f"ANN backend={bundle.backend}")
    STATE.trained_models["ANN"] = bundle
    STATE.active_model_name = "ANN"
    return {
        "run_id": run_id,
        "model_name": "ANN",
        "family": "ann",
        "backend": bundle.backend,
        "metrics": bundle.metrics,
        "history": bundle.history,
        "hyperparameters": bundle.hyperparameters,
        "n_train": bundle.n_train,
        "n_test": bundle.n_test,
        "class_labels": bundle.class_labels,
        "comparison": STATE.comparison_rows(),
        "disclaimer": MEDICAL_DISCLAIMER,
        "is_synthetic": STATE.is_synthetic,
        "tensorflow": tensorflow_status(),
    }


def training_status() -> dict:
    return {
        "has_dataset": STATE.has_dataset(),
        "preprocess_ready": STATE.preprocess_report is not None,
        "preprocess_report": STATE.preprocess_report,
        "trained_models": list(STATE.trained_models.keys()),
        "active_model": STATE.active_model_name,
        "comparison": STATE.comparison_rows(),
        "catalog": list_model_catalog(),
        "disclaimer": MEDICAL_DISCLAIMER,
    }
