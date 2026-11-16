"""Trainable ANN using TensorFlow/Keras, with sklearn MLP fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.ml.metrics import safe_classification_metrics
from app.ml.pipeline import (
    build_preprocessor,
    fit_label_encoder,
    split_feature_types,
    stratified_split,
    transform_matrix,
)

_TF_ERROR: str | None = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Keep TF quieter / lighter for local demos.
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    TF_AVAILABLE = False
    _TF_ERROR = str(exc)
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore


def tensorflow_available() -> bool:
    return TF_AVAILABLE


def tensorflow_status() -> dict[str, Any]:
    return {
        "available": TF_AVAILABLE,
        "error": _TF_ERROR,
        "backend": "tensorflow" if TF_AVAILABLE else "sklearn_mlp_fallback",
    }


def _build_keras_model(
    n_features: int,
    n_classes: int,
    *,
    hidden_layers: list[int],
    dropout: float,
    learning_rate: float,
) -> Any:
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    if n_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


@dataclass
class TrainedANNBundle:
    model_name: str
    backend: str
    preprocessor: Any
    model: Any
    label_encoder: LabelEncoder
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
    history: dict[str, list[float]] = field(default_factory=dict)
    family: str = "ann"

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        X = frame[self.features]
        Xt = transform_matrix(self.preprocessor, X)

        if self.backend == "tensorflow":
            proba_raw = np.asarray(self.model.predict(Xt, verbose=0))
            if proba_raw.ndim == 1 or proba_raw.shape[1] == 1:
                # binary sigmoid
                p1 = proba_raw.reshape(-1)
                classes = list(self.label_encoder.classes_)
                probabilities = [
                    {str(classes[0]): float(1 - p), str(classes[1]): float(p)}
                    for p in p1
                ]
                pred_idx = (p1 >= 0.5).astype(int)
            else:
                pred_idx = np.argmax(proba_raw, axis=1)
                classes = list(self.label_encoder.classes_)
                probabilities = [
                    {str(classes[i]): float(row[i]) for i in range(len(classes))}
                    for row in proba_raw
                ]
            labels = self.label_encoder.inverse_transform(pred_idx)
        else:
            pipe: Pipeline = self.model
            labels = np.asarray(pipe.predict(X)).astype(str)
            probabilities = None
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                proba = pipe.predict_proba(X)
                classes = [str(c) for c in pipe.named_steps["model"].classes_]
                probabilities = [
                    {classes[i]: float(row[i]) for i in range(len(classes))}
                    for row in proba
                ]

        return {
            "predictions": [str(x) for x in labels],
            "probabilities": probabilities,
            "class_labels": self.class_labels,
        }


def train_ann(
    frame: pd.DataFrame,
    *,
    features: list[str],
    hidden_layers: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 25,
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
    scale_numeric: bool = True,
    numeric_impute: str = "median",
) -> TrainedANNBundle:
    hidden_layers = hidden_layers or [64, 32]
    numeric, categorical = split_feature_types(frame, features)
    split = stratified_split(frame, features, test_size=test_size, seed=seed)
    preprocessor = build_preprocessor(
        numeric, categorical, scale_numeric=scale_numeric, numeric_impute=numeric_impute
    )
    preprocessor.fit(split["X_train"])
    label_encoder = fit_label_encoder(split["y_train"])
    class_labels = [str(c) for c in label_encoder.classes_]
    n_classes = len(class_labels)

    X_train = transform_matrix(preprocessor, split["X_train"])
    X_test = transform_matrix(preprocessor, split["X_test"])
    y_train = label_encoder.transform(split["y_train"].astype(str))
    y_test = label_encoder.transform(split["y_test"].astype(str))

    hyperparameters = {
        "hidden_layers": hidden_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    history: dict[str, list[float]] = {}

    if TF_AVAILABLE:
        keras.utils.set_random_seed(seed)
        model = _build_keras_model(
            X_train.shape[1],
            n_classes,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
        )
        # Class weights for rare classes
        class_weight = None
        values, counts = np.unique(y_train, return_counts=True)
        if len(values) > 1:
            total = float(counts.sum())
            class_weight = {
                int(v): float(total / (len(values) * c)) for v, c in zip(values, counts)
            }
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]
        fit = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )
        history = {k: [float(x) for x in v] for k, v in fit.history.items()}
        proba_raw = np.asarray(model.predict(X_test, verbose=0))
        if proba_raw.ndim == 1 or proba_raw.shape[1] == 1:
            p1 = proba_raw.reshape(-1)
            pred_idx = (p1 >= 0.5).astype(int)
            y_proba = np.column_stack([1 - p1, p1])
        else:
            pred_idx = np.argmax(proba_raw, axis=1)
            y_proba = proba_raw
        y_pred = label_encoder.inverse_transform(pred_idx)
        metrics = safe_classification_metrics(
            split["y_test"].astype(str),
            y_pred,
            labels=class_labels,
            y_proba=y_proba,
        )
        bundle_model = model
        backend = "tensorflow"
    else:
        # Fallback MLP — still a real trainable ANN, just not Keras.
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=max(epochs * 4, 50),
            random_state=seed,
        )
        pipe = Pipeline([("preprocess", preprocessor), ("model", mlp)])
        pipe.fit(split["X_train"], split["y_train"].astype(str))
        y_pred = pipe.predict(split["X_test"])
        y_proba = pipe.predict_proba(split["X_test"]) if hasattr(mlp, "predict_proba") else None
        metrics = safe_classification_metrics(
            split["y_test"].astype(str),
            y_pred,
            labels=class_labels,
            y_proba=y_proba,
        )
        history = {
            "loss": [float(x) for x in getattr(mlp, "loss_curve_", [])],
        }
        bundle_model = pipe
        backend = "sklearn_mlp_fallback"
        hyperparameters["fallback_reason"] = _TF_ERROR

    metrics["split_stratified"] = split["stratified"]
    metrics["split_warnings"] = split["warnings"]
    metrics["ann_backend"] = backend

    return TrainedANNBundle(
        model_name="ANN",
        backend=backend,
        preprocessor=preprocessor,
        model=bundle_model,
        label_encoder=label_encoder,
        features=features,
        numeric_features=numeric,
        categorical_features=categorical,
        class_labels=class_labels,
        metrics=metrics,
        hyperparameters=hyperparameters,
        n_train=int(len(split["X_train"])),
        n_test=int(len(split["X_test"])),
        seed=seed,
        test_size=test_size,
        history=history,
    )


def save_ann_bundle(bundle: TrainedANNBundle, path) -> None:
    """Persist ANN bundle. Keras models are saved beside the joblib meta file."""
    path = path
    meta_path = str(path)
    if bundle.backend == "tensorflow":
        model_dir = str(path) + ".keras"
        bundle.model.save(model_dir)
        payload = {
            "model_name": bundle.model_name,
            "backend": bundle.backend,
            "preprocessor": bundle.preprocessor,
            "model_path": model_dir,
            "label_encoder": bundle.label_encoder,
            "features": bundle.features,
            "numeric_features": bundle.numeric_features,
            "categorical_features": bundle.categorical_features,
            "class_labels": bundle.class_labels,
            "metrics": bundle.metrics,
            "hyperparameters": bundle.hyperparameters,
            "n_train": bundle.n_train,
            "n_test": bundle.n_test,
            "seed": bundle.seed,
            "test_size": bundle.test_size,
            "history": bundle.history,
            "family": bundle.family,
        }
        joblib.dump(payload, meta_path)
    else:
        joblib.dump(bundle, meta_path)


def load_ann_bundle(path) -> TrainedANNBundle:
    obj = joblib.load(path)
    if isinstance(obj, TrainedANNBundle):
        return obj
    if obj.get("backend") == "tensorflow":
        model = keras.models.load_model(obj["model_path"])
        return TrainedANNBundle(
            model_name=obj["model_name"],
            backend=obj["backend"],
            preprocessor=obj["preprocessor"],
            model=model,
            label_encoder=obj["label_encoder"],
            features=obj["features"],
            numeric_features=obj["numeric_features"],
            categorical_features=obj["categorical_features"],
            class_labels=obj["class_labels"],
            metrics=obj["metrics"],
            hyperparameters=obj["hyperparameters"],
            n_train=obj["n_train"],
            n_test=obj["n_test"],
            seed=obj["seed"],
            test_size=obj["test_size"],
            history=obj.get("history", {}),
        )
    raise ValueError("Unrecognized ANN artifact format.")
