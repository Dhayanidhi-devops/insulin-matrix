"""Preprocessing pipelines fitted on the training partition only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from app.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def split_feature_types(
    frame: pd.DataFrame, features: list[str]
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for feature in features:
        if feature in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(frame[feature]):
            # Prefer declared numeric list, else infer.
            if feature in CATEGORICAL_FEATURES:
                categorical.append(feature)
            else:
                numeric.append(feature)
        elif feature in CATEGORICAL_FEATURES:
            categorical.append(feature)
        else:
            parsed = pd.to_numeric(frame[feature], errors="coerce")
            original = frame[feature].notna()
            if int(original.sum()) and int((original & parsed.notna()).sum()) == int(
                original.sum()
            ):
                numeric.append(feature)
            else:
                categorical.append(feature)
    return numeric, categorical


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool = True,
    numeric_impute: str = "median",
) -> ColumnTransformer:
    transformers: list[tuple] = []
    if numeric_columns:
        steps: list[tuple] = [("imputer", SimpleImputer(strategy=numeric_impute))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric_columns))
    if categorical_columns:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", cat_pipe, categorical_columns))
    if not transformers:
        raise ValueError("No usable feature columns for preprocessing.")
    return ColumnTransformer(transformers, remainder="drop")


def stratified_split(
    frame: pd.DataFrame,
    features: list[str],
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    y = frame[TARGET_COLUMN].astype(str)
    counts = y.value_counts()
    # Stratify only when every class has at least 2 samples.
    can_stratify = bool((counts >= 2).all()) and counts.shape[0] > 1
    X = frame[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if can_stratify else None,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "stratified": can_stratify,
        "warnings": []
        if can_stratify
        else [
            "Stratified split disabled because at least one class has fewer than "
            "2 samples."
        ],
    }


def fit_label_encoder(y_train: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(y_train.astype(str))
    return encoder


def transform_matrix(
    preprocessor: ColumnTransformer, frame: pd.DataFrame
) -> np.ndarray:
    return np.asarray(preprocessor.transform(frame), dtype=np.float32)


def preprocessing_report(
    frame: pd.DataFrame,
    features: list[str],
    *,
    scale_numeric: bool = True,
    numeric_impute: str = "median",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    numeric, categorical = split_feature_types(frame, features)
    missing_before = {
        col: int(frame[col].isna().sum()) for col in features if frame[col].isna().any()
    }
    split = stratified_split(frame, features, test_size=test_size, seed=seed)
    preprocessor = build_preprocessor(
        numeric, categorical, scale_numeric=scale_numeric, numeric_impute=numeric_impute
    )
    preprocessor.fit(split["X_train"])
    X_train_t = transform_matrix(preprocessor, split["X_train"])
    X_test_t = transform_matrix(preprocessor, split["X_test"])

    return {
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "scale_numeric": scale_numeric,
        "numeric_impute": numeric_impute,
        "missing_before": missing_before,
        "n_train": int(len(split["X_train"])),
        "n_test": int(len(split["X_test"])),
        "stratified": split["stratified"],
        "warnings": split["warnings"],
        "transformed_train_shape": list(X_train_t.shape),
        "transformed_test_shape": list(X_test_t.shape),
        "class_counts_train": {
            str(k): int(v) for k, v in split["y_train"].value_counts().items()
        },
        "class_counts_test": {
            str(k): int(v) for k, v in split["y_test"].value_counts().items()
        },
        "preprocessor": preprocessor,
        "split": split,
    }
