"""Feature selection, leakage checks, and target derivation."""

from __future__ import annotations

import pandas as pd

from app.config import (
    CATEGORICAL_FEATURES,
    CRITERIA_FLAG_COLUMN,
    DIABETES_CLASSES,
    ID_COLUMN,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_TYPE_COLUMN,
    TARGET_COLUMN,
)


def derive_diabetes_category(frame: pd.DataFrame) -> pd.Series:
    """Map raw synthetic columns into the four screening classes.

    Type 1 / Type 2 come from the synthetic type label. Remaining rows are
    labeled Prediabetes or No Diabetes using fasting glucose / HbA1c cut-points
    consistent with common screening thresholds (demonstration only).
    """
    raw = frame[RAW_TYPE_COLUMN].astype(str)
    glucose = pd.to_numeric(frame["Fasting_Glucose_mg_dL"], errors="coerce")
    hba1c = pd.to_numeric(frame["HbA1c_percent"], errors="coerce")

    labels = pd.Series(index=frame.index, dtype="object")
    labels[raw.str.contains("Type 1", case=False, na=False)] = "Type 1 Diabetes"
    labels[raw.str.contains("Type 2", case=False, na=False)] = "Type 2 Diabetes"

    remaining = labels.isna()
    prediabetes = remaining & (
        (glucose >= 100) | (hba1c >= 5.7)
    )
    labels[prediabetes] = "Prediabetes"
    labels[labels.isna()] = "No Diabetes"
    return labels.astype(str)


def prepare_modeling_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy the workbook, derive the target, and keep modeling columns."""
    frame = raw.copy()
    if TARGET_COLUMN not in frame.columns:
        if RAW_TYPE_COLUMN not in frame.columns:
            raise ValueError(
                f"Dataset must include '{RAW_TYPE_COLUMN}' or '{TARGET_COLUMN}'."
            )
        frame[TARGET_COLUMN] = derive_diabetes_category(frame)
    return frame


def available_feature_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in frame.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    extras = [
        c
        for c in frame.columns
        if c not in LEAKAGE_COLUMNS
        and c not in numeric
        and c not in categorical
        and c != TARGET_COLUMN
    ]
    return {
        "numeric": numeric,
        "categorical": categorical,
        "other": extras,
        "default": numeric + categorical,
        "excluded_leakage": sorted(LEAKAGE_COLUMNS & set(frame.columns)),
    }


def validate_features(features: list[str], frame: pd.DataFrame) -> list[str]:
    if not features:
        raise ValueError("Select at least one feature.")
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"Unknown features: {', '.join(missing)}")
    leaked = [f for f in features if f in LEAKAGE_COLUMNS]
    if leaked:
        raise ValueError(
            "Refusing target leakage / identifier features: " + ", ".join(leaked)
        )
    if TARGET_COLUMN in features:
        raise ValueError("Target column cannot be used as a feature.")
    return list(features)


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    if TARGET_COLUMN not in frame.columns:
        return {}
    counts = frame[TARGET_COLUMN].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def rare_class_warnings(counts: dict[str, int], minimum: int = 10) -> list[str]:
    warnings: list[str] = []
    for label in DIABETES_CLASSES:
        n = int(counts.get(label, 0))
        if n == 0:
            warnings.append(f"Class '{label}' is absent from the loaded dataset.")
        elif n < minimum:
            warnings.append(
                f"Class '{label}' has only {n} rows — metrics for this class "
                "will be unstable and stratified splits may be limited."
            )
    return warnings


def dataset_profile(frame: pd.DataFrame) -> dict:
    features = available_feature_columns(frame)
    counts = class_distribution(frame)
    numeric_summary = {}
    for col in features["numeric"]:
        series = pd.to_numeric(frame[col], errors="coerce")
        numeric_summary[col] = {
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
            "missing": int(series.isna().sum()),
        }
    categorical_summary = {}
    for col in features["categorical"]:
        categorical_summary[col] = {
            str(k): int(v)
            for k, v in frame[col].fillna("(missing)").value_counts().items()
        }
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "columns": list(frame.columns),
        "features": features,
        "class_counts": counts,
        "rare_class_warnings": rare_class_warnings(counts),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "id_column_present": ID_COLUMN in frame.columns,
        "criteria_flag_present": CRITERIA_FLAG_COLUMN in frame.columns,
    }
