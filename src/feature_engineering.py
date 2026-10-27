"""Row-wise derived features and missing-data summaries.

Derived indices are formulas of the same row. They are not statistics
learned from other rows, so they can be computed before the participant split.
Imputation and scaling stay in the training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GLUCOSE_FIELDS, INSULIN_FIELDS
from src.indices import INDEX_COLUMNS, calculate_indices
from src.units import convert_glucose_value, convert_insulin_value, is_missing

DERIVED_SPECS = [
    {"key": "homa_ir", "label": "HOMA1-IR"},
    {"key": "homa_b", "label": "HOMA1-B"},
    {"key": "insulinogenic_index", "label": "Insulinogenic index"},
    {"key": "matsuda_index", "label": "Matsuda index"},
    {"key": "glucose_auc", "label": "Glucose area under the curve"},
    {"key": "insulin_auc", "label": "Insulin area under the curve"},
    {"key": "oral_disposition_index", "label": "Oral disposition index"},
    {"key": "bmi_from_height_weight", "label": "BMI from weight and height"},
]


def missing_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        missing = int(frame[column].isna().sum())
        rows.append(
            {
                "column": column,
                "missing": missing,
                "missing_percent": round(missing / len(frame) * 100, 2) if len(frame) else 0.0,
                "rows": len(frame),
            }
        )
    return pd.DataFrame(rows)


def duplicate_summary(frame: pd.DataFrame) -> dict:
    exact = int(frame.duplicated().sum()) if len(frame) else 0
    key_count = 0
    if "participant_id" in frame.columns:
        keys = ["participant_id"] + (["observation_id"] if "observation_id" in frame.columns else [])
        key_count = int(frame.duplicated(keys).sum())
    return {"exact_duplicate_rows": exact, "duplicate_key_rows": key_count}


def add_derived_features(frame: pd.DataFrame, units: dict, requested: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add requested derived columns. Existing columns are left unchanged."""
    out = frame.copy()
    messages: list[str] = []
    index_keys = [key for key in requested if key in INDEX_COLUMNS]
    if index_keys:
        calculated, _summary = calculate_indices(out, units)
        for key in index_keys:
            if key in out.columns:
                messages.append(f"{key} already exists and was not overwritten.")
                continue
            out[key] = calculated[key].to_numpy()
            failed = int(calculated[key].isna().sum())
            if failed:
                messages.append(f"{key} could not be calculated for {failed} rows. Those cells were left missing.")
    if "bmi_from_height_weight" in requested:
        if "weight_kg" in out.columns and "height_cm" in out.columns:
            height_m = pd.to_numeric(out["height_cm"], errors="coerce") / 100.0
            weight = pd.to_numeric(out["weight_kg"], errors="coerce")
            bmi = weight / height_m.pow(2)
            bmi = bmi.where(height_m > 0)
            out["bmi_from_height_weight"] = bmi
            messages.append("Derived BMI uses weight_kg / (height_cm / 100)² on each row. It was not learned from other rows.")
        else:
            messages.append("BMI was not derived because weight and height are not both mapped.")
    return out, messages


def add_normalized_units(frame: pd.DataFrame, units: dict) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Add mmol/L and µU/mL copies of mapped glucose and insulin columns."""
    out = frame.copy()
    column_units: dict[str, str] = {}
    messages: list[str] = []
    glucose_unit = units.get("glucose")
    insulin_unit = units.get("insulin")
    for column in frame.columns:
        if column in GLUCOSE_FIELDS and glucose_unit:
            name = f"{column}__mmol_L"
            out[name] = [_maybe_glucose(value, glucose_unit) for value in frame[column]]
            column_units[name] = "mmol/L"
            column_units[column] = glucose_unit
            messages.append(f"Added {name} by converting {column} from {glucose_unit} to mmol/L.")
        if column in INSULIN_FIELDS and insulin_unit:
            name = f"{column}__uU_mL"
            out[name] = [_maybe_insulin(value, insulin_unit) for value in frame[column]]
            column_units[name] = "µU/mL"
            column_units[column] = insulin_unit
            messages.append(f"Added {name} by converting {column} from {insulin_unit} to µU/mL.")
    if not messages:
        messages.append("No mapped glucose or insulin columns were available to normalize.")
    return out, column_units, messages


def _maybe_glucose(value, unit: str):
    if is_missing(value):
        return np.nan
    return convert_glucose_value(value, unit, "mmol/L")


def _maybe_insulin(value, unit: str):
    if is_missing(value):
        return np.nan
    return convert_insulin_value(value, unit, "µU/mL")


def column_units_for_features(features: list[str], units: dict) -> dict[str, str]:
    stored = {}
    for feature in features:
        if feature in GLUCOSE_FIELDS or feature.endswith("__mmol_L"):
            stored[feature] = "mmol/L" if feature.endswith("__mmol_L") else units.get("glucose")
        elif feature in INSULIN_FIELDS or feature.endswith("__uU_mL"):
            stored[feature] = "µU/mL" if feature.endswith("__uU_mL") else units.get("insulin")
    return {key: value for key, value in stored.items() if value}
