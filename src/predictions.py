"""Score new rows with a saved pipeline. The pipeline is not refitted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.errors import PredictionError, UnitError
from src.units import canonicalize_glucose_unit, canonicalize_insulin_unit, convert_glucose_value, convert_insulin_value, is_missing
from src.validation import Issue, ValidationReport


def validate_prediction_inputs(
    frame: pd.DataFrame,
    artifact: dict,
    *,
    units_confirmed: bool,
    provided_units: dict | None = None,
) -> ValidationReport:
    issues: list[Issue] = []
    features = list(artifact.get("features") or [])
    if frame is None or len(frame) == 0:
        issues.append(Issue("error", "EMPTY_DATASET", "Add at least one row before scoring."))
        return ValidationReport(issues=issues)
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        issues.append(Issue("error", "MISSING_COLUMN", "Missing required columns: " + ", ".join(missing), count=len(missing)))
    extra = [column for column in frame.columns if column not in features]
    if extra:
        issues.append(Issue("warning", "MISSING_COLUMN", "Extra columns will be ignored: " + ", ".join(extra[:12]), count=len(extra)))
    if not units_confirmed:
        issues.append(
            Issue(
                "error",
                "UNSUPPORTED_UNIT",
                "Confirm that the new values use the same units as the saved model. Nothing was converted automatically.",
            )
        )
    expected = artifact.get("units") or {}
    provided = provided_units or {}
    for column, expected_unit in expected.items():
        given = provided.get(column)
        if given and not _same_unit(expected_unit, given):
            issues.append(
                Issue(
                    "error",
                    "UNSUPPORTED_UNIT",
                    f"{column} was trained with {expected_unit}, but the new values were described as {given}. "
                    "Use the explicit conversion step or provide matching units. Values were not changed.",
                    column,
                )
            )
    numeric_features, categorical_features = _feature_kinds(artifact)
    for feature in numeric_features:
        if feature not in frame.columns:
            continue
        parsed = pd.to_numeric(frame[feature], errors="coerce")
        failed = int((frame[feature].notna() & parsed.isna()).sum())
        if failed:
            issues.append(Issue("error", "NON_NUMERIC", f"{failed} values in {feature} are not numeric.", feature, failed))
    for feature in categorical_features:
        if feature not in frame.columns:
            continue
        # Unseen categories are accepted because one-hot encoding ignores them.
        pass
    if features and all(feature in frame.columns for feature in features):
        empty_rows = int(frame.loc[:, features].isna().all(axis=1).sum())
        if empty_rows:
            issues.append(
                Issue(
                    "warning",
                    "MISSING_VALUES",
                    f"{empty_rows} rows are empty for every model input. The saved imputer would fill them with training statistics.",
                    count=empty_rows,
                )
            )
    return ValidationReport(issues=issues)


def _unit_family(unit: str) -> str | None:
    try:
        canonicalize_glucose_unit(unit)
        return "glucose"
    except UnitError:
        pass
    try:
        canonicalize_insulin_unit(unit)
        return "insulin"
    except UnitError:
        return None


def _same_unit(expected: str, given: str) -> bool:
    expected_family = _unit_family(expected)
    given_family = _unit_family(given)
    if expected_family is None or expected_family != given_family:
        return False
    if expected_family == "glucose":
        return canonicalize_glucose_unit(expected) == canonicalize_glucose_unit(given)
    return canonicalize_insulin_unit(expected) == canonicalize_insulin_unit(given)


def _feature_kinds(artifact: dict) -> tuple[list[str], list[str]]:
    preprocessing = artifact.get("preprocessing") or {}
    return list(preprocessing.get("numeric_features") or []), list(preprocessing.get("categorical_features") or [])


def convert_inputs_to_model_units(frame: pd.DataFrame, artifact: dict, provided_units: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Convert named columns only when the caller supplies the current unit."""
    out = frame.copy()
    notes = []
    expected = artifact.get("units") or {}
    for column, target_unit in expected.items():
        if column not in out.columns or column not in provided_units:
            continue
        source_unit = provided_units[column]
        if _same_unit(target_unit, source_unit):
            continue
        converted = []
        for value in out[column]:
            if is_missing(value):
                converted.append(np.nan)
                continue
            if _unit_family(target_unit) == "glucose":
                converted.append(convert_glucose_value(value, source_unit, target_unit))
            else:
                converted.append(convert_insulin_value(value, source_unit, target_unit))
        out[column] = converted
        notes.append(f"Converted {column} from {source_unit} to {target_unit} because that conversion was requested.")
    if not notes:
        notes.append("No columns needed conversion.")
    return out, notes


def predict_frame(artifact: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved pipeline. Fitting is not called."""
    features = list(artifact["features"])
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise PredictionError("Missing required columns: " + ", ".join(missing))
    model_input = frame.loc[:, features].copy()
    numeric, _categorical = _feature_kinds(artifact)
    for feature in numeric:
        model_input[feature] = pd.to_numeric(model_input[feature], errors="coerce")
    pipeline = artifact["pipeline"]
    predicted = pipeline.predict(model_input)
    task = artifact["task"]
    if task == "classification":
        result = pd.DataFrame({"predicted_class": np.asarray(predicted).astype(str)})
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(model_input)
            classes = [str(label) for label in artifact.get("classes") or pipeline.named_steps["model"].classes_]
            for index, label in enumerate(classes):
                result[f"probability_{label}"] = probabilities[:, index]
    else:
        result = pd.DataFrame({"predicted_value": np.asarray(predicted, dtype=float)})
    result.insert(0, "row_number", np.arange(1, len(result) + 1))
    result.insert(1, "output_note", "Research prediction. Not a diagnosis, treatment recommendation, or insulin dose.")
    if artifact.get("dataset_type") == "synthetic":
        result["training_data"] = "synthetic_demonstration"
    return result
