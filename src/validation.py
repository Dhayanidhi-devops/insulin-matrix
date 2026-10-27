"""Dataset validation. Errors block indices and training. Warnings do not."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.config import (
    BOOKKEEPING_COLUMNS,
    CANONICAL_VARIABLES,
    EXCLUDED_FROM_FEATURES,
    GLUCOSE_FIELDS,
    GLUCOSE_TIME_COLUMNS,
    IDENTIFIER_FIELDS,
    INSULIN_FIELDS,
    INSULIN_TIME_COLUMNS,
    OGTT_TIMES,
)
from src.errors import UnitError
from src.units import canonicalize_glucose_unit, canonicalize_insulin_unit, glucose_to_mmol, insulin_to_uU, is_missing

_CANONICAL_BY_KEY = {item["key"]: item for item in CANONICAL_VARIABLES}
_ID_COLLAPSE = re.compile(r"^(\d+)\.0+$")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    column: str | None = None
    count: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stamp: str = ""
    layout: str = "unspecified"
    readiness: dict[str, str] = field(default_factory=dict)
    target_column: str | None = None

    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def status(self) -> str:
        if self.has_errors():
            return "failed"
        if self.warning_count:
            return "passed_with_warnings"
        return "passed"

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["severity", "code", "column", "count", "message"])
        return pd.DataFrame([issue.as_dict() for issue in self.issues])


def empty_mapping() -> dict[str, str | None]:
    mapping = {item["key"]: None for item in CANONICAL_VARIABLES}
    mapping["target"] = None
    return mapping


def mapping_stamp(mapping: dict, units: dict, extra: list[str], content_hash: str) -> str:
    payload = {
        "mapping": {key: mapping.get(key) for key in sorted(mapping)},
        "units": units,
        "extra": list(extra),
        "content_hash": content_hash,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _issue(issues: list[Issue], severity: str, code: str, message: str, column: str | None = None, count: int | None = None) -> None:
    issues.append(Issue(severity, code, message, column, count))


def detect_layout(mapping: dict) -> tuple[str, list[Issue]]:
    issues: list[Issue] = []
    mapped = {key for key, source in mapping.items() if source and key != "target"}
    wide_fields = set(GLUCOSE_TIME_COLUMNS.values()) | set(INSULIN_TIME_COLUMNS.values())
    has_wide = bool(mapped & wide_fields)
    has_long = "time_point" in mapped and bool(mapped & {"glucose", "insulin"})
    if has_wide and has_long:
        _issue(
            issues,
            "warning",
            "INADEQUATE_TIMEPOINTS",
            "Both wide OGTT columns and a long-format time column are mapped. "
            "Wide columns will be used. The long-format time point will not be combined with them.",
        )
        return "wide", issues
    if has_long:
        return "long", issues
    fasting_only = bool(mapped & {"fasting_glucose", "fasting_insulin"}) and not has_wide
    if fasting_only:
        return "fasting_only", issues
    if has_wide:
        return "wide", issues
    return "unspecified", issues


def _coerce_numeric(series: pd.Series, name: str, issues: list[Issue]) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        _issue(issues, "error", "NON_NUMERIC", f"{name} contains boolean values and cannot be used as a number.", name)
        return pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    original = series.notna()
    converted = pd.to_numeric(series, errors="coerce")
    failed = int((original & converted.isna()).sum())
    if failed:
        _issue(
            issues,
            "error",
            "NON_NUMERIC",
            f"{failed} values in {name} could not be parsed as numbers.",
            name,
            failed,
        )
    elif int(original.sum()):
        _issue(
            issues,
            "warning",
            "NON_NUMERIC",
            f"{name} was stored as text. Those values were parsed as numbers in the analysis copy. The uploaded file was not modified.",
            name,
            int(original.sum()),
        )
    return converted


def _as_identifier(series: pd.Series) -> pd.Series:
    def convert(value):
        if is_missing(value):
            return ""
        return str(value).strip()

    return series.map(convert)


def _collapse_identifier(value: str) -> str:
    text = value.strip()
    match = _ID_COLLAPSE.match(text)
    if match:
        text = match.group(1)
    return text.casefold()


def _first_valid(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return valid.iloc[0]


def build_analysis_frame(
    frame: pd.DataFrame,
    mapping: dict,
    extra_covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, list[Issue], str, str | None]:
    """Rename mapped columns and pivot long OGTT sheets to one row per visit.

    The uploaded frame is not modified. Text that can be parsed is converted
    only in the returned analysis copy.
    """
    issues: list[Issue] = []
    extras = list(extra_covariates or [])
    layout, layout_issues = detect_layout(mapping)
    issues.extend(layout_issues)
    renamed = pd.DataFrame(index=frame.index)
    source_to_keys: dict[str, list[str]] = {}

    for key, source in mapping.items():
        if key == "target" or not source:
            continue
        if source not in frame.columns:
            _issue(issues, "error", "MISSING_COLUMN", f"Mapped column {source!r} for {key} is not in the worksheet.", key)
            continue
        source_to_keys.setdefault(source, []).append(key)
        renamed[key] = frame[source]

    for source, keys in source_to_keys.items():
        if len(keys) > 1:
            _issue(
                issues,
                "warning",
                "DUPLICATE_KEYS",
                f"The worksheet column {source!r} is mapped to more than one variable ({', '.join(keys)}). "
                "Each variable received a copy. Confirm that this is intentional.",
                source,
            )

    target_source = mapping.get("target")
    target_column = None
    if target_source:
        if target_source not in frame.columns:
            _issue(issues, "error", "MISSING_COLUMN", f"The modelling target column {target_source!r} is not in the worksheet.", "target")
        elif target_source in source_to_keys:
            target_column = source_to_keys[target_source][0]
        else:
            target_column = str(target_source)
            renamed[target_column] = frame[target_source]

    for column in extras:
        if column not in frame.columns:
            _issue(issues, "error", "MISSING_COLUMN", f"Selected covariate {column!r} is not in the worksheet.", column)
            continue
        if column in renamed.columns:
            _issue(issues, "warning", "DUPLICATE_KEYS", f"Covariate {column} was not added again because that name is already in the analysis table.", column)
            continue
        renamed[column] = frame[column]

    for column in BOOKKEEPING_COLUMNS:
        if column in frame.columns and column not in renamed.columns:
            renamed[column] = frame[column]

    # Keep unmapped worksheet columns so a modelling target can be chosen
    # without a second mapping step. Mapped sources are not copied twice.
    copied_sources = set(source_to_keys)
    for column in frame.columns:
        if column in copied_sources or column in renamed.columns:
            continue
        renamed[column] = frame[column]

    if renamed.empty and frame.empty:
        _issue(issues, "error", "EMPTY_DATASET", "The worksheet has no rows to validate.")
        return renamed, issues, layout, target_column

    for column in list(renamed.columns):
        spec = _CANONICAL_BY_KEY.get(column)
        if spec and spec["kind"] in {"glucose", "insulin", "numeric", "time"}:
            renamed[column] = _coerce_numeric(renamed[column], column, issues)
        elif spec and spec["kind"] == "id":
            renamed[column] = _as_identifier(renamed[column])
        elif column == "time_point":
            renamed[column] = _coerce_numeric(renamed[column], column, issues)

    if any(issue.severity == "error" and issue.code == "MISSING_COLUMN" for issue in issues):
        return renamed, issues, layout, target_column

    if layout == "long":
        renamed, long_issues = _pivot_long(renamed)
        issues.extend(long_issues)

    return renamed.reset_index(drop=True), issues, layout, target_column


def _pivot_long(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[Issue]]:
    issues: list[Issue] = []
    if "participant_id" not in frame.columns:
        _issue(issues, "error", "MISSING_COLUMN", "Long-format sheets need a participant ID before time points can be reshaped.", "participant_id")
        return frame, issues
    work = frame.copy()
    unknown = work["time_point"].notna() & ~work["time_point"].isin(OGTT_TIMES)
    if int(unknown.sum()):
        _issue(
            issues,
            "warning",
            "INADEQUATE_TIMEPOINTS",
            f"{int(unknown.sum())} rows use time points other than 0, 30, 60, 90, and 120 minutes. Those rows are omitted from index calculations.",
            "time_point",
            int(unknown.sum()),
        )
    work = work.loc[~unknown].copy()
    if work.empty:
        _issue(issues, "error", "INADEQUATE_TIMEPOINTS", "No supported OGTT time points remain after removing unrecognized times.", "time_point")
        return frame, issues

    group_cols = ["participant_id"] + (["observation_id"] if "observation_id" in work.columns else [])
    duplicate_mask = work.duplicated(group_cols + ["time_point"], keep=False)
    if int(duplicate_mask.sum()):
        _issue(
            issues,
            "error",
            "DUPLICATE_KEYS",
            "The same participant, observation, and time point occurs more than once. "
            "Rows were not deleted. Resolve the duplicates before calculating indices.",
            "time_point",
            int(duplicate_mask.sum()),
        )
        return frame, issues

    pieces = []
    rename_maps = (("glucose", GLUCOSE_TIME_COLUMNS), ("insulin", INSULIN_TIME_COLUMNS))
    for value_column, rename_map in rename_maps:
        if value_column not in work.columns:
            continue
        pivoted = work.pivot(index=group_cols, columns="time_point", values=value_column)
        renamed_columns = {}
        for column in pivoted.columns:
            minute = int(column) if float(column).is_integer() else column
            if minute in rename_map:
                renamed_columns[column] = rename_map[minute]
        pivoted = pivoted.rename(columns=renamed_columns)
        pieces.append(pivoted)

    if not pieces:
        _issue(issues, "error", "MISSING_COLUMN", "Long format needs a glucose column, an insulin column, or both.", "glucose")
        return frame, issues

    wide = pd.concat(pieces, axis=1)
    carry = [col for col in work.columns if col not in set(group_cols) | {"time_point", "glucose", "insulin"}]
    if carry:
        grouped = work.groupby(group_cols, sort=False)
        for column in carry:
            nunique = grouped[column].nunique(dropna=True)
            if bool((nunique > 1).any()):
                _issue(
                    issues,
                    "warning",
                    "INCONSISTENT_IDS",
                    f"{column} is not constant within a participant observation. The first non-missing value was kept for that observation.",
                    column,
                    int((nunique > 1).sum()),
                )
        covariate_frame = grouped[carry].agg(_first_valid)
        wide = wide.join(covariate_frame)
    return wide.reset_index(), issues


def index_readiness(frame: pd.DataFrame) -> dict[str, str]:
    columns = set(frame.columns)

    def has(name: str) -> bool:
        return name in columns

    readiness: dict[str, str] = {}
    if has("fasting_glucose") and has("fasting_insulin"):
        readiness["homa_ir"] = "ready"
        readiness["homa_b"] = "ready"
    else:
        message = "Fasting glucose and fasting insulin are not both available."
        readiness["homa_ir"] = message
        readiness["homa_b"] = message
    if all(has(name) for name in ("fasting_glucose", "glucose_30", "fasting_insulin", "insulin_30")):
        readiness["insulinogenic_index"] = "ready"
        readiness["oral_disposition_index"] = "ready when glucose rises and fasting insulin is positive"
    else:
        message = "Fasting and 30-minute glucose and insulin are not all available."
        readiness["insulinogenic_index"] = message
        readiness["oral_disposition_index"] = message
    paired_post = any(
        has(GLUCOSE_TIME_COLUMNS[minute]) and has(INSULIN_TIME_COLUMNS[minute])
        for minute in OGTT_TIMES
        if minute != 0
    )
    if has("fasting_glucose") and has("fasting_insulin") and paired_post:
        readiness["matsuda_index"] = "ready"
    else:
        readiness["matsuda_index"] = "Matsuda index needs paired fasting values and at least one paired post-load sample."
    glucose_times = [minute for minute, column in GLUCOSE_TIME_COLUMNS.items() if has(column)]
    insulin_times = [minute for minute, column in INSULIN_TIME_COLUMNS.items() if has(column)]
    readiness["glucose_auc"] = "ready" if len(glucose_times) >= 2 else "At least two glucose time points are required."
    readiness["insulin_auc"] = "ready" if len(insulin_times) >= 2 else "At least two insulin time points are required."
    return readiness


def _check_units(mapping: dict, units: dict, issues: list[Issue]) -> dict[str, str]:
    normalized = {"glucose": None, "insulin": None}
    mapped = {key for key, source in mapping.items() if source}
    needs_glucose = bool(mapped & GLUCOSE_FIELDS)
    needs_insulin = bool(mapped & INSULIN_FIELDS)
    if needs_glucose:
        try:
            normalized["glucose"] = canonicalize_glucose_unit(units.get("glucose") if units else None)
        except UnitError as exc:
            _issue(issues, "error", "UNSUPPORTED_UNIT", str(exc), "glucose")
    if needs_insulin:
        try:
            normalized["insulin"] = canonicalize_insulin_unit(units.get("insulin") if units else None)
        except UnitError as exc:
            _issue(issues, "error", "UNSUPPORTED_UNIT", str(exc), "insulin")
    return normalized


def _check_identifiers(frame: pd.DataFrame, issues: list[Issue]) -> None:
    if "participant_id" not in frame.columns:
        _issue(
            issues,
            "warning",
            "MISSING_COLUMN",
            "No participant ID is mapped. Model evaluation cannot keep repeated measurements together unless you confirm that each row is an independent participant.",
            "participant_id",
        )
        return
    blank = frame["participant_id"].astype(str).str.strip().eq("")
    if int(blank.sum()):
        _issue(issues, "error", "INCONSISTENT_IDS", "Participant ID is blank for one or more rows.", "participant_id", int(blank.sum()))
    collapsed: dict[str, set[str]] = {}
    for value in frame["participant_id"].astype(str):
        if not value.strip():
            continue
        collapsed.setdefault(_collapse_identifier(value), set()).add(value)
    collisions = {key: values for key, values in collapsed.items() if len(values) > 1}
    if collisions:
        examples = [", ".join(sorted(values)) for values in list(collisions.values())[:5]]
        _issue(
            issues,
            "warning",
            "INCONSISTENT_IDS",
            "Some participant IDs differ only by case or by a trailing .0 (" + "; ".join(examples) + "). They were not merged.",
            "participant_id",
            len(collisions),
        )


def _check_duplicates(frame: pd.DataFrame, issues: list[Issue]) -> None:
    if len(frame) and int(frame.duplicated().sum()):
        _issue(
            issues,
            "warning",
            "DUPLICATE_ROWS",
            f"{int(frame.duplicated().sum())} exact duplicate rows are present. They were not deleted.",
            count=int(frame.duplicated().sum()),
        )
    if "participant_id" not in frame.columns:
        return
    key = ["participant_id"] + (["observation_id"] if "observation_id" in frame.columns else [])
    if "observation_id" in frame.columns:
        duplicate_keys = frame.duplicated(key, keep=False)
        count = int(duplicate_keys.sum())
        if count:
            _issue(
                issues,
                "error",
                "DUPLICATE_KEYS",
                "The same participant and observation ID occurs on more than one row. Rows were not deleted.",
                "observation_id",
                count,
            )
    else:
        repeated = frame["participant_id"].duplicated(keep=False)
        count = int(repeated.sum())
        if count:
            _issue(
                issues,
                "warning",
                "DUPLICATE_KEYS",
                "Participant IDs repeat and no observation ID is mapped. Rows were kept. "
                "If these are repeated measurements, map an observation ID. "
                "Model evaluation will still keep all rows for a participant in one partition.",
                "participant_id",
                count,
            )


def _range_check(frame: pd.DataFrame, units: dict, issues: list[Issue]) -> None:
    glucose_unit = units.get("glucose")
    insulin_unit = units.get("insulin")
    for column in frame.columns:
        spec = _CANONICAL_BY_KEY.get(column)
        if spec is None or spec["kind"] not in {"glucose", "insulin", "numeric"}:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if spec["kind"] == "glucose" and glucose_unit:
            _glucose_bounds(series, column, glucose_unit, issues)
        elif spec["kind"] == "insulin" and insulin_unit:
            _insulin_bounds(series, column, insulin_unit, issues)
        elif column == "age":
            _simple_bounds(series, column, low=0, high=120, label="Age", issues=issues, warn_low=1, warn_high=100)
        elif column == "bmi":
            _simple_bounds(series, column, low=0, high=90, label="BMI", issues=issues, strict_low=True)
        elif column == "hba1c":
            invalid = series.notna() & (series <= 0)
            if int(invalid.sum()):
                _issue(issues, "error", "IMPOSSIBLE_VALUE", "HbA1c contains non-positive values.", column, int(invalid.sum()))


def _glucose_bounds(series: pd.Series, column: str, unit: str, issues: list[Issue]) -> None:
    mmol = []
    for value in series:
        if is_missing(value):
            mmol.append(math.nan)
            continue
        try:
            mmol.append(glucose_to_mmol(value, unit))
        except (UnitError, ValueError):
            mmol.append(math.nan)
    values = pd.Series(mmol, index=series.index)
    invalid = values.notna() & ((values <= 0) | (values > 55))
    if int(invalid.sum()):
        _issue(
            issues,
            "error",
            "IMPOSSIBLE_VALUE",
            f"{column} has values that are non-positive or above 55 mmol/L (about 990 mg/dL).",
            column,
            int(invalid.sum()),
        )


def _insulin_bounds(series: pd.Series, column: str, unit: str, issues: list[Issue]) -> None:
    converted = []
    for value in series:
        if is_missing(value):
            converted.append(math.nan)
            continue
        try:
            converted.append(insulin_to_uU(value, unit))
        except (UnitError, ValueError):
            converted.append(math.nan)
    values = pd.Series(converted, index=series.index)
    invalid = values.notna() & ((values < 0) | (values > 5000))
    zeros = values.notna() & values.eq(0)
    if int(invalid.sum()):
        _issue(
            issues,
            "error",
            "IMPOSSIBLE_VALUE",
            f"{column} has negative insulin values or values above 5000 µU/mL.",
            column,
            int(invalid.sum()),
        )
    if int(zeros.sum()):
        _issue(
            issues,
            "warning",
            "IMPOSSIBLE_VALUE",
            f"{column} contains zeros. Matsuda index and the oral disposition index are undefined when insulin is zero.",
            column,
            int(zeros.sum()),
        )


def _simple_bounds(series, column, low, high, label, issues, warn_low=None, warn_high=None, strict_low=False) -> None:
    present = series.notna()
    invalid = present & ((series < low) | (series > high) | (strict_low & series.eq(0)))
    if int(invalid.sum()):
        _issue(issues, "error", "IMPOSSIBLE_VALUE", f"{label} has values outside {low} to {high}.", column, int(invalid.sum()))
    if warn_low is not None:
        unusual = present & ~invalid & ((series < warn_low) | (series > warn_high))
        if int(unusual.sum()):
            _issue(issues, "warning", "IMPOSSIBLE_VALUE", f"{label} has unusual values that are still inside the allowed range.", column, int(unusual.sum()))


def _check_missing_and_times(frame: pd.DataFrame, mapping: dict, issues: list[Issue]) -> None:
    for column in frame.columns:
        if column in BOOKKEEPING_COLUMNS:
            continue
        missing = int(pd.isna(frame[column]).sum()) if column in frame else 0
        if missing and column not in IDENTIFIER_FIELDS:
            _issue(issues, "warning", "MISSING_VALUES", f"{column} is missing for {missing} rows.", column, missing)
    mapped = {key for key, source in mapping.items() if source}
    if "glucose_30" in mapped and "fasting_glucose" not in mapped and "glucose" not in mapped:
        _issue(issues, "warning", "INADEQUATE_TIMEPOINTS", "30-minute glucose is mapped without fasting glucose. The insulinogenic index cannot be calculated.", "glucose_30")
    if "insulin_30" in mapped and "fasting_insulin" not in mapped and "insulin" not in mapped:
        _issue(issues, "warning", "INADEQUATE_TIMEPOINTS", "30-minute insulin is mapped without fasting insulin.", "insulin_30")
    readiness = index_readiness(frame)
    for name, state in readiness.items():
        if state != "ready" and not state.startswith("ready when"):
            _issue(issues, "warning", "INADEQUATE_TIMEPOINTS", f"{name.replace('_', ' ')}: {state}", name)
    for family, columns in (("Glucose", GLUCOSE_TIME_COLUMNS), ("Insulin", INSULIN_TIME_COLUMNS)):
        present = [minute for minute, column in columns.items() if column in frame.columns]
        if present and set(present) != set(OGTT_TIMES):
            missing = [minute for minute in OGTT_TIMES if minute not in present]
            _issue(
                issues,
                "warning",
                "INADEQUATE_TIMEPOINTS",
                f"{family} samples are available at minutes {', '.join(str(minute) for minute in present)}. "
                f"Missing {', '.join(str(minute) for minute in missing)}. "
                "Area and Matsuda calculations use only the available samples and are not a full five-point OGTT.",
                family.lower(),
            )


def _check_target(frame: pd.DataFrame, target_column: str | None, issues: list[Issue]) -> None:
    if not target_column:
        _issue(issues, "warning", "MISSING_COLUMN", "No modelling target is mapped. You can still calculate indices.", "target")
        return
    if target_column in EXCLUDED_FROM_FEATURES:
        _issue(issues, "error", "TARGET_LEAKAGE", f"{target_column} is an identifier or bookkeeping column and cannot be a modelling target.", target_column)
        return
    if target_column not in frame.columns:
        _issue(issues, "error", "MISSING_COLUMN", f"The target {target_column} is not on the analysis table.", target_column)
        return
    missing = int(pd.isna(frame[target_column]).sum())
    if missing:
        _issue(issues, "warning", "MISSING_VALUES", f"The target is missing for {missing} rows. Those rows are excluded only when a model is trained, and the count is shown first.", target_column, missing)


def validate_dataset(
    original: pd.DataFrame,
    mapping: dict,
    units: dict | None,
    extra_covariates: list[str] | None = None,
    content_hash: str = "",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a mapped worksheet and return the analysis copy plus the report."""
    issues: list[Issue] = []
    if original is None or original.shape[1] == 0 or len(original) == 0:
        _issue(issues, "error", "EMPTY_DATASET", "The worksheet is empty. Choose a worksheet with a header row and data.")
        report = ValidationReport(issues=issues, stamp="", layout="unspecified", readiness={}, target_column=None)
        return pd.DataFrame(), report

    if not any(source for key, source in mapping.items() if key != "target") and not mapping.get("target"):
        _issue(issues, "error", "MISSING_COLUMN", "Map at least one spreadsheet column before validation.")

    normalized_units = _check_units(mapping, units or {}, issues)
    analysis, build_issues, layout, target_column = build_analysis_frame(original, mapping, extra_covariates)
    issues.extend(build_issues)
    if analysis.shape[0] == 0 and not any(issue.code == "EMPTY_DATASET" for issue in issues):
        _issue(issues, "error", "EMPTY_DATASET", "The analysis table has no rows.")
    else:
        _check_identifiers(analysis, issues)
        _check_duplicates(analysis, issues)
        _range_check(analysis, normalized_units, issues)
        _check_missing_and_times(analysis, mapping, issues)
        _check_target(analysis, target_column, issues)

    readiness = index_readiness(analysis) if len(analysis) else {}
    stamp = mapping_stamp(mapping, normalized_units, list(extra_covariates or []), content_hash)
    report = ValidationReport(
        issues=issues,
        stamp=stamp,
        layout=layout,
        readiness=readiness,
        target_column=target_column,
    )
    return analysis, report


def leakage_issues(features: list[str], target: str | None) -> list[Issue]:
    """Flag target leakage and identifier inputs. Does not change the feature list."""
    issues: list[Issue] = []
    if target and target in features:
        _issue(issues, "error", "TARGET_LEAKAGE", f"The target '{target}' is also selected as a model input.", target)
    for feature in features:
        if feature in EXCLUDED_FROM_FEATURES:
            _issue(
                issues,
                "error",
                "TARGET_LEAKAGE",
                f"'{feature}' is an identifier or bookkeeping column and is excluded from model inputs.",
                feature,
            )
    index_inputs = {
        "homa_ir": ["fasting_glucose", "fasting_insulin"],
        "homa_b": ["fasting_glucose", "fasting_insulin"],
        "insulinogenic_index": ["fasting_glucose", "glucose_30", "fasting_insulin", "insulin_30"],
        "oral_disposition_index": ["fasting_glucose", "glucose_30", "fasting_insulin", "insulin_30"],
        "matsuda_index": list(GLUCOSE_TIME_COLUMNS.values()) + list(INSULIN_TIME_COLUMNS.values()),
        "glucose_auc": list(GLUCOSE_TIME_COLUMNS.values()),
        "insulin_auc": list(INSULIN_TIME_COLUMNS.values()),
    }
    if target in index_inputs:
        overlap = [column for column in index_inputs[target] if column in features]
        if overlap:
            _issue(
                issues,
                "warning",
                "POTENTIAL_LEAKAGE",
                f"The target {target} is calculated from {', '.join(overlap)}. Models that use those inputs can reconstruct the target.",
                target,
            )
    return issues
