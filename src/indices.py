"""Physiological index calculations.

Every index below is a calculated research quantity, not a laboratory
measurement and not a diagnosis. HOMA values are HOMA1 only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.config import (
    GLUCOSE_TIME_COLUMNS,
    HOMA2_STATEMENT,
    HOMA_B_OFFSET_MMOL,
    HOMA_IR_DENOMINATOR,
    INSULIN_TIME_COLUMNS,
    MATSUDA_CONSTANT,
    OGTT_TIMES,
)
from src.units import (
    glucose_to_mgdl,
    glucose_to_mmol,
    insulin_to_uU,
    is_missing,
)

FORMULAS = {
    "homa_ir": (
        "HOMA1-IR = fasting glucose × fasting insulin / 22.5, "
        "with glucose in mmol/L and insulin in µU/mL. "
        "The mg/dL form is fasting glucose × fasting insulin / 405, "
        "with insulin still in µU/mL. Insulin Matrix uses a factor of "
        "18 mg/dL per mmol/L so the two forms match."
    ),
    "homa_b": (
        "HOMA1-B = 20 × fasting insulin / (fasting glucose − 3.5), "
        "with glucose in mmol/L and insulin in µU/mL. "
        "The mg/dL form is 360 × fasting insulin / (fasting glucose − 63). "
        "The index is undefined when fasting glucose is at or below "
        "3.5 mmol/L (63 mg/dL)."
    ),
    "insulinogenic_index": (
        "Insulinogenic index = (insulin at 30 min − fasting insulin) / "
        "(glucose at 30 min − fasting glucose). "
        "Reported values use insulin in µU/mL and glucose in mmol/L "
        "after conversion. The index is not calculated when glucose "
        "does not increase."
    ),
    "matsuda_index": (
        "Matsuda index = 10000 / sqrt(fasting glucose × fasting insulin "
        "× mean glucose × mean insulin). Glucose is converted to mg/dL "
        "and insulin to µU/mL before the constant is applied. "
        "The mean is the arithmetic mean of paired samples at the "
        "available times among 0, 30, 60, 90, and 120 minutes."
    ),
    "glucose_auc": (
        "Glucose area = Σ (tᵢ₊₁ − tᵢ) × (glucoseᵢ + glucoseᵢ₊₁) / 2, "
        "with time in minutes. The result uses the glucose unit supplied "
        "for the dataset and is a linear trapezoid over the available samples."
    ),
    "insulin_auc": (
        "Insulin area = Σ (tᵢ₊₁ − tᵢ) × (insulinᵢ + insulinᵢ₊₁) / 2, "
        "with time in minutes. The result uses the insulin unit supplied "
        "for the dataset."
    ),
    "oral_disposition_index": (
        "Oral disposition index = insulinogenic index / fasting insulin, "
        "with the insulinogenic index in (µU/mL) / (mmol/L) and fasting "
        "insulin in µU/mL. This is the oral disposition index described "
        "by Utzschneider and colleagues. It is not the product of the "
        "insulinogenic index and the Matsuda index, and it is not HOMA2."
    ),
}

INDEX_COLUMNS = (
    "homa_ir",
    "homa_b",
    "insulinogenic_index",
    "matsuda_index",
    "glucose_auc",
    "insulin_auc",
    "oral_disposition_index",
)

ASSUMPTIONS = {
    "homa_ir": "Fasting steady state. HOMA1, not HOMA2. Not a diagnosis of insulin resistance.",
    "homa_b": "Fasting steady state. Undefined at or below 3.5 mmol/L glucose. Not a beta-cell diagnosis.",
    "insulinogenic_index": "Requires a rise in glucose from 0 to 30 minutes. Sampling times are the named columns.",
    "matsuda_index": "Uses paired arithmetic means, not a continuous curve. Missing OGTT points change the result.",
    "glucose_auc": "Linear trapezoid on the samples present. Omitted samples are not imputed.",
    "insulin_auc": "Linear trapezoid on the samples present. Omitted samples are not imputed.",
    "oral_disposition_index": "Requires a defined insulinogenic index and positive fasting insulin.",
}


@dataclass(frozen=True)
class IndexValue:
    value: float | None
    status: str
    message: str | None = None

    @property
    def calculated(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


def _fail(message: str) -> IndexValue:
    return IndexValue(None, "not_calculated", message)


def _convert_glucose(value, unit: str) -> float | None:
    if is_missing(value):
        return None
    return glucose_to_mmol(value, unit)


def _convert_insulin(value, unit: str) -> float | None:
    if is_missing(value):
        return None
    return insulin_to_uU(value, unit)


def homa_ir(fasting_glucose, fasting_insulin, glucose_unit: str, insulin_unit: str) -> IndexValue:
    """HOMA1-IR. Returns a value of 0 when insulin is 0 and glucose is positive."""
    glucose = _convert_glucose(fasting_glucose, glucose_unit)
    insulin = _convert_insulin(fasting_insulin, insulin_unit)
    if glucose is None or insulin is None:
        return _fail("HOMA-IR needs fasting glucose and fasting insulin.")
    if glucose <= 0 or insulin < 0:
        return _fail("HOMA-IR is not calculated for non-positive glucose or negative insulin.")
    return IndexValue(glucose * insulin / HOMA_IR_DENOMINATOR, "ok", None)


def homa_b(fasting_glucose, fasting_insulin, glucose_unit: str, insulin_unit: str) -> IndexValue:
    """HOMA1-B. Undefined when glucose is at or below 3.5 mmol/L."""
    glucose = _convert_glucose(fasting_glucose, glucose_unit)
    insulin = _convert_insulin(fasting_insulin, insulin_unit)
    if glucose is None or insulin is None:
        return _fail("HOMA-B needs fasting glucose and fasting insulin.")
    if glucose <= 0 or insulin < 0:
        return _fail("HOMA-B is not calculated for non-positive glucose or negative insulin.")
    denominator = glucose - HOMA_B_OFFSET_MMOL
    if denominator <= 0:
        return _fail(
            "HOMA-B denominator is non-positive when fasting glucose is at or below 3.5 mmol/L."
        )
    return IndexValue((20.0 * insulin) / denominator, "ok", None)


def insulinogenic_index(
    fasting_glucose,
    glucose_30,
    fasting_insulin,
    insulin_30,
    glucose_unit: str,
    insulin_unit: str,
) -> IndexValue:
    """30-minute insulinogenic index in (µU/mL) per (mmol/L)."""
    g0 = _convert_glucose(fasting_glucose, glucose_unit)
    g30 = _convert_glucose(glucose_30, glucose_unit)
    i0 = _convert_insulin(fasting_insulin, insulin_unit)
    i30 = _convert_insulin(insulin_30, insulin_unit)
    if None in (g0, g30, i0, i30):
        return _fail("Insulinogenic index needs glucose and insulin at 0 and 30 minutes.")
    assert g0 is not None and g30 is not None and i0 is not None and i30 is not None
    if min(g0, g30) <= 0 or min(i0, i30) < 0:
        return _fail("Insulinogenic index is not calculated for invalid glucose or insulin values.")
    delta_g = g30 - g0
    if delta_g <= 0:
        return _fail(
            "Insulinogenic index is undefined when glucose does not increase from 0 to 30 minutes."
        )
    return IndexValue((i30 - i0) / delta_g, "ok", None)


def trapezoidal_auc(times, values) -> IndexValue:
    """Linear trapezoidal area. Time must be strictly increasing and in minutes."""
    points: list[tuple[float, float]] = []
    for time, value in zip(times, values):
        if is_missing(time) or is_missing(value):
            continue
        t_num = float(time)
        v_num = float(value)
        if not math.isfinite(t_num) or not math.isfinite(v_num):
            continue
        points.append((t_num, v_num))
    if len(points) < 2:
        return _fail("At least two time points are required for area under the curve.")
    ordered_times = [item[0] for item in points]
    if any(later <= earlier for earlier, later in zip(ordered_times, ordered_times[1:])):
        return _fail("Area under the curve requires strictly increasing time points.")
    if any(value < 0 for _, value in points):
        return _fail("Area under the curve is not calculated when a measurement is negative.")
    area = 0.0
    for (t0, y0), (t1, y1) in zip(points, points[1:]):
        area += (t1 - t0) * (y0 + y1) / 2.0
    used = {int(t) if float(t).is_integer() else t for t in ordered_times}
    expected = set(OGTT_TIMES)
    if used != expected:
        missing = sorted(expected - used)
        return IndexValue(
            area,
            "warning",
            "Area uses minutes "
            + ", ".join(str(t) for t in sorted(used))
            + ". Missing "
            + ", ".join(str(t) for t in missing)
            + ". This is not a full five-point OGTT area.",
        )
    return IndexValue(area, "ok", None)


def matsuda_index(
    glucose_by_time: dict[int, float | None],
    insulin_by_time: dict[int, float | None],
    glucose_unit: str,
    insulin_unit: str,
) -> IndexValue:
    """Matsuda insulin-sensitivity index on paired OGTT samples."""
    paired: list[tuple[int, float, float]] = []
    for minute in OGTT_TIMES:
        raw_g = glucose_by_time.get(minute)
        raw_i = insulin_by_time.get(minute)
        if is_missing(raw_g) or is_missing(raw_i):
            continue
        glucose = glucose_to_mgdl(raw_g, glucose_unit)
        insulin = insulin_to_uU(raw_i, insulin_unit)
        paired.append((minute, glucose, insulin))
    times = {item[0] for item in paired}
    if 0 not in times:
        return _fail("Matsuda index needs paired fasting glucose and fasting insulin.")
    if len(paired) < 2:
        return _fail("Matsuda index needs fasting values and at least one post-load pair.")
    if any(glucose <= 0 or insulin <= 0 for _, glucose, insulin in paired):
        return _fail("Matsuda index is undefined when paired glucose or insulin is zero or negative.")
    fasting = next(item for item in paired if item[0] == 0)
    mean_g = sum(item[1] for item in paired) / len(paired)
    mean_i = sum(item[2] for item in paired) / len(paired)
    value = MATSUDA_CONSTANT / math.sqrt(fasting[1] * fasting[2] * mean_g * mean_i)
    if times != set(OGTT_TIMES):
        return IndexValue(
            value,
            "warning",
            "Matsuda index uses paired minutes "
            + ", ".join(str(t) for t in sorted(times))
            + ". Missing samples change the mean.",
        )
    return IndexValue(value, "ok", None)


def oral_disposition_index(igi: IndexValue, fasting_insulin, insulin_unit: str) -> IndexValue:
    """Utzschneider oral disposition index: insulinogenic index / fasting insulin."""
    if not igi.calculated:
        return _fail("Oral disposition index requires a calculated insulinogenic index.")
    if is_missing(fasting_insulin):
        return _fail("Oral disposition index needs fasting insulin.")
    insulin = insulin_to_uU(fasting_insulin, insulin_unit)
    if insulin <= 0:
        return _fail("Oral disposition index is undefined when fasting insulin is zero or negative.")
    assert igi.value is not None
    return IndexValue(igi.value / insulin, "ok", None)


def _row_points(row: pd.Series, columns: dict[int, str]) -> dict[int, float | None]:
    points: dict[int, float | None] = {}
    for minute, column in columns.items():
        if column not in row.index or is_missing(row[column]):
            points[minute] = None
        else:
            points[minute] = float(row[column])
    return points


def _prefix(name: str, result: IndexValue) -> str | None:
    if result.message:
        return f"{name}: {result.message}"
    return None


def calculate_indices(frame: pd.DataFrame, units: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate indices for each row of a wide analysis table.

    Parameters
    ----------
    frame:
        One row per observation, using canonical column names.
    units:
        ``{"glucose": "mmol/L"|"mg/dL", "insulin": "µU/mL"|"pmol/L"}``.
    """
    glucose_unit = units.get("glucose") if units else None
    insulin_unit = units.get("insulin") if units else None
    if not glucose_unit or not insulin_unit:
        raise UnitError("Glucose and insulin units are required before indices are calculated.")

    rows: list[dict] = []
    for _, row in frame.iterrows():
        record: dict = {}
        for column in ("participant_id", "observation_id", "data_origin"):
            if column in frame.columns:
                record[column] = row[column]
        notes: list[str] = []

        homa_ir_value = homa_ir(
            row.get("fasting_glucose"),
            row.get("fasting_insulin"),
            glucose_unit,
            insulin_unit,
        )
        homa_b_value = homa_b(
            row.get("fasting_glucose"),
            row.get("fasting_insulin"),
            glucose_unit,
            insulin_unit,
        )
        igi = insulinogenic_index(
            row.get("fasting_glucose"),
            row.get("glucose_30"),
            row.get("fasting_insulin"),
            row.get("insulin_30"),
            glucose_unit,
            insulin_unit,
        )
        glucose_points = _row_points(row, GLUCOSE_TIME_COLUMNS)
        insulin_points = _row_points(row, INSULIN_TIME_COLUMNS)
        matsuda = matsuda_index(glucose_points, insulin_points, glucose_unit, insulin_unit)
        glucose_auc = trapezoidal_auc(
            [minute for minute, value in glucose_points.items() if value is not None],
            [value for value in glucose_points.values() if value is not None],
        )
        insulin_auc = trapezoidal_auc(
            [minute for minute, value in insulin_points.items() if value is not None],
            [value for value in insulin_points.values() if value is not None],
        )
        odi = oral_disposition_index(igi, row.get("fasting_insulin"), insulin_unit)

        values = {
            "homa_ir": homa_ir_value,
            "homa_b": homa_b_value,
            "insulinogenic_index": igi,
            "matsuda_index": matsuda,
            "glucose_auc": glucose_auc,
            "insulin_auc": insulin_auc,
            "oral_disposition_index": odi,
        }
        for name, result in values.items():
            record[name] = result.value
            note = _prefix(name, result)
            if note:
                notes.append(note)
        record["calculation_notes"] = " | ".join(notes)
        rows.append(record)

    result = pd.DataFrame(rows)
    summary = _summary(result, glucose_unit, insulin_unit)
    return result, summary


def _summary(result: pd.DataFrame, glucose_unit: str, insulin_unit: str) -> pd.DataFrame:
    units = {
        "homa_ir": "HOMA1-IR (dimensionless as conventionally reported)",
        "homa_b": "HOMA1-B (percent, conventional HOMA1 scale)",
        "insulinogenic_index": "(µU/mL) / (mmol/L)",
        "matsuda_index": "Matsuda index (mg/dL and µU/mL form)",
        "glucose_auc": f"{glucose_unit}·min",
        "insulin_auc": f"{insulin_unit}·min",
        "oral_disposition_index": "1 / (mmol/L), from IGI / fasting insulin",
    }
    rows = []
    for column in INDEX_COLUMNS:
        calculated = int(result[column].notna().sum()) if column in result.columns else 0
        total = len(result)
        rows.append(
            {
                "index": column,
                "formula": FORMULAS[column],
                "assumptions": ASSUMPTIONS[column],
                "unit": units[column],
                "n_calculated": calculated,
                "n_not_calculated": total - calculated,
                "homa2": HOMA2_STATEMENT if column.startswith("homa") else "",
            }
        )
    return pd.DataFrame(rows)


def definition_table() -> pd.DataFrame:
    """Return the documented formulas for export and the research guide."""
    return pd.DataFrame(
        {
            "index": list(FORMULAS),
            "formula": [FORMULAS[key] for key in FORMULAS],
            "assumptions": [ASSUMPTIONS[key] for key in FORMULAS],
            "homa2_statement": [HOMA2_STATEMENT if key.startswith("homa") else "" for key in FORMULAS],
        }
    )
