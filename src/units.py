"""Unit conversions for glucose and insulin.

HOMA1 uses a factor of 18 mg/dL per mmol/L so that dividing by 22.5 (mmol/L)
and dividing by 405 (mg/dL) are the same calculation. Insulin conversion uses
6.945 pmol/L per µU/mL. mIU/L is the same numeric scale as µU/mL.
"""

from __future__ import annotations

import math

import pandas as pd

from src.config import GLUCOSE_MGDL_PER_MMOL, INSULIN_PMOL_PER_UU
from src.errors import UnitError

def _fold_unit(unit: str) -> str:
    """Casefold a unit and treat the micro sign and Greek mu as the same character.

    Unicode casefold maps the micro sign to Greek mu, so both spellings have to
    be normalized after casefold or µU/mL will not match the alias table.
    """
    text = str(unit).strip().casefold()
    return text.replace("µ", "μ")


def _alias_table(pairs: dict[str, str]) -> dict[str, str]:
    return {_fold_unit(label): canonical for label, canonical in pairs.items()}


_GLUCOSE_ALIASES = _alias_table({"mmol/L": "mmol/L", "mg/dL": "mg/dL"})
_INSULIN_ALIASES = _alias_table(
    {
        "µU/mL": "µU/mL",
        "μU/mL": "µU/mL",
        "uU/mL": "µU/mL",
        "mIU/L": "µU/mL",
        "pmol/L": "pmol/L",
    }
)


def canonicalize_glucose_unit(unit: str | None) -> str:
    if unit is None or str(unit).strip() == "":
        raise UnitError("Glucose unit is missing. Choose mmol/L or mg/dL.")
    key = _fold_unit(unit)
    if key not in _GLUCOSE_ALIASES:
        raise UnitError(
            f"Unsupported glucose unit {unit!r}. Supported units are mmol/L and mg/dL."
        )
    return _GLUCOSE_ALIASES[key]


def canonicalize_insulin_unit(unit: str | None) -> str:
    if unit is None or str(unit).strip() == "":
        raise UnitError("Insulin unit is missing. Choose µU/mL, mIU/L, or pmol/L.")
    key = _fold_unit(unit)
    if key not in _INSULIN_ALIASES:
        raise UnitError(
            f"Unsupported insulin unit {unit!r}. Supported units are µU/mL, mIU/L, and pmol/L."
        )
    return _INSULIN_ALIASES[key]


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _as_float(value) -> float:
    if is_missing(value):
        raise ValueError("missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite")
    return number


def glucose_to_mmol(value, unit: str) -> float:
    canonical = canonicalize_glucose_unit(unit)
    number = _as_float(value)
    if canonical == "mmol/L":
        return number
    return number / GLUCOSE_MGDL_PER_MMOL


def glucose_to_mgdl(value, unit: str) -> float:
    return glucose_to_mmol(value, unit) * GLUCOSE_MGDL_PER_MMOL


def insulin_to_uU(value, unit: str) -> float:
    canonical = canonicalize_insulin_unit(unit)
    number = _as_float(value)
    if canonical == "µU/mL":
        return number
    return number / INSULIN_PMOL_PER_UU


def insulin_to_pmol(value, unit: str) -> float:
    return insulin_to_uU(value, unit) * INSULIN_PMOL_PER_UU


def convert_glucose_value(value, from_unit: str, to_unit: str) -> float:
    """Convert one glucose number between supported units."""
    mmol = glucose_to_mmol(value, from_unit)
    target = canonicalize_glucose_unit(to_unit)
    if target == "mmol/L":
        return mmol
    return mmol * GLUCOSE_MGDL_PER_MMOL


def convert_insulin_value(value, from_unit: str, to_unit: str) -> float:
    """Convert one insulin number between supported units."""
    uu = insulin_to_uU(value, from_unit)
    target = canonicalize_insulin_unit(to_unit)
    if target == "µU/mL":
        return uu
    return uu * INSULIN_PMOL_PER_UU
