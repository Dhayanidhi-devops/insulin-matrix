"""Constants, storage locations, and canonical variable definitions."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Insulin Matrix"
SUBTITLE = "AI-Powered Insulin and Metabolic Research Platform"
VERSION = "1.0.0"
DISCLAIMER = (
    "Insulin Matrix is a research prototype. Calculated indices and model "
    "outputs are not measurements, diagnoses, treatment recommendations, "
    "or insulin doses."
)
HOMA2_STATEMENT = (
    "HOMA2 is a separate nonlinear computer model distributed by the "
    "University of Oxford. Insulin Matrix does not approximate HOMA2 with "
    "the HOMA1 formulas."
)

# 18 mg/dL per mmol/L makes the 22.5 and 405 HOMA1 formulas agree.
GLUCOSE_MGDL_PER_MMOL = 18.0
INSULIN_PMOL_PER_UU = 6.945
HOMA_IR_DENOMINATOR = 22.5
HOMA_B_OFFSET_MMOL = 3.5
MATSUDA_CONSTANT = 10_000.0

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 250_000
PREVIEW_ROWS = 25
PLOT_ROW_CAP = 4_000
MIN_PARTICIPANTS_FOR_TRAINING = 8

GLUCOSE_UNIT_LABELS = ("mmol/L", "mg/dL")
INSULIN_UNIT_LABELS = (
    "µU/mL",
    "mIU/L (identical scale to µU/mL)",
    "pmol/L",
)

# Canonical stored units after the display label is interpreted.
INSULIN_LABEL_TO_UNIT = {
    "µU/mL": "µU/mL",
    "mIU/L (identical scale to µU/mL)": "µU/mL",
    "pmol/L": "pmol/L",
}

OGTT_TIMES = (0, 30, 60, 90, 120)

GLUCOSE_TIME_COLUMNS = {
    0: "fasting_glucose",
    30: "glucose_30",
    60: "glucose_60",
    90: "glucose_90",
    120: "glucose_120",
}
INSULIN_TIME_COLUMNS = {
    0: "fasting_insulin",
    30: "insulin_30",
    60: "insulin_60",
    90: "insulin_90",
    120: "insulin_120",
}

IDENTIFIER_FIELDS = ("participant_id", "observation_id", "time_point")
BOOKKEEPING_COLUMNS = ("data_origin", "generator_seed")
EXCLUDED_FROM_FEATURES = set(IDENTIFIER_FIELDS) | set(BOOKKEEPING_COLUMNS) | {
    "visit_id",
}

CANONICAL_VARIABLES = [
    {"key": "participant_id", "label": "Participant ID", "group": "Identifiers", "kind": "id"},
    {"key": "observation_id", "label": "Observation or visit ID", "group": "Identifiers", "kind": "id"},
    {"key": "time_point", "label": "Time point (minutes, long format)", "group": "Identifiers", "kind": "time"},
    {"key": "fasting_glucose", "label": "Fasting glucose", "group": "Glucose", "kind": "glucose"},
    {"key": "glucose_30", "label": "Glucose at 30 minutes", "group": "Glucose", "kind": "glucose"},
    {"key": "glucose_60", "label": "Glucose at 60 minutes", "group": "Glucose", "kind": "glucose"},
    {"key": "glucose_90", "label": "Glucose at 90 minutes", "group": "Glucose", "kind": "glucose"},
    {"key": "glucose_120", "label": "Glucose at 120 minutes", "group": "Glucose", "kind": "glucose"},
    {"key": "glucose", "label": "Glucose (single column, long format)", "group": "Glucose", "kind": "glucose"},
    {"key": "fasting_insulin", "label": "Fasting insulin", "group": "Insulin", "kind": "insulin"},
    {"key": "insulin_30", "label": "Insulin at 30 minutes", "group": "Insulin", "kind": "insulin"},
    {"key": "insulin_60", "label": "Insulin at 60 minutes", "group": "Insulin", "kind": "insulin"},
    {"key": "insulin_90", "label": "Insulin at 90 minutes", "group": "Insulin", "kind": "insulin"},
    {"key": "insulin_120", "label": "Insulin at 120 minutes", "group": "Insulin", "kind": "insulin"},
    {"key": "insulin", "label": "Insulin (single column, long format)", "group": "Insulin", "kind": "insulin"},
    {"key": "age", "label": "Age (years)", "group": "Covariates", "kind": "numeric"},
    {"key": "sex", "label": "Sex", "group": "Covariates", "kind": "category"},
    {"key": "bmi", "label": "Body mass index", "group": "Covariates", "kind": "numeric"},
    {"key": "weight_kg", "label": "Weight (kg)", "group": "Covariates", "kind": "numeric"},
    {"key": "height_cm", "label": "Height (cm)", "group": "Covariates", "kind": "numeric"},
    {"key": "waist_cm", "label": "Waist circumference (cm)", "group": "Covariates", "kind": "numeric"},
    {"key": "hba1c", "label": "HbA1c", "group": "Covariates", "kind": "numeric"},
]

UNMAPPED = "— not mapped —"
NUMERIC_KINDS = {"time", "glucose", "insulin", "numeric"}
GLUCOSE_FIELDS = {item["key"] for item in CANONICAL_VARIABLES if item["kind"] == "glucose"}
INSULIN_FIELDS = {item["key"] for item in CANONICAL_VARIABLES if item["kind"] == "insulin"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def storage_home() -> Path:
    """Return the writable project root, or INSULIN_MATRIX_HOME when set."""
    override = os.environ.get("INSULIN_MATRIX_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT


def data_dir() -> Path:
    path = storage_home() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_dir() -> Path:
    path = data_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path:
    path = data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir() -> Path:
    path = storage_home() / "artifacts" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "insulin_matrix.sqlite"


def safe_under_home(relative: str) -> Path:
    """Resolve a stored relative path and reject anything outside storage."""
    home = storage_home().resolve()
    candidate = (home / relative).resolve()
    if not candidate.is_relative_to(home):
        from src.errors import StorageError

        raise StorageError("Refusing to use a path outside the application storage directory.")
    return candidate
