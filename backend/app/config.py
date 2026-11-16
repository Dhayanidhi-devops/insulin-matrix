"""Application configuration and clinical feature definitions."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Insulin Matrix AI"
APP_VERSION = "2.0.0"
SUBTITLE = "Diabetes Screening & Classification Platform"

MEDICAL_DISCLAIMER = (
    "Demonstration only — not a medical diagnosis. Predictions are generated "
    "from synthetic or research data for model-development education. Do not "
    "use outputs for clinical decision-making, treatment, or insulin dosing."
)

SYNTHETIC_DATA_LABEL = (
    "Dataset is synthetic / simulated. It is not real patient data."
)

# backend/app/config.py -> parents[2] is the repository root when layout is repo/backend/app.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / (
    "Insulin_Matrix_Synthetic_Dataset_10000_Updated.xlsx"
)

# Target classes for the multiclass screening task.
DIABETES_CLASSES = [
    "No Diabetes",
    "Prediabetes",
    "Type 1 Diabetes",
    "Type 2 Diabetes",
]

ID_COLUMN = "Record_ID"
RAW_TYPE_COLUMN = "Diabetes_Type"
CRITERIA_FLAG_COLUMN = "Synthetic_Diabetes_Criteria_Flag"
TARGET_COLUMN = "diabetes_category"

# Never used as model inputs (IDs, labels, or target-derived flags).
LEAKAGE_COLUMNS = {
    ID_COLUMN,
    RAW_TYPE_COLUMN,
    CRITERIA_FLAG_COLUMN,
    TARGET_COLUMN,
    "prediction",
    "predicted_class",
    "predicted_label",
}

NUMERIC_FEATURES = [
    "Age_years",
    "Height_cm",
    "Weight_kg",
    "BMI_kg_m2",
    "Waist_Circumference_cm",
    "Fasting_Glucose_mg_dL",
    "HbA1c_percent",
    "C_Peptide_ng_mL",
    "Fasting_Insulin_uIU_mL",
    "Systolic_BP_mmHg",
    "Diastolic_BP_mmHg",
    "Triglycerides_mg_dL",
    "HDL_mg_dL",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Family_History_Diabetes",
    "Physical_Activity_Level",
]

DEFAULT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Patient Assessment form fields (ordered for the UI).
ASSESSMENT_FIELDS = [
    {"key": "Age_years", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1},
    {"key": "Sex", "label": "Sex", "type": "select", "options": ["Female", "Male"]},
    {"key": "Height_cm", "label": "Height (cm)", "type": "number", "min": 80, "max": 250, "step": 0.1},
    {"key": "Weight_kg", "label": "Weight (kg)", "type": "number", "min": 20, "max": 300, "step": 0.1},
    {"key": "BMI_kg_m2", "label": "BMI (kg/m²)", "type": "number", "min": 10, "max": 80, "step": 0.1},
    {
        "key": "Waist_Circumference_cm",
        "label": "Waist circumference (cm)",
        "type": "number",
        "min": 40,
        "max": 200,
        "step": 0.1,
    },
    {
        "key": "Family_History_Diabetes",
        "label": "Family history of diabetes",
        "type": "select",
        "options": ["No", "Yes"],
    },
    {
        "key": "Physical_Activity_Level",
        "label": "Physical activity",
        "type": "select",
        "options": ["Low", "Moderate", "High"],
    },
    {
        "key": "Fasting_Glucose_mg_dL",
        "label": "Fasting glucose (mg/dL)",
        "type": "number",
        "min": 40,
        "max": 500,
        "step": 0.1,
    },
    {"key": "HbA1c_percent", "label": "HbA1c (%)", "type": "number", "min": 3, "max": 20, "step": 0.1},
    {
        "key": "C_Peptide_ng_mL",
        "label": "C-peptide (ng/mL)",
        "type": "number",
        "min": 0,
        "max": 20,
        "step": 0.01,
    },
    {
        "key": "Fasting_Insulin_uIU_mL",
        "label": "Fasting insulin (µIU/mL)",
        "type": "number",
        "min": 0,
        "max": 300,
        "step": 0.01,
    },
    {
        "key": "Systolic_BP_mmHg",
        "label": "Systolic BP (mmHg)",
        "type": "number",
        "min": 60,
        "max": 250,
        "step": 1,
    },
    {
        "key": "Diastolic_BP_mmHg",
        "label": "Diastolic BP (mmHg)",
        "type": "number",
        "min": 30,
        "max": 150,
        "step": 1,
    },
    {
        "key": "Triglycerides_mg_dL",
        "label": "Triglycerides (mg/dL)",
        "type": "number",
        "min": 20,
        "max": 1000,
        "step": 0.1,
    },
    {"key": "HDL_mg_dL", "label": "HDL (mg/dL)", "type": "number", "min": 10, "max": 150, "step": 0.1},
]


def storage_home() -> Path:
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


def report_dir() -> Path:
    path = storage_home() / "artifacts" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "insulin_matrix.sqlite"
