from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import EXCLUDED_FROM_FEATURES
from src.feature_engineering import DERIVED_SPECS, duplicate_summary, missing_summary
from src.preprocessing import constant_features, incomplete_feature_rows
from src.ui import page_header, require_validated, sync_dataset_widgets
from src.validation import leakage_issues

page_header(
    "Data Processing",
    "Choose how missing values, duplicates, units, and derived columns are handled. Imputation statistics are learned later, from the training partition only.",
)
active = require_validated()
sync_dataset_widgets(active)
frame = active["analysis_frame"]
st.caption(f"Analysis table: {len(frame)} rows, layout {active.get('layout')}. The uploaded workbook is unchanged.")

candidate_columns = [column for column in frame.columns if column not in EXCLUDED_FROM_FEATURES]
st.subheader("Missing values")
st.dataframe(missing_summary(frame, candidate_columns), width="stretch", hide_index=True)
strategy = st.radio(
    "Missing-value handling",
    ["pipeline_impute", "drop_incomplete"],
    format_func=lambda value: "Impute inside the training pipeline" if value == "pipeline_impute" else "Drop incomplete rows when training",
    key="missing_strategy",
)
impute = st.selectbox("Numeric imputation statistic", ["median", "mean"], key="numeric_impute")
st.caption("The chosen statistic is estimated on training participants and stored in the model pipeline. It is not fit on the test participants.")

duplicates = duplicate_summary(frame)
st.subheader("Duplicate rows")
st.write(f"Exact duplicate rows: **{duplicates['exact_duplicate_rows']}**. Duplicate participant/observation keys: **{duplicates['duplicate_key_rows']}**.")
st.checkbox(
    f"When training, drop {duplicates['exact_duplicate_rows']} exact duplicate rows",
    key="drop_exact_duplicates",
)
if duplicates["exact_duplicate_rows"] and not st.session_state.get("drop_exact_duplicates"):
    st.warning("Exact duplicates are still in the table. They will not be removed unless you check the box.")

st.subheader("Features and target")
if "feature_selection" in st.session_state:
    st.session_state["feature_selection"] = [column for column in st.session_state["feature_selection"] if column in candidate_columns]
features = st.multiselect("Model inputs", candidate_columns, key="feature_selection")
target_options = candidate_columns
default_target = active.get("target_column")
if "model_target" not in st.session_state:
    st.session_state["model_target"] = default_target if default_target in target_options else (target_options[0] if target_options else None)
if st.session_state.get("model_target") not in target_options and target_options:
    st.session_state["model_target"] = target_options[0]
target = st.selectbox("Target", target_options, key="model_target") if target_options else None
task = st.radio("Task", ["Regression", "Classification"], key="model_task")

label_by_key = {spec["key"]: spec["label"] for spec in DERIVED_SPECS}
derived_options = list(label_by_key)
if "derived_features" in st.session_state:
    st.session_state["derived_features"] = [key for key in st.session_state["derived_features"] if key in label_by_key]
derived_choice = st.multiselect(
    "Derived columns to add before training",
    derived_options,
    format_func=lambda key: label_by_key[key],
    key="derived_features",
)
st.checkbox("Add mmol/L and µU/mL copies of mapped glucose and insulin columns", key="normalize_units")
st.checkbox("Standardize numeric features in the training pipeline", value=True, key="scale_numeric")
st.caption("Standardization is recommended for linear and logistic models. Tree models do not need it. The scaler is fit on the training partition only.")

selected_features = list(features)
derived_keys = list(derived_choice)
if target:
    issues = leakage_issues(selected_features, target)
    if issues:
        for issue in issues:
            if issue.severity == "error":
                st.error(issue.message)
            else:
                st.warning(issue.message)
    if strategy == "drop_incomplete":
        prospective = selected_features + [key for key in derived_keys if key not in frame.columns]
        existing = [column for column in prospective if column in frame.columns]
        if existing:
            count = incomplete_feature_rows(frame, existing)
            st.warning(f"Dropping incomplete rows would remove {count} current rows that are missing a selected feature. Derived columns are checked again at training time.")
    constants = constant_features(frame, [column for column in selected_features if column in frame.columns])
    if constants:
        st.error("These selected columns have a single value: " + ", ".join(constants) + ". Unselect them before training.")

st.subheader("Preprocessing that will be used")
st.dataframe(
    [
        {"setting": "Missing values", "value": "Median/mean imputer inside the pipeline" if strategy == "pipeline_impute" else "Drop rows that are missing a selected feature"},
        {"setting": "Numeric impute", "value": impute},
        {"setting": "Standardize numerics", "value": "Yes" if st.session_state.get("scale_numeric", True) else "No"},
        {"setting": "Normalize units", "value": "Yes" if st.session_state.get("normalize_units") else "No"},
        {"setting": "Derived columns", "value": ", ".join(derived_keys) or "None"},
        {"setting": "Features", "value": ", ".join(selected_features) or "None selected"},
        {"setting": "Target", "value": target or "None"},
        {"setting": "Task", "value": task},
        {"setting": "Excluded identifiers", "value": "participant_id, observation_id, data_origin, generator_seed"},
    ],
    width="stretch",
    hide_index=True,
)
st.caption("Derived index columns are formulas of the same row. They are not estimated from other participants.")
