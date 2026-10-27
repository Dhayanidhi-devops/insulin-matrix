from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import CANONICAL_VARIABLES, GLUCOSE_UNIT_LABELS, INSULIN_LABEL_TO_UNIT, INSULIN_UNIT_LABELS, UNMAPPED
from src.data_loader import apply_validation
from src.database import Database
from src.exports import validation_workbook
from src.ui import download_bytes, page_header, render_issues, require_dataset, set_active, show_exception, sync_dataset_widgets

page_header(
    "Variable Mapping and Validation",
    "Map spreadsheet columns to research variables and set glucose and insulin units. Units are not guessed from the values.",
)
active = require_dataset()
sync_dataset_widgets(active)
frame = active["frame"]
columns = list(frame.columns)
saved_mapping = active.get("mapping") or {}
saved_units = active.get("units") or {}

st.caption("Choose “not mapped” for anything this worksheet does not contain. Saving the mapping does not change the uploaded file.")
if active["dataset_type"] == "synthetic":
    st.info("The synthetic dataset mapping and units were filled from the generator specification. Review them, then run validation.")

grouped: dict[str, list] = {}
for spec in CANONICAL_VARIABLES:
    grouped.setdefault(spec["group"], []).append(spec)

mapping: dict[str, str | None] = {}
for group, specs in grouped.items():
    st.subheader(group)
    for spec in specs:
        key = f"map_{spec['key']}"
        options = [UNMAPPED] + columns
        if key not in st.session_state:
            current = saved_mapping.get(spec["key"])
            st.session_state[key] = current if current in columns else UNMAPPED
        if st.session_state[key] not in options:
            st.session_state[key] = UNMAPPED
        selected = st.selectbox(spec["label"], options, key=key)
        mapping[spec["key"]] = None if selected == UNMAPPED else selected

st.subheader("Units")
glucose_options = ["Select a unit", *GLUCOSE_UNIT_LABELS]
insulin_options = ["Select a unit", *INSULIN_UNIT_LABELS]
if "unit_glucose" not in st.session_state:
    st.session_state["unit_glucose"] = saved_units.get("glucose") if saved_units.get("glucose") in GLUCOSE_UNIT_LABELS else "Select a unit"
if "unit_insulin" not in st.session_state:
    stored_insulin = saved_units.get("insulin")
    label = next((name for name, canonical in INSULIN_LABEL_TO_UNIT.items() if canonical == stored_insulin), "Select a unit")
    st.session_state["unit_insulin"] = label if label in insulin_options else "Select a unit"
glucose_choice = st.selectbox("Glucose unit", glucose_options, key="unit_glucose")
insulin_choice = st.selectbox("Insulin unit", insulin_options, key="unit_insulin")
st.caption("mIU/L uses the same numbers as µU/mL. pmol/L is converted with 6.945 pmol/L per µU/mL when an index needs µU/mL.")

mapped_sources = {value for value in mapping.values() if value}
remaining = [column for column in columns if column not in mapped_sources and column not in {"data_origin", "generator_seed"}]
if "extra_covariates" not in st.session_state:
    st.session_state["extra_covariates"] = [column for column in (active.get("extra_covariates") or []) if column in remaining]
else:
    st.session_state["extra_covariates"] = [column for column in st.session_state["extra_covariates"] if column in remaining]
extras = st.multiselect("Additional research covariates", remaining, key="extra_covariates")

target_options = [UNMAPPED] + columns
if "map_target" not in st.session_state:
    current_target = saved_mapping.get("target")
    st.session_state["map_target"] = current_target if current_target in columns else UNMAPPED
if st.session_state["map_target"] not in target_options:
    st.session_state["map_target"] = UNMAPPED
target = st.selectbox("Modelling target", target_options, key="map_target")
mapping["target"] = None if target == UNMAPPED else target

units = {
    "glucose": None if glucose_choice == "Select a unit" else glucose_choice,
    "insulin": None if insulin_choice == "Select a unit" else INSULIN_LABEL_TO_UNIT.get(insulin_choice, insulin_choice),
}

if st.button("Validate mapped dataset", type="primary"):
    try:
        updated = dict(active)
        updated["mapping"] = mapping
        updated["units"] = units
        updated["extra_covariates"] = extras
        validated = apply_validation(Database(), updated)
        set_active(validated)
        st.rerun()
    except Exception as exc:
        show_exception(exc)

report = active.get("validation")
if report is None:
    st.info("Validation has not been run for the current mapping.")
else:
    st.subheader("Validation result")
    st.write(f"Status: **{report.status.replace('_', ' ')}**. Layout: **{report.layout}**.")
    render_issues(report)
    if report.readiness:
        st.subheader("Index readiness")
        st.dataframe(
            [{"index": key, "status": value} for key, value in report.readiness.items()],
            width="stretch",
            hide_index=True,
        )
    profile_columns = report.to_frame()
    from src.excel_handler import profile_table

    download_bytes(
        "Download validation report",
        validation_workbook(
            profile_columns if not profile_columns.empty else report.to_frame(),
            profile_table(frame),
            dataset_id=active["dataset_id"],
            dataset_type=active["dataset_type"],
        ),
        "Insulin_Matrix_Validation_Report.xlsx",
        "download_validation",
    )
