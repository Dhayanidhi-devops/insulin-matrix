from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER
from src.database import Database
from src.excel_handler import list_worksheets, read_worksheet
from src.exports import prediction_workbook
from src.predictions import convert_inputs_to_model_units, predict_frame, validate_prediction_inputs
from src.services import configuration_from_artifact, open_artifact
from src.ui import download_bytes, page_header, render_issues, show_exception

page_header(
    "Prediction Lab",
    "Score new rows with a saved pipeline. The model is not retrained. Outputs are not diagnoses or insulin doses.",
)
st.caption(DISCLAIMER)

experiments = Database().list_experiments()
if not experiments:
    st.info("Train a model before scoring new rows. There is no saved pipeline yet.")
    st.stop()

labels = {f"{row['created_at']} · {row['model_name']} · {row['target']}": row["experiment_id"] for row in experiments}
choice = st.selectbox("Saved model", list(labels))
try:
    _experiment, artifact = open_artifact(labels[choice])
except Exception as exc:
    show_exception(exc)
    st.stop()

if artifact.get("dataset_type") == "synthetic":
    st.warning("This pipeline was trained on synthetic demonstration data. Predictions are not clinical evidence.")

features = artifact["features"]
st.subheader("Required inputs")
st.write(", ".join(features))
units = artifact.get("units") or {}
if units:
    st.dataframe([{"column": key, "training_unit": value} for key, value in units.items()], width="stretch", hide_index=True)
else:
    st.caption("No glucose or insulin unit metadata was stored for the selected features.")
st.dataframe(
    [{"feature": feature, "role": "numeric" if feature in (artifact.get("preprocessing") or {}).get("numeric_features", []) else "categorical"} for feature in features],
    width="stretch",
    hide_index=True,
)

confirmed = st.checkbox("The values I will score use the units stored with this model", key="units_confirmed")
provided_units = {}
for column, expected in units.items():
    provided_units[column] = st.text_input(f"Unit of the new values in {column}", value=expected, key=f"pred_unit_{column}")

source = st.radio("Input rows", ["Type rows", "Upload a worksheet"], key="prediction_source")
incoming = None
if source == "Type rows":
    template = pd.DataFrame([{feature: None for feature in features}])
    incoming = st.data_editor(template, num_rows="dynamic", width="stretch", key="prediction_editor")
else:
    upload = st.file_uploader("Worksheet for scoring", type=["xlsx", "xlsm", "csv"], key="prediction_upload")
    if upload is not None:
        try:
            from io import BytesIO

            buffer = BytesIO(upload.getvalue())
            suffix = Path(upload.name).suffix.lower()
            sheets = list_worksheets(buffer, format_hint=suffix)
            buffer.seek(0)
            sheet = sheets[0] if len(sheets) == 1 else st.selectbox("Worksheet", sheets, key="prediction_sheet")
            buffer.seek(0)
            incoming = read_worksheet(buffer, None if len(sheets) == 1 else sheet, format_hint=suffix)
            st.dataframe(incoming.head(20), width="stretch", hide_index=True)
        except Exception as exc:
            show_exception(exc)

if st.session_state.get("prediction_converted_id") != artifact.get("experiment_id"):
    st.session_state.pop("converted_prediction_frame", None)

if incoming is not None and st.button("Convert values to the stored units"):
    try:
        converted, notes = convert_inputs_to_model_units(incoming, artifact, provided_units)
        st.session_state["converted_prediction_frame"] = converted
        st.session_state["prediction_converted_id"] = artifact.get("experiment_id")
        for note in notes:
            st.caption(note)
        st.dataframe(converted.head(20), width="stretch", hide_index=True)
    except Exception as exc:
        show_exception(exc)

converted_frame = st.session_state.get("converted_prediction_frame")
frame_to_score = converted_frame if isinstance(converted_frame, pd.DataFrame) else incoming
if frame_to_score is not None:
    units_for_check = provided_units
    if isinstance(converted_frame, pd.DataFrame):
        units_for_check = {column: expected for column, expected in units.items()}
        st.caption("Scoring uses the converted table. The conversion ran only because it was requested.")
    report = validate_prediction_inputs(
        frame_to_score,
        artifact,
        units_confirmed=confirmed,
        provided_units=units_for_check,
    )
    render_issues(report)
    if st.button("Generate predictions", type="primary", disabled=report.has_errors()):
        try:
            predictions = predict_frame(artifact, frame_to_score)
            st.session_state["latest_predictions"] = predictions
            st.session_state["latest_prediction_artifact"] = artifact
            st.success("Predictions were produced by the saved pipeline. The model was not retrained.")
            st.dataframe(predictions, width="stretch", hide_index=True)
        except Exception as exc:
            show_exception(exc)

predictions = st.session_state.get("latest_predictions")
stored_artifact = st.session_state.get("latest_prediction_artifact")
if isinstance(predictions, pd.DataFrame) and stored_artifact is not None:
    download_bytes(
        "Download predictions",
        prediction_workbook(
            predictions,
            configuration_from_artifact(stored_artifact),
            dataset_id=stored_artifact.get("dataset_id") or "",
            dataset_type=stored_artifact.get("dataset_type") or "real",
        ),
        "Insulin_Matrix_Predictions.xlsx",
        "download_predictions",
    )
