from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.data_loader import load_synthetic
from src.database import Database
from src.excel_handler import export_notes, workbook_bytes
from src.services import experiment_comparison
from src.synthetic_data import GENERATION_NOTES, SYNTHETIC_DATASET_NAME, generate_synthetic_dataset
from src.ui import get_active, metric_row, page_header, set_active, show_exception

page_header(
    "Research Dashboard",
    "Track imported worksheets, validation, calculated indices, and saved models. Counts come from the local registry.",
)

try:
    summary = Database().summary_counts()
except Exception as exc:
    summary = None
    show_exception(exc)

if summary is None:
    st.stop()

if summary["datasets"] == 0:
    st.info("No dataset has been loaded yet. The counts below are zero.")

metric_row(
    [
        ("Imported datasets", summary["datasets"]),
        ("Participants", summary["participants"]),
        ("Observations", summary["observations"]),
    ]
)
metric_row(
    [
        ("Calculated index values", summary["index_values"]),
        ("Trained models", summary["models"]),
        ("Latest experiment", "None" if summary["latest_experiment"] is None else summary["latest_experiment"]["model_name"]),
    ]
)
if summary["datasets_without_participant_count"]:
    st.caption(
        f"Participant IDs were not available for {summary['datasets_without_participant_count']} dataset(s). Those datasets are not included in the participant total."
    )

left, right = st.columns(2)
with left:
    st.subheader("Latest experiment")
    latest = summary["latest_experiment"]
    if latest is None:
        st.write("No experiments yet.")
    else:
        st.write(f"**{latest['model_name']}** predicting `{latest['target']}`")
        st.caption(
            f"{latest['created_at']} · {latest['dataset_type']} · "
            f"train participants {latest['n_train_participants']} · test participants {latest['n_test_participants']}"
        )
        if latest["dataset_type"] == "synthetic":
            st.caption("Synthetic experiment. Not clinical evidence.")
with right:
    st.subheader("Dataset validation")
    active = get_active()
    if active is None:
        st.write("No active dataset.")
    else:
        validation = active.get("validation")
        status = "Not validated" if validation is None else validation.status.replace("_", " ")
        st.write(f"**{active['display_name']}**")
        st.caption(f"{active['dataset_type']} · {status} · {len(active['frame'])} rows")

st.subheader("Recent experiments")
history = experiment_comparison()
if history.empty:
    st.write("Train a model to add an experiment. The table stays empty until then.")
else:
    st.dataframe(history.head(8), width="stretch", hide_index=True)

st.subheader("Start here")
st.markdown(
    """
    1. Load the synthetic demonstration dataset, or upload an Excel workbook in Excel Dataset Explorer.
    2. Inspect the worksheet, columns, missing values, and charts.
    3. Map columns and units, then run validation.
    4. Calculate the physiological indices you have the inputs for.
    5. Review missing data and choose features on Data Processing.
    6. Train a model and read the held-out metrics.
    7. Score new rows in Prediction Lab and download the workbook.
    8. Return to Experiment History to compare or delete a run.
    """
)

st.subheader("Synthetic demonstration dataset")
st.caption(GENERATION_NOTES)
if st.button("Load synthetic demonstration dataset", type="primary"):
    try:
        with st.spinner("Generating and validating the synthetic dataset..."):
            loaded = load_synthetic(Database())
        set_active(loaded)
        st.rerun()
    except Exception as exc:
        show_exception(exc)

synthetic = generate_synthetic_dataset()
notes = export_notes(
    dataset_id="",
    dataset_type="synthetic",
    description="Synthetic demonstration workbook generated inside Insulin Matrix.",
)
st.download_button(
    "Download synthetic workbook",
    data=workbook_bytes({"Synthetic_OGTT": synthetic}, notes),
    file_name="Insulin_Matrix_Synthetic_Demonstration.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.caption(SYNTHETIC_DATASET_NAME + ". The workbook is labeled synthetic on the Export_Notes sheet.")
