from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.database import Database, load_json
from src.services import experiment_comparison
from src.ui import RESULT_KEY, page_header, show_exception

page_header(
    "Experiment History",
    "Local experiment records. Deleting a record also deletes its model file. Personal identifiers from the spreadsheet are not stored here.",
)

dataset_type = st.selectbox("Dataset type", ["All", "real", "synthetic"])
task = st.selectbox("Task", ["All", "regression", "classification"])
text = st.text_input("Search target, model, label, or experiment ID")
rows = Database().list_experiments(
    dataset_type=None if dataset_type == "All" else dataset_type,
    task=None if task == "All" else task,
    text=text.strip() or None,
)
table = experiment_comparison(rows)
if table.empty:
    st.info("No experiments match these filters.")
    st.stop()

st.dataframe(table, width="stretch", hide_index=True)
labels = {f"{row['created_at']} · {row['model_name']} · {row['experiment_id']}": row for row in rows}
selected_label = st.selectbox("Experiment detail", list(labels))
selected = labels[selected_label]
metrics = load_json(selected.get("metrics_json"), {})
features = load_json(selected.get("features_json"), [])
hyperparameters = load_json(selected.get("hyperparameters_json"), {})
st.subheader(selected["experiment_id"])
if selected["dataset_type"] == "synthetic":
    st.warning("Synthetic demonstration experiment. Not clinical evidence.")
st.write(f"Model: **{selected['model_name']}**. Target: `{selected['target']}`. Task: {selected['task']}.")
st.caption(
    f"{selected['created_at']} · {selected['dataset_label']} · "
    f"train participants {selected['n_train_participants']} · test participants {selected['n_test_participants']}"
)
st.markdown("**Features**")
st.write(", ".join(features) if features else "None recorded")
st.markdown("**Hyperparameters**")
st.json(hyperparameters)
st.markdown("**Metrics**")
st.json(metrics)
if selected.get("notes"):
    st.markdown("**Notes**")
    st.write(selected["notes"])

st.subheader("Delete")
st.caption("Deletion removes the history row and the joblib model file. This cannot be undone from the application.")
if st.checkbox("Delete the selected experiment and its saved model", key="confirm_delete_experiment"):
    if st.button("Delete experiment"):
        try:
            outcome = Database().delete_experiment(selected["experiment_id"])
            last = st.session_state.get(RESULT_KEY)
            if isinstance(last, dict) and last.get("experiment_id") == selected["experiment_id"]:
                st.session_state.pop(RESULT_KEY, None)
            if outcome.get("removed_file"):
                st.success("The experiment and its model file were deleted.")
            else:
                st.success("The experiment was deleted. No model file was present.")
            st.rerun()
        except Exception as exc:
            show_exception(exc)
else:
    st.caption("Check the box to enable deletion.")
