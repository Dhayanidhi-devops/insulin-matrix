from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import PLOT_ROW_CAP, PREVIEW_ROWS
from src.data_loader import read_saved_worksheet, register_dataset, save_upload, worksheet_names
from src.database import Database
from src.excel_handler import profile_table
from src.exports import summary_workbook
from src.ui import download_bytes, get_active, page_header, set_active, show_exception
from src.visualization import box_plot, correlation_heatmap, histogram

page_header(
    "Excel Dataset Explorer",
    "Upload one workbook, choose a worksheet, and inspect it before mapping. Sheets are not combined, and the original file is not overwritten.",
)

uploaded = st.file_uploader("Workbook or CSV", type=["xlsx", "xlsm", "csv"])
if uploaded is not None:
    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("pending_token") != token:
        try:
            _path, relative = save_upload(uploaded.name, uploaded.getvalue())
            sheets = worksheet_names(relative)
            st.session_state["pending_upload"] = {
                "relative_path": relative,
                "original_name": uploaded.name,
                "sheets": sheets,
            }
            st.session_state["pending_token"] = token
        except Exception as exc:
            show_exception(exc)
            st.session_state.pop("pending_upload", None)

pending = st.session_state.get("pending_upload")
if pending:
    st.write(f"File received: **{pending['original_name']}**")
    sheet = pending["sheets"][0] if len(pending["sheets"]) == 1 else st.selectbox("Worksheet", pending["sheets"])
    if len(pending["sheets"]) > 1:
        st.caption("This workbook has more than one worksheet. Only the worksheet you select will be loaded.")
    try:
        preview = read_saved_worksheet(pending["relative_path"], sheet)
    except Exception as exc:
        preview = None
        show_exception(exc)
    if preview is not None:
        st.caption(f"Worksheet `{sheet}` has {len(preview):,} rows and {preview.shape[1]} columns. Showing the first {min(PREVIEW_ROWS, len(preview))} rows.")
        st.dataframe(preview.head(PREVIEW_ROWS), width="stretch", hide_index=True)
        if st.button("Use this worksheet", type="primary"):
            try:
                display = pending["original_name"]
                if len(pending["sheets"]) > 1:
                    display = f"{display} — {sheet}"
                active = register_dataset(
                    Database(),
                    preview,
                    display_name=display,
                    dataset_type="real",
                    sheet_name=sheet,
                    storage_relpath=pending["relative_path"],
                )
                set_active(active)
                st.session_state.pop("pending_upload", None)
                st.session_state.pop("pending_token", None)
                st.rerun()
            except Exception as exc:
                show_exception(exc)

active = get_active()
if active is None:
    st.info("No dataset is active yet. Upload a workbook above, or load the synthetic demonstration dataset from the Research Dashboard.")
    st.stop()

frame: pd.DataFrame = active["frame"]
profile = profile_table(frame)
st.subheader(active["display_name"])
kind = "Synthetic demonstration" if active["dataset_type"] == "synthetic" else "Research dataset"
st.caption(f"{kind} · worksheet {active['sheet_name']} · dataset {active['dataset_id']}")
if profile["header_notes"]:
    for note in profile["header_notes"]:
        st.caption(note)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", profile["n_rows"])
c2.metric("Columns", profile["n_columns"])
c3.metric("Duplicate rows", profile["duplicate_rows"])
c4.metric("Missing cells", int(frame.isna().sum().sum()))

st.subheader("Columns and data types")
st.dataframe(profile["columns"], width="stretch", hide_index=True)
if not profile["numeric_summary"].empty:
    st.subheader("Numeric summary")
    st.dataframe(profile["numeric_summary"], width="stretch", hide_index=True)

st.subheader("Preview")
st.caption(f"First {min(PREVIEW_ROWS, len(frame))} of {len(frame)} rows.")
st.dataframe(frame.head(PREVIEW_ROWS), width="stretch", hide_index=True)
if profile["duplicate_rows"]:
    st.warning(f"{profile['duplicate_rows']} exact duplicate rows are present. They are shown here and were not deleted.")
    st.dataframe(frame.loc[frame.duplicated(keep=False)].head(10), width="stretch")

chart_frame = frame
if len(frame) > PLOT_ROW_CAP:
    chart_frame = frame.sample(PLOT_ROW_CAP, random_state=0)
    st.caption(f"Charts use a sample of {PLOT_ROW_CAP} rows so the page stays responsive. Summary tables use every row.")

numeric_columns = list(frame.select_dtypes(include="number").columns)
tabs = st.tabs(["Histogram", "Box plot", "Correlation"])
with tabs[0]:
    if not numeric_columns:
        st.info("This worksheet has no numeric columns to plot.")
    else:
        column = st.selectbox("Numeric column", numeric_columns, key="explorer_hist")
        st.plotly_chart(histogram(chart_frame[column], f"Histogram of {column}"), width="stretch")
with tabs[1]:
    if not numeric_columns:
        st.info("This worksheet has no numeric columns to plot.")
    else:
        chosen = st.multiselect("Columns", numeric_columns, default=numeric_columns[: min(4, len(numeric_columns))], key="explorer_box")
        if chosen:
            st.plotly_chart(box_plot(chart_frame, chosen), width="stretch")
with tabs[2]:
    heat = correlation_heatmap(chart_frame[numeric_columns[:20]] if numeric_columns else chart_frame)
    if heat is None:
        st.info("A correlation heatmap needs at least two numeric columns.")
    else:
        if len(numeric_columns) > 20:
            st.caption("The heatmap uses the first 20 numeric columns.")
        st.plotly_chart(heat, width="stretch")

download_bytes(
    "Download dataset summary",
    summary_workbook(profile, frame, dataset_id=active["dataset_id"], dataset_type=active["dataset_type"], display_name=active["display_name"]),
    "Insulin_Matrix_Dataset_Summary.xlsx",
    "download_summary",
)

st.subheader("Remove the stored workbook")
st.caption("This deletes the local upload, the processed copy, and the registry row. Experiment metrics are kept.")
if st.checkbox("Delete this dataset's stored files", key="confirm_delete_dataset"):
    if st.button("Delete stored dataset"):
        try:
            Database().delete_dataset(active["dataset_id"])
            st.session_state.pop("active", None)
            st.rerun()
        except Exception as exc:
            show_exception(exc)
