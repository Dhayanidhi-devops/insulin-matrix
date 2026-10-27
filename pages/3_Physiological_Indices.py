from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import HOMA2_STATEMENT
from src.exports import indices_workbook
from src.indices import ASSUMPTIONS, FORMULAS, INDEX_COLUMNS
from src.services import calculate_and_store
from src.ui import download_bytes, page_header, require_validated, set_active, show_exception

page_header(
    "Physiological Indices",
    "HOMA1, the insulinogenic index, the Matsuda index, trapezoidal areas, and the oral disposition index. These are calculated quantities, not assays or diagnoses.",
)
st.caption(HOMA2_STATEMENT)
active = require_validated()

readiness = active["validation"].readiness
st.subheader("What can be calculated")
st.dataframe([{"index": key, "status": value} for key, value in readiness.items()], width="stretch", hide_index=True)

for key in INDEX_COLUMNS:
    with st.expander(key.replace("_", " ")):
        st.write(FORMULAS[key])
        st.caption(ASSUMPTIONS[key])

if st.button("Calculate indices", type="primary"):
    try:
        updated = calculate_and_store(active)
        set_active(updated)
        st.rerun()
    except Exception as exc:
        show_exception(exc)

indices = active.get("indices")
summary = active.get("index_summary")
if indices is None or summary is None:
    st.info("Indices have not been calculated for the active dataset.")
    st.stop()

st.subheader("Results")
st.caption(f"{active['index_value_count']} index values were calculated. Blank cells are explained in calculation notes.")
st.dataframe(summary, width="stretch", hide_index=True)
st.dataframe(indices, width="stretch", hide_index=True)
notes = indices.loc[indices["calculation_notes"].astype(str).str.len() > 0, "calculation_notes"]
if len(notes):
    st.subheader("Rows that were not fully calculated")
    st.dataframe(indices.loc[indices["calculation_notes"].astype(str).str.len() > 0], width="stretch", hide_index=True)

download_bytes(
    "Download calculated indices",
    indices_workbook(indices, summary, dataset_id=active["dataset_id"], dataset_type=active["dataset_type"]),
    "Insulin_Matrix_Calculated_Indices.xlsx",
    "download_indices",
)
