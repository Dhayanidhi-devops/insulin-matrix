"""Insulin Matrix entry point.

Run from the project folder:

    python -m streamlit run app.py

If the `python` command is not available, use `python3 -m streamlit run app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Insulin Matrix",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui import apply_theme, render_sidebar

apply_theme()
render_sidebar()

navigation = st.navigation(
    [
        st.Page("pages/0_Research_Dashboard.py", title="Research Dashboard", default=True),
        st.Page("pages/1_Dataset_Explorer.py", title="Excel Dataset Explorer"),
        st.Page("pages/2_Variable_Mapping.py", title="Variable Mapping and Validation"),
        st.Page("pages/3_Physiological_Indices.py", title="Physiological Indices"),
        st.Page("pages/4_Data_Processing.py", title="Data Processing"),
        st.Page("pages/5_Model_Training.py", title="Model Training"),
        st.Page("pages/6_Model_Evaluation.py", title="Model Evaluation"),
        st.Page("pages/7_Prediction_Lab.py", title="Prediction Lab"),
        st.Page("pages/8_Experiment_History.py", title="Experiment History"),
        st.Page("pages/9_Research_Guide.py", title="Documentation and Research Notes"),
    ],
    expanded=True,
)
navigation.run()
