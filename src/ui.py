"""Shared Streamlit layout for Insulin Matrix."""

from __future__ import annotations

import streamlit as st

from src.config import APP_NAME, DISCLAIMER, SUBTITLE
from src.database import Database
from src.validation import ValidationReport

ACTIVE_KEY = "active"
RESULT_KEY = "last_result"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px;}
        h1, h2, h3 {color: #0B3D5C; letter-spacing: -0.02em;}
        .im-kicker {color: #0F766E; font-weight: 650; letter-spacing: 0.08em;
                    text-transform: uppercase; font-size: 0.78rem; margin-bottom: 0.15rem;}
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #D7E3EE;
            border-left: 4px solid #0F766E;
            padding: 0.7rem 0.85rem;
            border-radius: 8px;
        }
        div[data-testid="stSidebar"] {background: #F3F7FB;}
        .im-note {color: #3E5166; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"**{APP_NAME}**")
        st.caption(SUBTITLE)
        active = get_active()
        if active is None:
            st.caption("No dataset is active in this session.")
        else:
            kind = "Synthetic demonstration" if active["dataset_type"] == "synthetic" else "Research dataset"
            st.caption(f"{kind}: {active['display_name']}")
            validation = active.get("validation")
            if isinstance(validation, ValidationReport):
                st.caption(f"Validation: {validation.status.replace('_', ' ')}")
        st.caption(DISCLAIMER)


def page_header(title: str, lede: str) -> None:
    st.markdown(f"<p class='im-kicker'>{APP_NAME}</p>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(lede)
    active = get_active()
    if active and active.get("dataset_type") == "synthetic":
        st.info("Synthetic demonstration data is active. Results from this table are not clinical evidence.", icon="ℹ️")


def get_active() -> dict | None:
    return st.session_state.get(ACTIVE_KEY)


def set_active(active: dict) -> None:
    st.session_state[ACTIVE_KEY] = active
    st.session_state.pop("widget_dataset_id", None)


def sync_dataset_widgets(active: dict) -> None:
    """Drop widget values when the active dataset changes so old mappings are not reused."""
    if st.session_state.get("widget_dataset_id") == active["dataset_id"]:
        return
    reset_keys = {
        "unit_glucose",
        "unit_insulin",
        "map_target",
        "extra_covariates",
        "feature_selection",
        "model_target",
        "model_task",
        "model_name",
        "missing_strategy",
        "numeric_impute",
        "scale_numeric",
        "normalize_units",
        "derived_features",
        "drop_exact_duplicates",
        "allow_row_split",
    }
    for key in list(st.session_state.keys()):
        if key.startswith("map_") or key in reset_keys:
            del st.session_state[key]
    st.session_state["widget_dataset_id"] = active["dataset_id"]


def require_dataset() -> dict:
    active = get_active()
    if not active:
        st.info("Load the synthetic demonstration dataset from the Research Dashboard, or upload a workbook in Excel Dataset Explorer.")
        st.stop()
    return active


def require_validated() -> dict:
    active = require_dataset()
    validation = active.get("validation")
    if not isinstance(validation, ValidationReport):
        st.info("Open Variable Mapping and Validation, map the columns and units, then run validation.")
        st.stop()
    if validation.has_errors():
        st.error("Validation reported errors. Resolve them before this step. No rows were deleted automatically.")
        st.dataframe(validation.to_frame(), width="stretch", hide_index=True)
        st.stop()
    if active.get("analysis_frame") is None:
        st.info("The analysis table is missing. Run validation again.")
        st.stop()
    return active


def render_issues(report: ValidationReport) -> None:
    if not report.issues:
        st.success("Validation found no errors or warnings.")
        return
    errors = report.to_frame()
    error_rows = errors[errors["severity"] == "error"] if not errors.empty else errors
    warning_rows = errors[errors["severity"] == "warning"] if not errors.empty else errors
    if report.error_count:
        st.error(f"{report.error_count} error(s) need to be resolved before continuing.")
        st.dataframe(error_rows, width="stretch", hide_index=True)
    if report.warning_count:
        st.warning(f"{report.warning_count} warning(s). Review them before interpreting results. Rows were not deleted.")
        st.dataframe(warning_rows, width="stretch", hide_index=True)


def metric_row(items: list[tuple[str, str | int | float]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def download_bytes(label: str, data: bytes, file_name: str, key: str) -> None:
    st.download_button(label, data=data, file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=key)


def show_exception(exc: BaseException) -> None:
    from src.errors import public_message

    st.error(public_message(exc))


def recent_experiments(limit: int = 8) -> list[dict]:
    try:
        return Database().list_experiments()[:limit]
    except Exception as exc:
        st.error("The experiment history could not be read.")
        show_exception(exc)
        return []
