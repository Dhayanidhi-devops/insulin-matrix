"""Assemble the worksheets that researchers download."""

from __future__ import annotations

import json

import pandas as pd

from src.excel_handler import export_notes, workbook_bytes
from src.indices import definition_table


def workbook_from_sheets(sheets: dict[str, pd.DataFrame], notes: dict[str, str]) -> bytes:
    return workbook_bytes(sheets, notes)


def summary_workbook(profile: dict, frame: pd.DataFrame, *, dataset_id: str, dataset_type: str, display_name: str) -> bytes:
    overview = pd.DataFrame(
        [
            {"item": "dataset_id", "value": dataset_id},
            {"item": "display_name", "value": display_name},
            {"item": "dataset_type", "value": dataset_type},
            {"item": "rows", "value": profile["n_rows"]},
            {"item": "columns", "value": profile["n_columns"]},
            {"item": "duplicate_rows", "value": profile["duplicate_rows"]},
            {"item": "sheet", "value": profile.get("sheet_name") or ""},
        ]
    )
    sheets = {"Dataset_Summary": overview, "Column_Profile": profile["columns"]}
    if isinstance(profile.get("numeric_summary"), pd.DataFrame) and not profile["numeric_summary"].empty:
        sheets["Numeric_Summary"] = profile["numeric_summary"]
    notes = export_notes(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        description="Column profile and summary statistics for one worksheet.",
    )
    return workbook_from_sheets(sheets, notes)


def validation_workbook(report_frame: pd.DataFrame, profile: dict, *, dataset_id: str, dataset_type: str) -> bytes:
    sheets = {
        "Validation_Report": report_frame,
        "Dataset_Summary": profile["columns"],
    }
    notes = export_notes(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        description="Validation findings for the mapped worksheet. Errors and warnings are listed separately.",
    )
    return workbook_from_sheets(sheets, notes)


def indices_workbook(indices: pd.DataFrame, summary: pd.DataFrame, *, dataset_id: str, dataset_type: str) -> bytes:
    sheets = {
        "Calculated_Indices": indices,
        "Index_Summary": summary,
        "Index_Definitions": definition_table(),
    }
    notes = export_notes(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        description="Calculated physiological indices. These are formulas applied to the mapped columns, not laboratory assays.",
        extra={"homa_version": "HOMA1 only. HOMA2 is not calculated."},
    )
    return workbook_from_sheets(sheets, notes)


def model_workbook(metrics: dict, importance: pd.DataFrame | None, configuration: dict, *, dataset_id: str, dataset_type: str) -> bytes:
    sheets = {
        "Model_Metrics": pd.DataFrame([{"metric": key, "value": _plain(value)} for key, value in metrics.items()]),
        "Experiment_Configuration": _config_frame(configuration),
    }
    if importance is not None and not importance.empty:
        sheets["Feature_Importance"] = importance
    notes = export_notes(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        description="Held-out test metrics and the training configuration that produced them.",
        extra={"metric_set": "Test-partition metrics are the primary result. Training metrics are labeled with a train_ prefix."},
    )
    return workbook_from_sheets(sheets, notes)


def prediction_workbook(predictions: pd.DataFrame, configuration: dict, *, dataset_id: str, dataset_type: str) -> bytes:
    sheets = {
        "Predictions": predictions,
        "Experiment_Configuration": _config_frame(configuration),
    }
    notes = export_notes(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        description="Predictions from a previously saved model. The model was not retrained for this export.",
    )
    return workbook_from_sheets(sheets, notes)


def _config_frame(configuration: dict) -> pd.DataFrame:
    rows = []
    for key, value in configuration.items():
        rows.append({"item": key, "value": _plain(value)})
    return pd.DataFrame(rows)


def _plain(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return ""
    return str(value)
