from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from src.excel_handler import ExcelError, list_worksheets, profile_table, read_worksheet, workbook_bytes
from src.exports import indices_workbook, validation_workbook
from src.indices import calculate_indices
from src.synthetic_data import generate_synthetic_dataset


def _write(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def test_reads_selected_worksheet_without_combining(tmp_path):
    path = tmp_path / "book.xlsx"
    _write(
        path,
        {
            "Alpha": pd.DataFrame({"a": [1, 2]}),
            "Beta": pd.DataFrame({"b": [9, 8, 7]}),
        },
    )
    before = path.read_bytes()
    assert list_worksheets(path) == ["Alpha", "Beta"]
    with pytest.raises(ExcelError, match="not combined"):
        read_worksheet(path)
    beta = read_worksheet(path, "Beta")
    assert list(beta.columns) == ["b"]
    assert len(beta) == 3
    assert path.read_bytes() == before


def test_empty_and_malformed_workbooks(tmp_path):
    empty = tmp_path / "empty.xlsx"
    _write(empty, {"Sheet1": pd.DataFrame()})
    frame = read_worksheet(empty, "Sheet1")
    assert len(frame) == 0

    malformed = tmp_path / "bad.xlsx"
    malformed.write_bytes(b"this is not a workbook")
    with pytest.raises(ExcelError, match="could not be read"):
        list_worksheets(malformed)


def test_legacy_xls_is_rejected(tmp_path):
    path = tmp_path / "old.xls"
    path.write_bytes(b"not really xls")
    with pytest.raises(ExcelError, match="xlrd"):
        list_worksheets(path)


def test_csv_round_trip_and_profile(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"glucose": [5, 6], "note": ["a", "a"]}).to_csv(path, index=False)
    frame = read_worksheet(path)
    profile = profile_table(frame)
    assert profile["n_rows"] == 2
    assert profile["duplicate_rows"] == 0
    assert "glucose" in set(profile["columns"]["column"])


def test_export_contains_relevant_sheets_and_synthetic_label():
    frame = generate_synthetic_dataset().rename(
        columns={
            "fasting_glucose_mmol": "fasting_glucose",
            "fasting_insulin_uU_ml": "fasting_insulin",
            "glucose_30min_mmol": "glucose_30",
            "insulin_30min_uU_ml": "insulin_30",
            "glucose_60min_mmol": "glucose_60",
            "insulin_60min_uU_ml": "insulin_60",
            "glucose_90min_mmol": "glucose_90",
            "insulin_90min_uU_ml": "insulin_90",
            "glucose_120min_mmol": "glucose_120",
            "insulin_120min_uU_ml": "insulin_120",
        }
    )
    indices, summary = calculate_indices(frame, {"glucose": "mmol/L", "insulin": "µU/mL"})
    payload = indices_workbook(indices.head(5), summary, dataset_id="ds_test", dataset_type="synthetic")
    sheets = pd.ExcelFile(BytesIO(payload)).sheet_names
    assert "Export_Notes" in sheets
    assert "Calculated_Indices" in sheets
    assert "Model_Metrics" not in sheets
    notes = pd.read_excel(BytesIO(payload), sheet_name="Export_Notes")
    text = " ".join(notes["value"].astype(str))
    assert "SYNTHETIC" in text
    assert "Insulin Matrix" in text

    validation = validation_workbook(
        pd.DataFrame([{"severity": "warning", "code": "MISSING_VALUES", "message": "example"}]),
        {"columns": pd.DataFrame({"column": ["a"]})},
        dataset_id="ds_test",
        dataset_type="real",
    )
    validation_sheets = pd.ExcelFile(BytesIO(validation)).sheet_names
    assert "Validation_Report" in validation_sheets
    assert "Predictions" not in validation_sheets


def test_workbook_bytes_round_trip():
    payload = workbook_bytes({"Dataset_Summary": pd.DataFrame({"rows": [3]})}, {"application": "Insulin Matrix"})
    loaded = pd.read_excel(BytesIO(payload), sheet_name=None)
    assert set(loaded) == {"Export_Notes", "Dataset_Summary"}
    assert loaded["Dataset_Summary"].loc[0, "rows"] == 3
