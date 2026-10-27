"""Load worksheets into a dataset record and register local metadata."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pandas as pd

from src.config import (
    MAX_UPLOAD_BYTES,
    processed_dir,
    safe_under_home,
    upload_dir,
)
from src.database import Database, dump_json, utc_now
from src.errors import ExcelError, StorageError
from src.excel_handler import list_worksheets, read_worksheet
from src.synthetic_data import (
    SYNTHETIC_DATASET_NAME,
    SYNTHETIC_MAPPING,
    generate_synthetic_dataset,
)
from src.validation import validate_dataset

RELATIVE_UPLOADS = "data/uploads"
RELATIVE_PROCESSED = "data/processed"


def content_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def save_upload(original_name: str, data: bytes) -> tuple[Path, str]:
    if not data:
        raise ExcelError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExcelError("The file exceeds the 50 MB limit.")
    suffix = Path(original_name).suffix.lower()
    if suffix == ".xls":
        raise ExcelError(
            "Legacy .xls workbooks are not supported. Save the file as .xlsx. "
            "Reading .xls would require the optional xlrd package, which is not installed."
        )
    if suffix not in {".xlsx", ".xlsm", ".csv"}:
        raise ExcelError("Upload a .xlsx, .xlsm, or .csv file.")
    destination = upload_dir() / f"{uuid.uuid4().hex}{suffix}"
    destination.write_bytes(data)
    relative = f"{RELATIVE_UPLOADS}/{destination.name}"
    return destination, relative


def worksheet_names(relative_path: str) -> list[str]:
    return list_worksheets(safe_under_home(relative_path))


def read_saved_worksheet(relative_path: str, sheet_name: str | None) -> pd.DataFrame:
    return read_worksheet(safe_under_home(relative_path), sheet_name)


def _participant_count(frame: pd.DataFrame, mapping: dict | None) -> int | None:
    if mapping and mapping.get("participant_id") and mapping["participant_id"] in frame.columns:
        values = frame[mapping["participant_id"]].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        return int(values.nunique())
    if "participant_id" in frame.columns:
        values = frame["participant_id"].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        return int(values.nunique()) if len(values) else None
    return None


def register_dataset(
    database: Database,
    frame: pd.DataFrame,
    *,
    display_name: str,
    dataset_type: str,
    sheet_name: str,
    storage_relpath: str | None,
    mapping: dict | None = None,
    units: dict | None = None,
) -> dict:
    """Insert a dataset row, reusing the identifier when the same table was stored before."""
    digest = content_hash(frame)
    existing = database.find_dataset_by_hash(digest, dataset_type)
    dataset_id = existing["dataset_id"] if existing else new_id("ds")
    record = {
        "dataset_id": dataset_id,
        "created_at": existing["created_at"] if existing else utc_now(),
        "display_name": display_name[:120],
        "dataset_type": dataset_type,
        "content_hash": digest,
        "n_rows": int(len(frame)),
        "n_participants": _participant_count(frame, mapping),
        "n_columns": int(frame.shape[1]),
        "validation_status": existing["validation_status"] if existing else "not_validated",
        "n_index_values": existing["n_index_values"] if existing else 0,
        "sheet_name": sheet_name,
        "storage_relpath": storage_relpath or (existing["storage_relpath"] if existing else None),
        "processed_relpath": existing["processed_relpath"] if existing else None,
        "mapping_json": dump_json(mapping) if mapping else (existing["mapping_json"] if existing else None),
        "units_json": dump_json(units) if units else (existing["units_json"] if existing else None),
    }
    if existing and storage_relpath and existing.get("storage_relpath") and existing["storage_relpath"] != storage_relpath:
        _remove_if_present(storage_relpath)
        record["storage_relpath"] = existing["storage_relpath"]
    database.upsert_dataset(record)
    active = {
        "dataset_id": dataset_id,
        "display_name": record["display_name"],
        "dataset_type": dataset_type,
        "sheet_name": sheet_name,
        "content_hash": digest,
        "storage_relpath": record["storage_relpath"],
        "frame": frame,
        "mapping": mapping,
        "extra_covariates": [],
        "units": units or {"glucose": None, "insulin": None},
        "analysis_frame": None,
        "target_column": None,
        "layout": None,
        "validation": None,
        "indices": None,
        "index_summary": None,
        "index_value_count": int(record["n_index_values"] or 0),
        "reused_existing": existing is not None,
    }
    return active


def store_processed_copy(dataset_id: str, frame: pd.DataFrame) -> str:
    path = processed_dir() / f"{dataset_id}_analysis.csv"
    frame.to_csv(path, index=False)
    return f"{RELATIVE_PROCESSED}/{path.name}"


def apply_validation(database: Database, active: dict) -> dict:
    analysis, report = validate_dataset(
        active["frame"],
        active["mapping"] or {},
        active["units"],
        active.get("extra_covariates") or [],
        active["content_hash"],
    )
    active = dict(active)
    active["analysis_frame"] = analysis
    active["validation"] = report
    active["target_column"] = report.target_column
    active["layout"] = report.layout
    active["indices"] = None
    active["index_summary"] = None
    processed = store_processed_copy(active["dataset_id"], analysis if len(analysis) else active["frame"])
    database.update_dataset(
        active["dataset_id"],
        validation_status=report.status,
        mapping_json=dump_json(active["mapping"]),
        units_json=dump_json(active["units"]),
        processed_relpath=processed,
        n_participants=_participant_count(active["frame"], active["mapping"]),
        n_rows=int(len(active["frame"])),
        n_columns=int(active["frame"].shape[1]),
    )
    return active


def load_synthetic(database: Database, seed: int = 42) -> dict:
    frame = generate_synthetic_dataset(seed=seed)
    buffer_path = upload_dir() / f"{uuid.uuid4().hex}.xlsx"
    frame.to_excel(buffer_path, index=False, sheet_name="Synthetic_OGTT")
    relative = f"{RELATIVE_UPLOADS}/{buffer_path.name}"
    active = register_dataset(
        database,
        frame,
        display_name=SYNTHETIC_DATASET_NAME,
        dataset_type="synthetic",
        sheet_name="Synthetic_OGTT",
        storage_relpath=relative,
        mapping=dict(SYNTHETIC_MAPPING),
        units={"glucose": "mmol/L", "insulin": "µU/mL"},
    )
    active["mapping"] = dict(SYNTHETIC_MAPPING)
    active["units"] = {"glucose": "mmol/L", "insulin": "µU/mL"}
    active["extra_covariates"] = []
    return apply_validation(database, active)


def _remove_if_present(relative: str) -> None:
    try:
        path = safe_under_home(relative)
    except StorageError:
        return
    if path.is_file():
        path.unlink()
