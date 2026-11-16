"""Dataset load, upload, and profiling."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from app.config import (
    DEFAULT_SAMPLE_PATH,
    MEDICAL_DISCLAIMER,
    SYNTHETIC_DATA_LABEL,
    TARGET_COLUMN,
    upload_dir,
)
from app.database import utc_now
from app.ml.features import (
    class_distribution,
    dataset_profile,
    prepare_modeling_frame,
    rare_class_warnings,
)
from app.state import STATE


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Supported uploads: .xlsx, .xlsm, .csv")


def _store_dataset(
    frame: pd.DataFrame,
    *,
    display_name: str,
    dataset_type: str,
    is_synthetic: bool,
    source_path: Path | None = None,
    notes: str = "",
) -> dict:
    prepared = prepare_modeling_frame(frame)
    dataset_id = uuid.uuid4().hex[:12]
    dest = upload_dir() / f"{dataset_id}_{Path(display_name).name}"
    if source_path and source_path.exists() and source_path.suffix.lower() in {
        ".xlsx",
        ".xlsm",
        ".csv",
    }:
        shutil.copy2(source_path, dest)
    else:
        dest = upload_dir() / f"{dataset_id}.xlsx"
        prepared.to_excel(dest, index=False)

    counts = class_distribution(prepared)
    relpath = str(dest.relative_to(dest.parents[1])) if False else str(dest)
    # Store path relative to project when possible.
    try:
        from app.config import storage_home

        relpath = str(dest.relative_to(storage_home()))
    except Exception:
        relpath = str(dest)

    record = {
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "display_name": display_name,
        "dataset_type": dataset_type,
        "is_synthetic": 1 if is_synthetic else 0,
        "n_rows": int(len(prepared)),
        "n_columns": int(prepared.shape[1]),
        "storage_relpath": relpath,
        "class_counts_json": json.dumps(counts),
        "notes": notes or (SYNTHETIC_DATA_LABEL if is_synthetic else ""),
    }
    STATE.db.upsert_dataset(record)
    STATE.frame = prepared
    STATE.dataset_id = dataset_id
    STATE.dataset_name = display_name
    STATE.is_synthetic = is_synthetic
    STATE.preprocess_report = None
    STATE.trained_models = {}
    STATE.active_model_name = None

    profile = dataset_profile(prepared)
    return {
        "dataset_id": dataset_id,
        "display_name": display_name,
        "dataset_type": dataset_type,
        "is_synthetic": is_synthetic,
        "synthetic_label": SYNTHETIC_DATA_LABEL if is_synthetic else None,
        "disclaimer": MEDICAL_DISCLAIMER,
        "target_column": TARGET_COLUMN,
        "class_counts": counts,
        "rare_class_warnings": rare_class_warnings(counts),
        "profile": profile,
        "preview": prepared.head(25).fillna("").astype(str).to_dict(orient="records"),
    }


def load_default_sample() -> dict:
    path = DEFAULT_SAMPLE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Default synthetic dataset not found at {path}. "
            "Place Insulin_Matrix_Synthetic_Dataset_10000_Updated.xlsx under data/sample/."
        )
    frame = _read_table(path)
    return _store_dataset(
        frame,
        display_name=path.name,
        dataset_type="synthetic",
        is_synthetic=True,
        source_path=path,
        notes=SYNTHETIC_DATA_LABEL,
    )


def upload_dataset(filename: str, content: bytes, *, mark_synthetic: bool = False) -> dict:
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("Upload exceeds 50 MB limit.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".xlsm", ".csv"}:
        raise ValueError("Supported uploads: .xlsx, .xlsm, .csv")
    digest = hashlib.sha256(content).hexdigest()[:10]
    temp = upload_dir() / f"upload_{digest}{suffix}"
    temp.write_bytes(content)
    frame = _read_table(temp)
    if len(frame) > 250_000:
        raise ValueError("Upload exceeds 250,000 row limit.")
    return _store_dataset(
        frame,
        display_name=filename,
        dataset_type="synthetic" if mark_synthetic else "uploaded",
        is_synthetic=mark_synthetic,
        source_path=temp,
        notes=SYNTHETIC_DATA_LABEL if mark_synthetic else "User-uploaded dataset",
    )


def current_dataset_status() -> dict:
    if not STATE.has_dataset():
        return {
            "loaded": False,
            "sample_available": DEFAULT_SAMPLE_PATH.exists(),
            "sample_path": str(DEFAULT_SAMPLE_PATH),
            "disclaimer": MEDICAL_DISCLAIMER,
        }
    frame = STATE.require_frame()
    profile = dataset_profile(frame)
    return {
        "loaded": True,
        "dataset_id": STATE.dataset_id,
        "display_name": STATE.dataset_name,
        "is_synthetic": STATE.is_synthetic,
        "synthetic_label": SYNTHETIC_DATA_LABEL if STATE.is_synthetic else None,
        "disclaimer": MEDICAL_DISCLAIMER,
        "target_column": TARGET_COLUMN,
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "class_counts": profile["class_counts"],
        "rare_class_warnings": profile["rare_class_warnings"],
        "features": profile["features"],
        "sample_available": DEFAULT_SAMPLE_PATH.exists(),
    }


def dataset_preview(limit: int = 50) -> dict:
    frame = STATE.require_frame()
    limit = max(1, min(limit, 200))
    return {
        "columns": list(frame.columns),
        "n_rows": int(len(frame)),
        "rows": frame.head(limit).where(pd.notnull(frame.head(limit)), None).to_dict(orient="records"),
        "is_synthetic": STATE.is_synthetic,
        "synthetic_label": SYNTHETIC_DATA_LABEL if STATE.is_synthetic else None,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def dataset_stats() -> dict:
    frame = STATE.require_frame()
    profile = dataset_profile(frame)
    return {
        "dataset_id": STATE.dataset_id,
        "display_name": STATE.dataset_name,
        "is_synthetic": STATE.is_synthetic,
        "synthetic_label": SYNTHETIC_DATA_LABEL if STATE.is_synthetic else None,
        "disclaimer": MEDICAL_DISCLAIMER,
        "profile": profile,
    }
