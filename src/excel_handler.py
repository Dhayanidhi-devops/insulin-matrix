"""Read and write Excel workbooks without executing macros or modifying sources."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.config import APP_NAME, MAX_ROWS, VERSION
from src.errors import ExcelError

SUPPORTED_SUFFIXES = {".xlsx", ".xlsm", ".csv"}
LEGACY_XLS_MESSAGE = (
    "Legacy .xls workbooks are not supported. Save the file as .xlsx. "
    "Reading .xls would require the optional xlrd package, which is not installed."
)


def _suffix(source, format_hint: str | None = None) -> str:
    if format_hint:
        suffix = format_hint if format_hint.startswith(".") else f".{format_hint}"
        return suffix.lower()
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    raise ExcelError("A file extension is required to read this upload.")


def _reject_legacy(suffix: str) -> None:
    if suffix == ".xls":
        raise ExcelError(LEGACY_XLS_MESSAGE)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ExcelError("Upload a .xlsx, .xlsm, or .csv file. Other formats are not supported.")


def _rewind(source) -> None:
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            return


def _open_excel(source):
    _rewind(source)
    try:
        return pd.ExcelFile(source, engine="openpyxl")
    except ExcelError:
        raise
    except Exception as exc:
        raise ExcelError(
            "The workbook could not be read. It may be malformed, encrypted, or not a valid .xlsx or .xlsm file."
        ) from exc


def list_worksheets(source, format_hint: str | None = None) -> list[str]:
    """Return worksheet names. CSV files report a single label, ``CSV``."""
    suffix = _suffix(source, format_hint)
    _reject_legacy(suffix)
    if suffix == ".csv":
        return ["CSV"]
    workbook = _open_excel(source)
    try:
        names = list(workbook.sheet_names)
    finally:
        workbook.close()
        _rewind(source)
    if not names:
        raise ExcelError("The workbook has no worksheets.")
    return names


def _clean_headers(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    columns = []
    changed = False
    for column in frame.columns:
        text = str(column).strip()
        if text != str(column):
            changed = True
        columns.append(text)
    if changed:
        notes.append("Surrounding spaces were removed from column names. Cell values were not changed.")
    out = frame.copy()
    out.columns = columns
    if out.columns.duplicated().any():
        duplicates = sorted({name for name in out.columns if list(out.columns).count(name) > 1})
        raise ExcelError(
            "The worksheet has duplicate column names ("
            + ", ".join(duplicates)
            + "). Rename them so each column can be mapped unambiguously."
        )
    return out, notes


def _read_csv(source) -> pd.DataFrame:
    try:
        if isinstance(source, (str, Path)):
            sample = Path(source).read_bytes()[:8192]
        elif isinstance(source, bytes):
            sample = source[:8192]
        else:
            sample = source.read(8192)
            source.seek(0)
        text = sample.decode("utf-8-sig", errors="replace")
        first = text.splitlines()[0] if text else ""
        separator = ";" if first.count(";") > first.count(",") else ","
        frame = pd.read_csv(source, sep=separator, encoding="utf-8-sig")
    except ExcelError:
        raise
    except UnicodeDecodeError as exc:
        raise ExcelError("The CSV file must be encoded as UTF-8.") from exc
    except Exception as exc:
        raise ExcelError("The CSV file could not be read.") from exc
    frame.attrs["delimiter"] = separator
    return frame


def read_worksheet(source, sheet_name: str | None = None, format_hint: str | None = None) -> pd.DataFrame:
    """Read one worksheet. Multiple sheets are never combined."""
    suffix = _suffix(source, format_hint)
    _reject_legacy(suffix)
    if suffix == ".csv":
        frame = _read_csv(source)
        chosen = "CSV"
    else:
        names = list_worksheets(source, format_hint)
        if sheet_name is None:
            if len(names) != 1:
                raise ExcelError(
                    "This workbook has multiple worksheets ("
                    + ", ".join(names)
                    + "). Select one worksheet. Sheets are not combined."
                )
            chosen = names[0]
        else:
            if sheet_name not in names:
                raise ExcelError(
                    f"Worksheet {sheet_name!r} was not found. Available worksheets: {', '.join(names)}."
                )
            chosen = sheet_name
        try:
            _rewind(source)
            frame = pd.read_excel(source, sheet_name=chosen, engine="openpyxl")
        except Exception as exc:
            raise ExcelError("The selected worksheet could not be read.") from exc
    frame, notes = _clean_headers(frame)
    if len(frame) > MAX_ROWS:
        raise ExcelError(
            f"This worksheet has {len(frame):} rows, which exceeds the limit of {MAX_ROWS:,}."
        )
    frame.attrs["sheet_name"] = chosen
    frame.attrs["header_notes"] = notes
    return frame


def profile_table(frame: pd.DataFrame) -> dict:
    """Summarise a table for the dataset explorer. Does not modify the table."""
    missing = frame.isna().sum()
    column_rows = []
    for column in frame.columns:
        series = frame[column]
        column_rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(missing[column]),
                "missing_percent": float(missing[column] / len(frame) * 100) if len(frame) else 0.0,
                "unique_values": int(series.nunique(dropna=True)),
            }
        )
    numeric = frame.select_dtypes(include="number")
    describe = numeric.describe().transpose().reset_index().rename(columns={"index": "column"}) if not numeric.empty else pd.DataFrame()
    return {
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()) if len(frame) else 0,
        "columns": pd.DataFrame(column_rows),
        "numeric_summary": describe,
        "header_notes": list(frame.attrs.get("header_notes", [])),
        "delimiter": frame.attrs.get("delimiter"),
        "sheet_name": frame.attrs.get("sheet_name"),
    }


def _safe_sheet_name(name: str, used: set[str]) -> str:
    cleaned = "".join("_" if char in r"[]:*?/\\" else char for char in str(name)).strip() or "Sheet"
    cleaned = cleaned[:31]
    candidate = cleaned
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = cleaned[: 31 - len(suffix)] + suffix
        counter += 1
    used.add(candidate)
    return candidate


def workbook_bytes(sheets: dict[str, pd.DataFrame], notes: dict[str, str] | None = None) -> bytes:
    """Build an .xlsx file. Notes become the Export_Notes worksheet."""
    ordered: dict[str, pd.DataFrame] = {}
    used: set[str] = set()
    if notes is not None:
        note_frame = pd.DataFrame({"item": list(notes.keys()), "value": ["" if value is None else str(value) for value in notes.values()]})
        ordered[_safe_sheet_name("Export_Notes", used)] = note_frame
    for name, frame in sheets.items():
        table = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        if table.shape[1] == 0:
            table = pd.DataFrame({"message": ["This worksheet has no columns."]})
        ordered[_safe_sheet_name(name, used)] = table
    buffer = BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, table in ordered.items():
                table.to_excel(writer, sheet_name=name, index=False)
    except Exception as exc:
        raise ExcelError("The Excel export could not be created.") from exc
    return buffer.getvalue()


def export_notes(
    *,
    dataset_id: str | None,
    dataset_type: str | None,
    description: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Standard export metadata. Synthetic results are labeled explicitly."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    notes = {
        "application": APP_NAME,
        "version": VERSION,
        "generated_at_utc": timestamp,
        "dataset_id": dataset_id or "",
        "dataset_type": dataset_type or "",
        "description": description,
        "disclaimer": (
            "Research output only. Not a diagnosis, treatment recommendation, or insulin dose."
        ),
    }
    if dataset_type == "synthetic":
        notes["synthetic_data_label"] = (
            "SYNTHETIC DEMONSTRATION DATA. These rows and any model results derived from them "
            "are not clinical measurements and are not clinical evidence."
        )
    else:
        notes["synthetic_data_label"] = "Not labeled as synthetic. Treat this as a user-supplied research dataset."
    if extra:
        notes.update(extra)
    return notes
