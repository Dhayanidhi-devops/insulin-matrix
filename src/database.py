"""SQLite metadata for datasets and experiment history.

The database stores labels, configuration, and metrics. It does not store
spreadsheet cell values. Uploaded workbooks and model files live beside it
and are removed when the matching record is deleted.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import artifact_dir, database_path, safe_under_home
from src.errors import StorageError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    display_name TEXT NOT NULL,
    dataset_type TEXT NOT NULL CHECK (dataset_type IN ('real', 'synthetic')),
    content_hash TEXT,
    n_rows INTEGER,
    n_participants INTEGER,
    n_columns INTEGER,
    validation_status TEXT,
    n_index_values INTEGER DEFAULT 0,
    sheet_name TEXT,
    storage_relpath TEXT,
    processed_relpath TEXT,
    mapping_json TEXT,
    units_json TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset_id TEXT,
    dataset_label TEXT,
    dataset_type TEXT NOT NULL,
    task TEXT NOT NULL,
    target TEXT NOT NULL,
    features_json TEXT NOT NULL,
    model_name TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    n_train_participants INTEGER,
    n_test_participants INTEGER,
    n_train_rows INTEGER,
    n_test_rows INTEGER,
    seed INTEGER,
    test_size REAL,
    artifact_id TEXT,
    split_mode TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    created_at TEXT NOT NULL,
    relpath TEXT NOT NULL,
    kind TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments (created_at DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class Database:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def upsert_dataset(self, record: dict) -> None:
        columns = [
            "dataset_id",
            "created_at",
            "display_name",
            "dataset_type",
            "content_hash",
            "n_rows",
            "n_participants",
            "n_columns",
            "validation_status",
            "n_index_values",
            "sheet_name",
            "storage_relpath",
            "processed_relpath",
            "mapping_json",
            "units_json",
        ]
        payload = {key: record.get(key) for key in columns}
        payload["created_at"] = payload["created_at"] or utc_now()
        payload["n_index_values"] = payload["n_index_values"] or 0
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "dataset_id")
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO datasets ({", ".join(columns)})
                VALUES ({", ".join(":" + column for column in columns)})
                ON CONFLICT(dataset_id) DO UPDATE SET {assignments}
                """,
                payload,
            )

    def update_dataset(self, dataset_id: str, **fields) -> None:
        if not fields:
            return
        allowed = {
            "display_name",
            "validation_status",
            "n_index_values",
            "n_rows",
            "n_participants",
            "n_columns",
            "mapping_json",
            "units_json",
            "processed_relpath",
            "sheet_name",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StorageError("Refusing to write unsupported dataset fields.")
        clause = ", ".join(f"{key}=:{key}" for key in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE datasets SET {clause} WHERE dataset_id=:dataset_id", {**fields, "dataset_id": dataset_id})

    def find_dataset_by_hash(self, content_hash: str, dataset_type: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE content_hash=? AND dataset_type=? ORDER BY created_at DESC LIMIT 1",
                (content_hash, dataset_type),
            ).fetchone()
        return dict(row) if row else None

    def list_datasets(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    def delete_dataset(self, dataset_id: str) -> dict:
        """Delete the dataset row, its upload, and its processed copy.

        Experiment history is kept. Metrics stay available, and the stored
        dataset label on each experiment does not depend on this row.
        """
        record = self.get_dataset(dataset_id)
        if record is None:
            raise StorageError("That dataset is not in the local registry.")
        removed = []
        for key in ("storage_relpath", "processed_relpath"):
            relative = record.get(key)
            if relative:
                removed.append(str(_unlink_relative(relative)))
        with self._connect() as conn:
            conn.execute("DELETE FROM datasets WHERE dataset_id=?", (dataset_id,))
        return {"dataset_id": dataset_id, "removed_files": removed}

    def insert_experiment(self, record: dict) -> None:
        columns = [
            "experiment_id",
            "created_at",
            "dataset_id",
            "dataset_label",
            "dataset_type",
            "task",
            "target",
            "features_json",
            "model_name",
            "hyperparameters_json",
            "metrics_json",
            "n_train_participants",
            "n_test_participants",
            "n_train_rows",
            "n_test_rows",
            "seed",
            "test_size",
            "artifact_id",
            "split_mode",
            "notes",
        ]
        payload = {key: record.get(key) for key in columns}
        payload["created_at"] = payload["created_at"] or utc_now()
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO experiments ({', '.join(columns)}) VALUES ({', '.join(':' + column for column in columns)})",
                payload,
            )

    def insert_artifact(self, artifact_id: str, experiment_id: str, relpath: str, kind: str = "model") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (artifact_id, experiment_id, created_at, relpath, kind) VALUES (?, ?, ?, ?, ?)",
                (artifact_id, experiment_id, utc_now(), relpath, kind),
            )

    def get_experiment(self, experiment_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        return dict(row) if row else None

    def get_artifact(self, artifact_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return dict(row) if row else None

    def list_experiments(
        self,
        *,
        dataset_type: str | None = None,
        task: str | None = None,
        model_name: str | None = None,
        text: str | None = None,
    ) -> list[dict]:
        clauses = []
        params: list[str] = []
        if dataset_type:
            clauses.append("dataset_type=?")
            params.append(dataset_type)
        if task:
            clauses.append("task=?")
            params.append(task)
        if model_name:
            clauses.append("model_name=?")
            params.append(model_name)
        if text:
            clauses.append("(target LIKE ? OR model_name LIKE ? OR dataset_label LIKE ? OR experiment_id LIKE ?)")
            like = f"%{text}%"
            params.extend([like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM experiments {where} ORDER BY created_at DESC", params).fetchall()
        return [dict(row) for row in rows]

    def delete_experiment(self, experiment_id: str) -> dict:
        """Delete an experiment and its model file together."""
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise StorageError("That experiment is not in the history.")
        artifact = self.get_artifact(experiment["artifact_id"]) if experiment.get("artifact_id") else None
        removed = None
        if artifact and artifact.get("relpath"):
            removed = str(_unlink_relative(artifact["relpath"]))
        with self._connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE experiment_id=?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE experiment_id=?", (experiment_id,))
        return {"experiment_id": experiment_id, "removed_file": removed, "artifact_missing": artifact is None}

    def summary_counts(self) -> dict:
        with self._connect() as conn:
            datasets = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(n_rows), 0) AS rows, COALESCE(SUM(n_participants), 0) AS participants, COALESCE(SUM(n_index_values), 0) AS indices FROM datasets").fetchone()
            unknown_participants = conn.execute("SELECT COUNT(*) AS n FROM datasets WHERE n_participants IS NULL").fetchone()
            models = conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()
            latest = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 1").fetchone()
        return {
            "datasets": int(datasets["n"]),
            "observations": int(datasets["rows"]),
            "participants": int(datasets["participants"]),
            "datasets_without_participant_count": int(unknown_participants["n"]),
            "index_values": int(datasets["indices"]),
            "models": int(models["n"]),
            "latest_experiment": dict(latest) if latest else None,
        }


def _unlink_relative(relative: str) -> Path:
    path = safe_under_home(relative)
    if path.exists() and not path.is_file():
        raise StorageError("Refusing to delete a storage entry that is not a file.")
    if path.is_file():
        path.unlink()
    return path


def dump_json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def load_json(value: str | None, fallback):
    if not value:
        return fallback
    return json.loads(value)
