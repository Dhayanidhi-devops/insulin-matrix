"""SQLite persistence for datasets, training runs, and prediction history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import database_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    display_name TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL DEFAULT 1,
    n_rows INTEGER,
    n_columns INTEGER,
    storage_relpath TEXT,
    class_counts_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset_id TEXT,
    model_name TEXT NOT NULL,
    model_family TEXT NOT NULL,
    features_json TEXT NOT NULL,
    hyperparameters_json TEXT,
    metrics_json TEXT NOT NULL,
    class_labels_json TEXT,
    artifact_relpath TEXT,
    n_train INTEGER,
    n_test INTEGER,
    seed INTEGER,
    test_size REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    run_id TEXT,
    model_name TEXT,
    input_json TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    probabilities_json TEXT,
    explanation_json TEXT,
    source TEXT NOT NULL,
    is_synthetic_context INTEGER NOT NULL DEFAULT 1,
    disclaimer TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON training_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions (created_at DESC);
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
        conn = sqlite3.connect(self.path, timeout=60)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

    def upsert_dataset(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, created_at, display_name, dataset_type, is_synthetic,
                    n_rows, n_columns, storage_relpath, class_counts_json, notes
                ) VALUES (
                    :dataset_id, :created_at, :display_name, :dataset_type, :is_synthetic,
                    :n_rows, :n_columns, :storage_relpath, :class_counts_json, :notes
                )
                ON CONFLICT(dataset_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    dataset_type=excluded.dataset_type,
                    is_synthetic=excluded.is_synthetic,
                    n_rows=excluded.n_rows,
                    n_columns=excluded.n_columns,
                    storage_relpath=excluded.storage_relpath,
                    class_counts_json=excluded.class_counts_json,
                    notes=excluded.notes
                """,
                record,
            )

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        return self._row_to_dict(row)

    def insert_training_run(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs (
                    run_id, created_at, dataset_id, model_name, model_family,
                    features_json, hyperparameters_json, metrics_json, class_labels_json,
                    artifact_relpath, n_train, n_test, seed, test_size, notes
                ) VALUES (
                    :run_id, :created_at, :dataset_id, :model_name, :model_family,
                    :features_json, :hyperparameters_json, :metrics_json, :class_labels_json,
                    :artifact_relpath, :n_train, :n_test, :seed, :test_size, :notes
                )
                """,
                record,
            )

    def list_training_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_training_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row)

    def delete_training_run(self, run_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM training_runs WHERE run_id = ?", (run_id,)
            )
            return cur.rowcount > 0

    def insert_prediction(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    prediction_id, created_at, run_id, model_name, input_json,
                    predicted_class, probabilities_json, explanation_json,
                    source, is_synthetic_context, disclaimer
                ) VALUES (
                    :prediction_id, :created_at, :run_id, :model_name, :input_json,
                    :predicted_class, :probabilities_json, :explanation_json,
                    :source, :is_synthetic_context, :disclaimer
                )
                """,
                record,
            )

    def list_predictions(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def delete_prediction(self, prediction_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            )
            return cur.rowcount > 0

    def clear_predictions(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM predictions")
            return cur.rowcount

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value_json) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (key, json.dumps(value)),
            )

    def overview_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            datasets = conn.execute("SELECT COUNT(*) AS c FROM datasets").fetchone()["c"]
            runs = conn.execute("SELECT COUNT(*) AS c FROM training_runs").fetchone()["c"]
            preds = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        return {
            "datasets": int(datasets),
            "training_runs": int(runs),
            "predictions": int(preds),
        }
