"""Unit tests for target derivation and FastAPI health."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.config import DEFAULT_SAMPLE_PATH, TARGET_COLUMN
from app.main import app
from app.ml.features import derive_diabetes_category, prepare_modeling_frame, validate_features


def test_derive_categories_cover_four_labels():
    frame = pd.read_excel(DEFAULT_SAMPLE_PATH)
    labels = derive_diabetes_category(frame)
    assert set(labels.unique()) <= {
        "No Diabetes",
        "Prediabetes",
        "Type 1 Diabetes",
        "Type 2 Diabetes",
    }
    assert (labels == "Type 1 Diabetes").sum() >= 1


def test_leakage_features_rejected():
    frame = prepare_modeling_frame(pd.read_excel(DEFAULT_SAMPLE_PATH))
    try:
        validate_features(["Record_ID", "Age_years"], frame)
        assert False, "expected leakage rejection"
    except ValueError as exc:
        assert "leakage" in str(exc).lower() or "Refusing" in str(exc)


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "disclaimer" in body
