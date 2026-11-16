"""Individual / batch prediction and lightweight explainability."""

from __future__ import annotations

import json
import uuid
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    ASSESSMENT_FIELDS,
    DEFAULT_FEATURES,
    MEDICAL_DISCLAIMER,
    SYNTHETIC_DATA_LABEL,
)
from app.database import utc_now
from app.ml.ann import TrainedANNBundle
from app.ml.models import TrainedModelBundle
from app.state import STATE


def assessment_schema() -> dict:
    return {
        "fields": ASSESSMENT_FIELDS,
        "disclaimer": MEDICAL_DISCLAIMER,
        "synthetic_label": SYNTHETIC_DATA_LABEL,
        "active_model": STATE.active_model_name,
        "trained_models": list(STATE.trained_models.keys()),
    }


def _bundle_predict(bundle, frame: pd.DataFrame) -> dict[str, Any]:
    if isinstance(bundle, (TrainedModelBundle, TrainedANNBundle)):
        return bundle.predict_frame(frame)
    raise ValueError("Unsupported model bundle type.")


def _explanation_for_row(
    bundle, frame: pd.DataFrame, predicted_class: str, probabilities: dict | None
) -> dict[str, Any]:
    """Heuristic explanation from permutation importance + local feature values."""
    importance = []
    metrics = getattr(bundle, "metrics", {}) or {}
    for item in metrics.get("feature_importance") or []:
        importance.append(item)

    row = frame.iloc[0].to_dict()
    top_features = []
    for item in importance[:8]:
        feat = item["feature"]
        top_features.append(
            {
                "feature": feat,
                "importance": item.get("importance"),
                "value": row.get(feat),
            }
        )

    # Clinical context notes (not diagnostic rules).
    notes = []
    glucose = row.get("Fasting_Glucose_mg_dL")
    hba1c = row.get("HbA1c_percent")
    cpep = row.get("C_Peptide_ng_mL")
    try:
        if glucose is not None and float(glucose) >= 126:
            notes.append("Fasting glucose is in a diabetes-range screening band (≥126 mg/dL).")
        elif glucose is not None and float(glucose) >= 100:
            notes.append("Fasting glucose is in a prediabetes-range screening band (100–125 mg/dL).")
    except Exception:
        pass
    try:
        if hba1c is not None and float(hba1c) >= 6.5:
            notes.append("HbA1c is in a diabetes-range screening band (≥6.5%).")
        elif hba1c is not None and float(hba1c) >= 5.7:
            notes.append("HbA1c is in a prediabetes-range screening band (5.7–6.4%).")
    except Exception:
        pass
    try:
        if cpep is not None and float(cpep) < 0.6 and predicted_class == "Type 1 Diabetes":
            notes.append("Low C-peptide can accompany Type 1–like synthetic profiles.")
    except Exception:
        pass

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "top_features": top_features,
        "clinical_screening_notes": notes,
        "method": "permutation_importance_plus_value_context",
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def _persist_prediction(
    *,
    model_name: str,
    inputs: dict,
    predicted_class: str,
    probabilities: dict | None,
    explanation: dict | None,
    source: str,
) -> str:
    prediction_id = uuid.uuid4().hex[:12]
    run_id = None
    # Best-effort link to latest matching training run.
    for run in STATE.db.list_training_runs(limit=50):
        if run["model_name"] == model_name:
            run_id = run["run_id"]
            break
    STATE.db.insert_prediction(
        {
            "prediction_id": prediction_id,
            "created_at": utc_now(),
            "run_id": run_id,
            "model_name": model_name,
            "input_json": json.dumps(inputs),
            "predicted_class": predicted_class,
            "probabilities_json": json.dumps(probabilities) if probabilities else None,
            "explanation_json": json.dumps(explanation) if explanation else None,
            "source": source,
            "is_synthetic_context": 1 if STATE.is_synthetic else 0,
            "disclaimer": MEDICAL_DISCLAIMER,
        }
    )
    return prediction_id


def predict_one(
    payload: dict[str, Any],
    *,
    model_name: str | None = None,
) -> dict:
    bundle = STATE.get_model(model_name)
    name = model_name or STATE.active_model_name
    # Fill BMI if height/weight provided.
    data = dict(payload)
    if (
        ("BMI_kg_m2" not in data or data.get("BMI_kg_m2") in (None, ""))
        and data.get("Height_cm")
        and data.get("Weight_kg")
    ):
        height_m = float(data["Height_cm"]) / 100.0
        if height_m > 0:
            data["BMI_kg_m2"] = round(float(data["Weight_kg"]) / (height_m**2), 2)

    frame = pd.DataFrame([data])
    # Ensure all model features exist.
    for feat in bundle.features:
        if feat not in frame.columns:
            frame[feat] = np.nan

    result = _bundle_predict(bundle, frame)
    predicted = result["predictions"][0]
    probs = result["probabilities"][0] if result["probabilities"] else None
    explanation = _explanation_for_row(bundle, frame, predicted, probs)
    prediction_id = _persist_prediction(
        model_name=name or bundle.model_name,
        inputs=data,
        predicted_class=predicted,
        probabilities=probs,
        explanation=explanation,
        source="individual",
    )
    return {
        "prediction_id": prediction_id,
        "model_name": name,
        "predicted_class": predicted,
        "probabilities": probs,
        "explanation": explanation,
        "is_synthetic_context": STATE.is_synthetic,
        "synthetic_label": SYNTHETIC_DATA_LABEL if STATE.is_synthetic else None,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def predict_batch_frame(
    frame: pd.DataFrame,
    *,
    model_name: str | None = None,
) -> dict:
    bundle = STATE.get_model(model_name)
    name = model_name or STATE.active_model_name
    working = frame.copy()
    for feat in bundle.features:
        if feat not in working.columns:
            working[feat] = np.nan
    result = _bundle_predict(bundle, working)
    rows = []
    for i, pred in enumerate(result["predictions"]):
        probs = result["probabilities"][i] if result["probabilities"] else None
        inputs = working.iloc[i][bundle.features].to_dict()
        prediction_id = _persist_prediction(
            model_name=name or bundle.model_name,
            inputs={k: (None if pd.isna(v) else v) for k, v in inputs.items()},
            predicted_class=pred,
            probabilities=probs,
            explanation=None,
            source="batch",
        )
        row = {
            "prediction_id": prediction_id,
            "predicted_class": pred,
            "probabilities": probs,
        }
        rows.append(row)

    summary = (
        pd.Series(result["predictions"]).value_counts().astype(int).to_dict()
    )
    out = working.copy()
    out["predicted_class"] = result["predictions"]
    if result["probabilities"]:
        for label in bundle.class_labels:
            out[f"proba_{label}"] = [
                (p or {}).get(label) for p in result["probabilities"]
            ]

    return {
        "model_name": name,
        "n_rows": len(rows),
        "predictions": rows,
        "class_summary": summary,
        "preview": out.head(50).fillna("").astype(object).where(pd.notnull(out.head(50)), None).to_dict(orient="records"),
        "result_frame": out,
        "is_synthetic_context": STATE.is_synthetic,
        "synthetic_label": SYNTHETIC_DATA_LABEL if STATE.is_synthetic else None,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def predict_batch_upload(
    filename: str,
    content: bytes,
    *,
    model_name: str | None = None,
) -> dict:
    suffix = filename.lower().rsplit(".", 1)[-1]
    bio = BytesIO(content)
    if suffix in {"xlsx", "xlsm"}:
        frame = pd.read_excel(bio)
    elif suffix == "csv":
        frame = pd.read_csv(bio)
    else:
        raise ValueError("Batch upload must be .xlsx, .xlsm, or .csv")
    if frame.empty:
        raise ValueError("Batch file is empty.")
    return predict_batch_frame(frame, model_name=model_name)


def history(limit: int = 100) -> dict:
    rows = STATE.db.list_predictions(limit=limit)
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "input": json.loads(row["input_json"]) if row.get("input_json") else {},
                "probabilities": json.loads(row["probabilities_json"])
                if row.get("probabilities_json")
                else None,
                "explanation": json.loads(row["explanation_json"])
                if row.get("explanation_json")
                else None,
                "disclaimer": row.get("disclaimer") or MEDICAL_DISCLAIMER,
            }
        )
    return {
        "items": parsed,
        "count": len(parsed),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def get_prediction(prediction_id: str) -> dict:
    row = STATE.db.get_prediction(prediction_id)
    if not row:
        raise ValueError("Prediction not found.")
    return {
        **row,
        "input": json.loads(row["input_json"]) if row.get("input_json") else {},
        "probabilities": json.loads(row["probabilities_json"])
        if row.get("probabilities_json")
        else None,
        "explanation": json.loads(row["explanation_json"])
        if row.get("explanation_json")
        else None,
        "disclaimer": row.get("disclaimer") or MEDICAL_DISCLAIMER,
    }
