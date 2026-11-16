"""Primary REST API for Insulin Matrix AI."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.config import (
    APP_NAME,
    APP_VERSION,
    ASSESSMENT_FIELDS,
    MEDICAL_DISCLAIMER,
    SYNTHETIC_DATA_LABEL,
)
from app.ml.ann import tensorflow_status
from app.ml.features import class_distribution
from app.schemas import (
    ActiveModelRequest,
    ANNTrainRequest,
    PdfReportRequest,
    PredictRequest,
    PreprocessRequest,
    SettingsUpdate,
    TrainRequest,
)
from app.services import dataset_service, predict_service, report_service, train_service
from app.state import STATE

router = APIRouter(prefix="/api")


def _http_error(exc: Exception) -> HTTPException:
    status = 400
    if isinstance(exc, FileNotFoundError):
        status = 404
    return HTTPException(status_code=status, detail=str(exc))


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "tensorflow": tensorflow_status(),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@router.get("/overview")
def overview() -> dict[str, Any]:
    counts = STATE.db.overview_counts()
    class_counts = {}
    rare = []
    if STATE.has_dataset():
        class_counts = class_distribution(STATE.frame)
        from app.ml.features import rare_class_warnings

        rare = rare_class_warnings(class_counts)
    comparison = STATE.comparison_rows()
    best = comparison[0] if comparison else None
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "disclaimer": MEDICAL_DISCLAIMER,
        "synthetic_label": SYNTHETIC_DATA_LABEL if STATE.is_synthetic else None,
        "dataset_loaded": STATE.has_dataset(),
        "dataset_name": STATE.dataset_name,
        "is_synthetic": STATE.is_synthetic,
        "n_rows": int(len(STATE.frame)) if STATE.has_dataset() else 0,
        "class_counts": class_counts,
        "rare_class_warnings": rare,
        "storage_counts": counts,
        "trained_models": list(STATE.trained_models.keys()),
        "active_model": STATE.active_model_name,
        "best_model": best,
        "comparison": comparison,
        "preprocess_ready": STATE.preprocess_report is not None,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@router.get("/dataset/status")
def dataset_status():
    return dataset_service.current_dataset_status()


@router.post("/dataset/load-default")
def dataset_load_default():
    try:
        return dataset_service.load_default_sample()
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/dataset/upload")
async def dataset_upload(
    file: UploadFile = File(...),
    mark_synthetic: bool = Form(False),
):
    try:
        content = await file.read()
        return dataset_service.upload_dataset(
            file.filename or "upload.xlsx",
            content,
            mark_synthetic=mark_synthetic,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/dataset/preview")
def dataset_preview(limit: int = Query(50, ge=1, le=200)):
    try:
        return dataset_service.dataset_preview(limit=limit)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/dataset/stats")
def dataset_stats():
    try:
        return dataset_service.dataset_stats()
    except Exception as exc:
        raise _http_error(exc) from exc


# ---------------------------------------------------------------------------
# Preprocessing & training
# ---------------------------------------------------------------------------


@router.post("/preprocess")
def preprocess(body: PreprocessRequest):
    try:
        return train_service.run_preprocessing(
            body.features,
            scale_numeric=body.scale_numeric,
            numeric_impute=body.numeric_impute,
            test_size=body.test_size,
            seed=body.seed,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/preprocess/status")
def preprocess_status():
    return {
        "ready": STATE.preprocess_report is not None,
        "report": STATE.preprocess_report,
        "config": STATE.preprocess_config,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@router.get("/models/catalog")
def models_catalog():
    return train_service.list_model_catalog()


@router.post("/train")
def train(body: TrainRequest):
    try:
        return train_service.train_models(
            body.model_names,
            features=body.features,
            hyperparameters=body.hyperparameters,
            test_size=body.test_size,
            seed=body.seed,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/ann/train")
def ann_train(body: ANNTrainRequest):
    try:
        return train_service.train_ann_model(
            features=body.features,
            hidden_layers=body.hidden_layers,
            dropout=body.dropout,
            learning_rate=body.learning_rate,
            epochs=body.epochs,
            batch_size=body.batch_size,
            test_size=body.test_size,
            seed=body.seed,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/ann/analytics")
def ann_analytics():
    bundle = STATE.trained_models.get("ANN")
    if bundle is None:
        return {
            "trained": False,
            "tensorflow": tensorflow_status(),
            "disclaimer": MEDICAL_DISCLAIMER,
        }
    return {
        "trained": True,
        "model_name": "ANN",
        "backend": getattr(bundle, "backend", "unknown"),
        "metrics": bundle.metrics,
        "history": getattr(bundle, "history", {}),
        "hyperparameters": bundle.hyperparameters,
        "class_labels": bundle.class_labels,
        "n_train": bundle.n_train,
        "n_test": bundle.n_test,
        "tensorflow": tensorflow_status(),
        "disclaimer": MEDICAL_DISCLAIMER,
        "is_synthetic": STATE.is_synthetic,
    }


@router.get("/train/status")
def train_status():
    return train_service.training_status()


@router.get("/comparison")
def comparison():
    rows = STATE.comparison_rows()
    if not rows:
        # hydrate from DB if memory empty
        for run in STATE.db.list_training_runs(limit=50):
            metrics = json.loads(run["metrics_json"])
            rows.append(
                {
                    "model_name": run["model_name"],
                    "family": run["model_family"],
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_weighted": metrics.get("f1_weighted"),
                    "roc_auc_ovr_weighted": metrics.get("roc_auc_ovr_weighted"),
                    "n_train": run.get("n_train"),
                    "n_test": run.get("n_test"),
                    "class_labels": json.loads(run["class_labels_json"] or "[]"),
                    "from_db_only": True,
                }
            )
        rows.sort(key=lambda r: (r.get("f1_macro") or 0), reverse=True)
    return {
        "models": rows,
        "active_model": STATE.active_model_name,
        "disclaimer": MEDICAL_DISCLAIMER,
        "is_synthetic": STATE.is_synthetic,
    }


@router.post("/models/active")
def set_active_model(body: ActiveModelRequest):
    if body.model_name not in STATE.trained_models:
        raise HTTPException(status_code=404, detail="Model not loaded in memory. Retrain it.")
    STATE.active_model_name = body.model_name
    return {"active_model": STATE.active_model_name}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@router.get("/assessment/schema")
def assessment_schema():
    return predict_service.assessment_schema()


@router.post("/predict")
def predict(body: PredictRequest):
    try:
        return predict_service.predict_one(body.features, model_name=body.model_name)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_name: str | None = Form(None),
):
    try:
        content = await file.read()
        result = predict_service.predict_batch_upload(
            file.filename or "batch.xlsx",
            content,
            model_name=model_name,
        )
        # Do not serialize the full frame in JSON.
        result = {k: v for k, v in result.items() if k != "result_frame"}
        return result
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/predict/{prediction_id}")
def get_prediction(prediction_id: str):
    try:
        return predict_service.get_prediction(prediction_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/history")
def history(limit: int = Query(100, ge=1, le=500)):
    return predict_service.history(limit=limit)


@router.delete("/history/{prediction_id}")
def delete_history_item(prediction_id: str):
    ok = STATE.db.delete_prediction(prediction_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"deleted": prediction_id}


@router.delete("/history")
def clear_history():
    n = STATE.db.clear_predictions()
    return {"deleted": n}


@router.get("/experiments")
def experiments(limit: int = Query(50, ge=1, le=200)):
    rows = STATE.db.list_training_runs(limit=limit)
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "features": json.loads(row["features_json"]),
                "hyperparameters": json.loads(row["hyperparameters_json"] or "{}"),
                "metrics": json.loads(row["metrics_json"]),
                "class_labels": json.loads(row["class_labels_json"] or "[]"),
            }
        )
    return {"items": parsed, "disclaimer": MEDICAL_DISCLAIMER}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@router.post("/reports/pdf")
def reports_pdf(body: PdfReportRequest):
    try:
        prediction = None
        if body.prediction_id:
            prediction = predict_service.get_prediction(body.prediction_id)
        content = report_service.build_prediction_pdf(
            prediction=prediction,
            comparison=body.include_comparison,
        )
        filename = "Insulin_Matrix_Report.pdf"
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/reports/excel")
def reports_excel():
    try:
        content = report_service.build_comparison_excel()
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": 'attachment; filename="Insulin_Matrix_Comparison.xlsx"'
            },
        )
    except Exception as exc:
        raise _http_error(exc) from exc


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@router.get("/settings")
def get_settings():
    return {
        "settings": STATE.settings,
        "assessment_fields": ASSESSMENT_FIELDS,
        "disclaimer": MEDICAL_DISCLAIMER,
        "app": APP_NAME,
        "version": APP_VERSION,
    }


@router.put("/settings")
def update_settings(body: SettingsUpdate):
    data = body.model_dump(exclude_none=True)
    STATE.settings.update(data)
    for key, value in data.items():
        STATE.db.set_setting(key, value)
    return {"settings": STATE.settings}
