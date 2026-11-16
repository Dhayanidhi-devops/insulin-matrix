"""PDF (ReportLab) and Excel (OpenPyXL) report generation from real results."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import APP_NAME, APP_VERSION, MEDICAL_DISCLAIMER, SYNTHETIC_DATA_LABEL, report_dir
from app.database import utc_now
from app.state import STATE


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Hero",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#0B3D4A"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Warn",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#8A1C1C"),
            leading=12,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CenterSmall",
            parent=styles["Normal"],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#44555A"),
        )
    )
    return styles


def build_prediction_pdf(
    *,
    prediction: dict[str, Any] | None = None,
    comparison: bool = False,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=f"{APP_NAME} Report",
    )
    styles = _styles()
    story = []

    story.append(Paragraph(APP_NAME, styles["Hero"]))
    story.append(
        Paragraph(
            f"Version {APP_VERSION} · Generated {utc_now()}",
            styles["CenterSmall"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph(MEDICAL_DISCLAIMER, styles["Warn"]))
    if STATE.is_synthetic:
        story.append(Paragraph(SYNTHETIC_DATA_LABEL, styles["Warn"]))

    if prediction:
        story.append(Paragraph("Individual Prediction Report", styles["Heading2"]))
        story.append(
            Paragraph(
                f"Prediction ID: <b>{prediction.get('prediction_id', '')}</b>",
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                f"Model: <b>{prediction.get('model_name', '')}</b>",
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                f"Predicted class: <b>{prediction.get('predicted_class', '')}</b>",
                styles["Body"],
            )
        )
        story.append(Spacer(1, 8))

        probs = prediction.get("probabilities") or {}
        if probs:
            story.append(Paragraph("Class probabilities", styles["Heading3"]))
            data = [["Class", "Probability"]]
            for label, value in sorted(probs.items(), key=lambda x: -float(x[1] or 0)):
                data.append([str(label), f"{float(value):.4f}"])
            table = Table(data, colWidths=[3.2 * inch, 1.5 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D4A")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D6DA")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 10))

        inputs = prediction.get("input") or {}
        if inputs:
            story.append(Paragraph("Input features", styles["Heading3"]))
            data = [["Feature", "Value"]]
            for key, value in inputs.items():
                data.append([str(key), str(value)])
            table = Table(data, colWidths=[3.2 * inch, 2.5 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#146B78")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D6DA")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFB")]),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 10))

        explanation = prediction.get("explanation") or {}
        notes = explanation.get("clinical_screening_notes") or []
        if notes:
            story.append(Paragraph("Screening context notes", styles["Heading3"]))
            for note in notes:
                story.append(Paragraph(f"• {note}", styles["Body"]))

    if comparison or not prediction:
        story.append(Paragraph("Model Comparison", styles["Heading2"]))
        rows = STATE.comparison_rows()
        if not rows:
            # Fall back to DB training runs.
            for run in STATE.db.list_training_runs(limit=20):
                metrics = json.loads(run["metrics_json"])
                rows.append(
                    {
                        "model_name": run["model_name"],
                        "family": run["model_family"],
                        "accuracy": metrics.get("accuracy"),
                        "f1_macro": metrics.get("f1_macro"),
                        "precision_macro": metrics.get("precision_macro"),
                        "recall_macro": metrics.get("recall_macro"),
                    }
                )
        if rows:
            data = [["Model", "Family", "Accuracy", "F1 macro", "Precision", "Recall"]]
            for row in rows:
                data.append(
                    [
                        str(row.get("model_name")),
                        str(row.get("family")),
                        f"{float(row.get('accuracy') or 0):.4f}",
                        f"{float(row.get('f1_macro') or 0):.4f}",
                        f"{float(row.get('precision_macro') or 0):.4f}",
                        f"{float(row.get('recall_macro') or 0):.4f}",
                    ]
                )
            table = Table(
                data,
                colWidths=[1.4 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D4A")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D6DA")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]),
                    ]
                )
            )
            story.append(table)
        else:
            story.append(Paragraph("No trained models available yet.", styles["Body"]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(MEDICAL_DISCLAIMER, styles["Warn"]))
    doc.build(story)
    return buffer.getvalue()


def build_comparison_excel() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Model_Comparison"
    header_fill = PatternFill("solid", fgColor="0B3D4A")
    header_font = Font(color="FFFFFF", bold=True)

    ws.append([APP_NAME, f"v{APP_VERSION}", utc_now()])
    ws.append([MEDICAL_DISCLAIMER])
    if STATE.is_synthetic:
        ws.append([SYNTHETIC_DATA_LABEL])
    ws.append([])

    headers = [
        "model_name",
        "family",
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "f1_weighted",
        "roc_auc_ovr_weighted",
        "n_train",
        "n_test",
    ]
    ws.append(headers)
    for col, _ in enumerate(headers, start=1):
        cell = ws.cell(ws.max_row, col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    rows = STATE.comparison_rows()
    if not rows:
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
                }
            )

    for row in rows:
        ws.append([row.get(h) for h in headers])

    # Metrics detail sheet
    detail = wb.create_sheet("Run_Metrics")
    detail.append(["run_id", "model_name", "created_at", "metric", "value"])
    for run in STATE.db.list_training_runs(limit=50):
        metrics = json.loads(run["metrics_json"])
        for key in [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "f1_weighted",
            "roc_auc_ovr_weighted",
        ]:
            detail.append(
                [run["run_id"], run["model_name"], run["created_at"], key, metrics.get(key)]
            )

    notes = wb.create_sheet("Export_Notes")
    notes.append(["note", MEDICAL_DISCLAIMER])
    notes.append(["synthetic", SYNTHETIC_DATA_LABEL if STATE.is_synthetic else "Uploaded / unmarked"])
    notes.append(["dataset_id", STATE.dataset_id])
    notes.append(["dataset_name", STATE.dataset_name])

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def build_batch_excel(result_frame) -> bytes:
    buffer = BytesIO()
    # pandas ExcelWriter uses openpyxl
    with BytesIO() as bio:
        result_frame.to_excel(bio, index=False, sheet_name="Batch_Predictions")
        # Re-open to add notes sheet
        from openpyxl import load_workbook

        bio.seek(0)
        wb = load_workbook(bio)
        notes = wb.create_sheet("Export_Notes")
        notes.append(["disclaimer", MEDICAL_DISCLAIMER])
        notes.append(
            [
                "synthetic",
                SYNTHETIC_DATA_LABEL if STATE.is_synthetic else "Context may include real uploads",
            ]
        )
        notes.append(["generated_at", utc_now()])
        out = BytesIO()
        wb.save(out)
        return out.getvalue()


def save_report_bytes(content: bytes, filename: str) -> Path:
    path = report_dir() / filename
    path.write_bytes(content)
    return path
