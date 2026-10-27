from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.database import Database
from src.evaluation import confusion_frame
from src.exports import model_workbook
from src.services import experiment_comparison, open_artifact
from src.ui import RESULT_KEY, download_bytes, get_active, page_header, show_exception
from src.visualization import (
    actual_vs_predicted,
    confusion_heatmap,
    importance_bar,
    precision_recall_figure,
    residual_plot,
    roc_figure,
)

page_header(
    "Model Evaluation",
    "Held-out test metrics are the primary result. Training metrics are labeled separately. Empty plots mean no model has been scored.",
)


def _view_from_artifact(artifact: dict) -> dict:
    importance = pd.DataFrame(artifact["importance_records"]) if artifact.get("importance_records") else None
    permutation = pd.DataFrame(artifact["permutation_records"]) if artifact.get("permutation_records") else None
    return {
        "experiment_id": artifact.get("experiment_id"),
        "metrics": artifact.get("metrics") or {},
        "messages": artifact.get("messages") or [],
        "evaluation": artifact.get("evaluation") or {},
        "importance": importance,
        "permutation": permutation,
        "model_name": artifact.get("model_name"),
        "target": artifact.get("target"),
        "task": artifact.get("task"),
        "features": artifact.get("features") or [],
        "dataset_type": artifact.get("dataset_type"),
        "dataset_id": artifact.get("dataset_id"),
        "split_mode": artifact.get("split_mode"),
        "preprocessing": artifact.get("preprocessing") or {},
        "hyperparameters": artifact.get("hyperparameters") or {},
        "seed": artifact.get("seed"),
        "test_size": artifact.get("test_size"),
    }


experiments = Database().list_experiments()
if experiments:
    labels = {f"{row['created_at']} · {row['model_name']} · {row['experiment_id']}": row["experiment_id"] for row in experiments}
    selected_label = st.selectbox("Saved experiment", list(labels))
    if st.button("Load this experiment"):
        try:
            _experiment, artifact = open_artifact(labels[selected_label])
            st.session_state[RESULT_KEY] = _view_from_artifact(artifact)
            st.rerun()
        except Exception as exc:
            show_exception(exc)

result = st.session_state.get(RESULT_KEY)
st.subheader("Model comparison")
comparison = experiment_comparison(experiments)
if comparison.empty:
    st.info("No experiments are stored. Train a model to see test metrics.")
else:
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption("Blank metric cells are not zeros. They are metrics that do not apply to that task.")

if not result:
    st.info("Train a model, or load a saved experiment, to see test-set plots.")
    st.stop()

if result.get("dataset_type") == "synthetic":
    st.warning("This experiment used synthetic demonstration data. The metrics are not clinical evidence.")

st.subheader(f"{result['model_name']} — test partition")
st.caption(
    f"Experiment {result['experiment_id']} · split {result['split_mode']} · "
    f"seed {result['seed']} · test proportion {result['test_size']} · target `{result['target']}`"
)
metrics = result["metrics"]
test_items = {key: value for key, value in metrics.items() if not str(key).startswith("train_") and key not in {"split_mode", "metric_set", "cv_scope"}}
train_items = {key: value for key, value in metrics.items() if str(key).startswith("train_")}
st.markdown("**Test metrics**")
st.dataframe(
    [{"metric": key, "value": "" if value is None else str(value)} for key, value in test_items.items()],
    width="stretch",
    hide_index=True,
)
if train_items:
    st.markdown("**Training-partition metrics**")
    st.caption("These are not the held-out result.")
    st.dataframe(
        [{"metric": key, "value": "" if value is None else str(value)} for key, value in train_items.items()],
        width="stretch",
        hide_index=True,
    )
for message in result.get("messages") or []:
    st.warning(message)

evaluation = result.get("evaluation") or {}
y_true = evaluation.get("y_true") or []
y_pred = evaluation.get("y_pred") or []
if y_true and y_pred and result["task"] == "regression":
    st.plotly_chart(actual_vs_predicted(y_true, y_pred), width="stretch")
    st.plotly_chart(residual_plot(y_true, y_pred), width="stretch")
elif y_true and y_pred and result["task"] == "classification":
    labels = evaluation.get("classes") or sorted(set(map(str, y_true)))
    matrix = confusion_frame(y_true, y_pred, labels)
    st.plotly_chart(confusion_heatmap(matrix), width="stretch")
    positive = evaluation.get("positive_class")
    proba = evaluation.get("y_proba")
    if positive and proba and positive in labels:
        scores = [row[labels.index(positive)] for row in proba]
        try:
            st.plotly_chart(roc_figure(y_true, scores, positive), width="stretch")
            st.plotly_chart(precision_recall_figure(y_true, scores, positive), width="stretch")
        except Exception as exc:
            show_exception(exc)
    elif metrics.get("roc_auc_note"):
        st.caption(metrics["roc_auc_note"])

importance = result.get("importance")
if isinstance(importance, pd.DataFrame) and not importance.empty:
    st.subheader("Model feature importance")
    st.caption("Values come from the fitted model after preprocessing. Coefficients are absolute values.")
    st.plotly_chart(importance_bar(importance, "importance", "Transformed-feature importance"), width="stretch")
    st.dataframe(importance, width="stretch", hide_index=True)
permutation = result.get("permutation")
if isinstance(permutation, pd.DataFrame) and not permutation.empty:
    st.subheader("Permutation importance on the test partition")
    st.plotly_chart(importance_bar(permutation, "importance_mean", "Permutation importance — test partition"), width="stretch")
    st.dataframe(permutation, width="stretch", hide_index=True)

active = get_active()
dataset_id = result.get("dataset_id") or (active["dataset_id"] if active else "")
dataset_type = result.get("dataset_type") or "real"
configuration = {
    "experiment_id": result.get("experiment_id"),
    "model_name": result.get("model_name"),
    "target": result.get("target"),
    "task": result.get("task"),
    "features": result.get("features"),
    "hyperparameters": result.get("hyperparameters"),
    "preprocessing": result.get("preprocessing"),
    "seed": result.get("seed"),
    "test_size": result.get("test_size"),
    "split_mode": result.get("split_mode"),
    "dataset_type": dataset_type,
}
download_bytes(
    "Download model metrics",
    model_workbook(
        metrics,
        importance if isinstance(importance, pd.DataFrame) else None,
        configuration,
        dataset_id=dataset_id,
        dataset_type=dataset_type,
    ),
    "Insulin_Matrix_Model_Metrics.xlsx",
    "download_metrics",
)
st.caption("The workbook contains test metrics, feature importance when available, and the experiment configuration.")
