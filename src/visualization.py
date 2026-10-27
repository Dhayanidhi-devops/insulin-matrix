"""Plotly figures for Insulin Matrix. Charts use a white background and blue/teal marks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

NAVY = "#0B3D5C"
TEAL = "#0F766E"
BLUE = "#1D4E89"
AMBER = "#B86E00"


def _style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "font": {"size": 16, "color": NAVY}},
        font={"family": "Source Sans 3, Segoe UI, sans-serif", "color": "#1A2332", "size": 13},
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[NAVY, TEAL, BLUE, "#3D8B8B", AMBER],
        margin={"l": 48, "r": 24, "t": 56, "b": 48},
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E6EEF5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E6EEF5", zeroline=False)
    return fig


def histogram(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=series.dropna(), marker_color=TEAL, opacity=0.9))
    fig.update_layout(xaxis_title=str(series.name or "Value"), yaxis_title="Rows")
    return _style(fig, title)


def box_plot(frame: pd.DataFrame, columns: list[str]) -> go.Figure:
    fig = go.Figure()
    for column in columns:
        fig.add_trace(go.Box(y=pd.to_numeric(frame[column], errors="coerce"), name=column, marker_color=NAVY))
    fig.update_layout(yaxis_title="Value")
    return _style(fig, "Distribution of selected columns")


def correlation_heatmap(frame: pd.DataFrame) -> go.Figure | None:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    matrix = numeric.corr(numeric_only=True)
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.columns),
            colorscale=[[0, "#F4E7D4"], [0.5, "#F7FBFC"], [1, TEAL]],
            zmin=-1,
            zmax=1,
            colorbar={"title": "r"},
        )
    )
    return _style(fig, "Correlation of numeric columns")


def actual_vs_predicted(y_true, y_pred) -> go.Figure:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    low = float(np.nanmin([actual.min(), predicted.min()]))
    high = float(np.nanmax([actual.max(), predicted.max()]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="Test rows", marker={"color": TEAL, "size": 8}))
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Identity", line={"color": NAVY, "dash": "dash"}))
    fig.update_layout(xaxis_title="Actual test value", yaxis_title="Predicted test value")
    return _style(fig, "Actual versus predicted — test partition")


def residual_plot(y_true, y_pred) -> go.Figure:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residual = actual - predicted
    fig = go.Figure(go.Scatter(x=predicted, y=residual, mode="markers", name="Residual", marker={"color": NAVY, "size": 8}))
    fig.add_hline(y=0, line_dash="dash", line_color=TEAL)
    fig.update_layout(xaxis_title="Predicted test value", yaxis_title="Actual − predicted")
    return _style(fig, "Residuals — test partition")


def roc_figure(y_true, scores, positive_class: str) -> go.Figure:
    actual = (np.asarray(y_true).astype(str) == positive_class).astype(int)
    fpr, tpr, _thresholds = roc_curve(actual, np.asarray(scores, dtype=float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Positive class: {positive_class}", line={"color": TEAL, "width": 2}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"color": "#9AA8B5", "dash": "dash"}))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return _style(fig, "ROC curve — test partition")


def precision_recall_figure(y_true, scores, positive_class: str) -> go.Figure:
    actual = (np.asarray(y_true).astype(str) == positive_class).astype(int)
    precision, recall, _thresholds = precision_recall_curve(actual, np.asarray(scores, dtype=float))
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", name=f"Positive class: {positive_class}", line={"color": NAVY, "width": 2}))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return _style(fig, "Precision–recall curve — test partition")


def confusion_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=[[0, "#F4FBFA"], [1, NAVY]],
            text=matrix.to_numpy(),
            texttemplate="%{text}",
            colorbar={"title": "Rows"},
        )
    )
    return _style(fig, "Confusion matrix — test partition")


def importance_bar(table: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    ordered = table.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=ordered[value_column], y=ordered["feature"], orientation="h", marker_color=TEAL))
    fig.update_layout(xaxis_title=value_column.replace("_", " "), yaxis_title="")
    return _style(fig, title)
