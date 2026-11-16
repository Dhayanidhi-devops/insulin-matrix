"""Pydantic request / response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    features: list[str] | None = None
    scale_numeric: bool | None = None
    numeric_impute: str | None = None
    test_size: float | None = Field(default=None, ge=0.05, le=0.5)
    seed: int | None = None


class TrainRequest(BaseModel):
    model_names: list[str]
    features: list[str] | None = None
    hyperparameters: dict[str, Any] | None = None
    test_size: float | None = Field(default=None, ge=0.05, le=0.5)
    seed: int | None = None


class ANNTrainRequest(BaseModel):
    features: list[str] | None = None
    hidden_layers: list[int] | None = None
    dropout: float = Field(default=0.2, ge=0.0, le=0.8)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    epochs: int = Field(default=25, ge=1, le=200)
    batch_size: int = Field(default=64, ge=8, le=512)
    test_size: float | None = Field(default=None, ge=0.05, le=0.5)
    seed: int | None = None


class PredictRequest(BaseModel):
    model_name: str | None = None
    features: dict[str, Any]


class ActiveModelRequest(BaseModel):
    model_name: str


class SettingsUpdate(BaseModel):
    default_test_size: float | None = Field(default=None, ge=0.05, le=0.5)
    default_seed: int | None = None
    scale_numeric: bool | None = None
    numeric_impute: str | None = None
    auto_load_sample: bool | None = None


class PdfReportRequest(BaseModel):
    prediction_id: str | None = None
    include_comparison: bool = True
