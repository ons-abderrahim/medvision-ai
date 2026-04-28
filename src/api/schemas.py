"""Pydantic request/response schemas for the MedVision AI API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class LabelConfidence(BaseModel):
    label: str = Field(..., example="Pleural Effusion")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.87)


class PredictionResponse(BaseModel):
    predictions: list[LabelConfidence]
    model: str = Field(..., example="densenet_chexpert")
    top_label: str = Field(..., example="Pleural Effusion")
    top_confidence: float = Field(..., example=0.87)


class ModelInfo(BaseModel):
    name: str
    description: str
    modality: str
    loaded: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    device: str
