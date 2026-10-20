"""Contrato JSON exigido pelo enunciado — mantenha estes nomes de campo exatos."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Retângulo em pixels da imagem original (não normalizado)."""

    x_min: int = Field(..., description="Coordenada X do canto superior esquerdo")
    y_min: int = Field(..., description="Coordenada Y do canto superior esquerdo")
    x_max: int = Field(..., description="Coordenada X do canto inferior direito")
    y_max: int = Field(..., description="Coordenada Y do canto inferior direito")


class Detection(BaseModel):
    """Um objeto detectado."""

    class_name: str = Field(..., description="Nome da classe (ex.: pizza)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança entre 0 e 1")
    box: Box = Field(..., description="Caixa delimitadora em pixels")


class DetectResponse(BaseModel):
    """Resposta do POST /detect — sem imagem desenhada, só estrutura."""

    status: Literal["success"] = "success"
    detections: list[Detection] = Field(default_factory=list)
