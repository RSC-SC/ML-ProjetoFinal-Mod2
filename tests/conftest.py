"""Fixtures: detector falso para testes rápidos (sem baixar pesos)."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.services.detector as detector_module
from app.core.config import get_settings
from main import create_app

#: IDs e nomes usados pelos modelos falsos (COCO).
FAKE_NAMES = {0: "person", 39: "bottle", 53: "pizza"}


class FakeBoxes:
    """Espelha a API de `result.boxes` da Ultralytics."""

    def __init__(self, xyxy: list, conf: list, cls: list) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)


class FakeResult:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes
        self.names = FAKE_NAMES


class FakeModel:
    """Modelo stub: devolve resultados pré-calculados sem inferência."""

    def __init__(self, results: list[FakeResult]) -> None:
        self._results = results
        self.names = FAKE_NAMES
        self.last_kwargs: dict[str, Any] = {}

    def predict(self, _image: Any, **kwargs: Any) -> list[FakeResult]:
        self.last_kwargs = kwargs
        return self._results


def make_png(size: int = 64, color: tuple[int, int, int] = (180, 180, 180)) -> bytes:
    """Gera uma imagem PNG válida em memória (usada como upload de teste)."""
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fresh_detector() -> Any:
    """Garante detector e configuração limpos a cada teste (sem estado entre testes)."""
    detector_module._detector = None
    get_settings.cache_clear()
    yield
    detector_module._detector = None
    get_settings.cache_clear()


@pytest.fixture
def client() -> Any:
    """Cliente de teste com a API sem carregar o modelo real."""
    with TestClient(create_app(load_model=False)) as test_client:
        yield test_client


@pytest.fixture
def fake_model() -> FakeModel:
    """Detector com uma detecção de pizza (53) e uma de person (0) fora do escopo."""
    model = FakeModel(
        [FakeResult(FakeBoxes([[10.2, 20.6, 110.4, 210.8], [1, 2, 30, 40]], [0.91, 0.7], [53, 0]))]
    )
    detector_module.get_detector()._model = model
    return model
