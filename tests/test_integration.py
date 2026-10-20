"""Teste de integração: modelo real carregado uma única vez.

Roda com `pytest -m integration` (baixa `yolo11n.pt` ~5,6 MB na primeira vez).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from fastapi.testclient import TestClient  # noqa: E402

from main import create_app  # noqa: E402
from tests.conftest import make_png  # noqa: E402


@pytest.fixture
def real_client(fresh_detector):
    """API com o modelo de verdade (cara: baixa pesos na primeira execução).

    Depende de `fresh_detector` para que o detector seja limpo ANTES do
    startup recarregar os pesos (senão o reset autouse apagava o modelo já carregado).
    """
    try:
        with TestClient(create_app(load_model=True)) as client:
            yield client
    except Exception as exc:  # noqa: BLE001 - offline ou sem espaço não pode quebrar a suite
        pytest.skip(f"modelo real indisponível: {exc}")


def test_health_confirma_modelo_carregado(real_client):
    body = real_client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model"] == "yolo11n.pt"
    assert body["device"] in {"mps", "cpu"}


def test_detect_com_imagem_real_devolve_contrato_valido(real_client):
    """Imagem sintética: provo o caminho completo da inferência (pode não detectar nada)."""
    resp = real_client.post("/detect", files={"file": ("sintetica.png", make_png(320), "image/png")})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert isinstance(body["detections"], list)
    for det in body["detections"]:
        assert set(det) == {"class_name", "confidence", "box"}
        assert 0.0 <= det["confidence"] <= 1.0
        box = det["box"]
        assert box["x_max"] > box["x_min"] and box["y_max"] > box["y_min"]


def test_detect_com_foto_de_exemplo_quando_existir(real_client):
    """Se houver fotos em examples/, usa uma delas e registra o resultado real."""
    examples = sorted(Path("examples").glob("*")) if Path("examples").exists() else []
    if not examples:
        pytest.skip("pasta examples/ ainda não populada")

    alvo = examples[0]
    with open(alvo, "rb") as fh:
        resp = real_client.post(
            "/detect", files={"file": (alvo.name, fh.read(), "application/octet-stream")}
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    print(f"\n{alvo.name}: {len(body['detections'])} detecção(ões) -> "
          f"{[d['class_name'] for d in body['detections']]}")
