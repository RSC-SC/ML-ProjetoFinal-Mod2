"""Testes da API: contrato JSON, validação de entrada e tratamento de erros."""
from __future__ import annotations

from tests.conftest import FakeBoxes, FakeModel, FakeResult, make_png

PNG = make_png()


def test_detect_devolve_o_contrato_exato_do_enunciado(client, fake_model):
    """O JSON precisa ter exatamente `status` + `detections[].{class_name, confidence, box}`."""
    resp = client.post("/detect", files={"file": ("pizza.png", PNG, "image/png")})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"status", "detections"}
    assert body["status"] == "success"
    assert len(body["detections"]) == 1

    det = body["detections"][0]
    assert set(det) == {"class_name", "confidence", "box"}
    assert set(det["box"]) == {"x_min", "y_min", "x_max", "y_max"}


def test_detect_filtra_classes_fora_do_escopo_do_dataset(client, fake_model):
    """scope=dataset reporta só as 13 classes — `person` (COCO 0) não deve aparecer."""
    body = client.post("/detect", files={"file": ("x.png", PNG, "image/png")}).json()

    names = [d["class_name"] for d in body["detections"]]
    assert names == ["pizza"]
    assert fake_model.last_kwargs["conf"] == 0.25
    assert fake_model.last_kwargs["device"] in {"mps", "cpu"}


def test_coordenadas_sao_pixels_inteiros_da_imagem_original(client, fake_model):
    det = client.post("/detect", files={"file": ("x.png", PNG, "image/png")}).json()["detections"][0]

    box = det["box"]
    assert all(isinstance(box[k], int) for k in ("x_min", "y_min", "x_max", "y_max"))
    assert (box["x_min"], box["y_min"], box["x_max"], box["y_max"]) == (10, 21, 110, 211)


def test_confiance_arredondada_entre_zero_e_um(client, fake_model):
    conf = client.post("/detect", files={"file": ("x.png", PNG, "image/png")}).json()[
        "detections"
    ][0]["confidence"]
    assert 0.0 <= conf <= 1.0
    assert conf == 0.91


def test_imagem_sem_nada_detectado_retorna_lista_vazia(client):
    """Imagem sem objetos: continua success, com detections: []."""
    detector_module_model = FakeModel([FakeResult(FakeBoxes([], [], []))])
    import app.services.detector as d

    d.get_detector()._model = detector_module_model

    resp = client.post("/detect", files={"file": ("vazio.png", PNG, "image/png")})

    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "detections": []}


def test_rejeita_content_type_nao_imagem(client, fake_model):
    resp = client.post("/detect", files={"file": ("dados.txt", b"ola", "text/plain")})

    assert resp.status_code == 400
    assert "Content-Type" in resp.json()["detail"]


def test_rejeita_bytes_que_nao_sao_imagem(client, fake_model):
    """Content-Type mente: a validação real é a decodificação."""
    resp = client.post("/detect", files={"file": ("falso.png", b"nao sou imagem", "image/png")})

    assert resp.status_code == 400
    assert "imagem" in resp.json()["detail"].lower()


def test_rejeita_arquivo_vazio(client, fake_model):
    resp = client.post("/detect", files={"file": ("vazio.png", b"", "image/png")})

    assert resp.status_code == 400
    assert "vazio" in resp.json()["detail"].lower()


def test_rejeita_upload_acima_do_limite(client, fake_model, monkeypatch):
    import app.core.config as config

    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    config.get_settings.cache_clear()

    payload = b"0" * (2 * 1024 * 1024)  # 2 MB
    resp = client.post("/detect", files={"file": ("grande.png", payload, "image/png")})

    assert resp.status_code == 413
    assert "1 MB" in resp.json()["detail"]


def test_retorna_503_quando_o_modelo_nao_foi_carregado(client):
    resp = client.post("/detect", files={"file": ("x.png", PNG, "image/png")})

    assert resp.status_code == 503
    assert "não carregado" in resp.json()["detail"]


def test_health_informa_modelo_device_e_escopo(client):
    body = client.get("/health").json()

    assert body["status"] == "model_not_loaded"
    assert body["model"] == "yolo11n.pt"
    assert body["device"] in {"mps", "cpu"}
    assert body["scope"] == "dataset"


def test_cors_so_é_habilitado_quando_configurado(monkeypatch):
    """Sem CORS_ORIGINS o preflight não é respondido; com a variável, é liberado."""
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from main import create_app

    origem = "http://localhost:5500"
    headers = {"Origin": origem, "Access-Control-Request-Method": "POST"}

    get_settings.cache_clear()
    with TestClient(create_app(load_model=False)) as sem_cors:
        assert "access-control-allow-origin" not in sem_cors.options(
            "/detect", headers=headers
        ).headers

    monkeypatch.setenv("CORS_ORIGINS", f"{origem}, http://outro.local")
    get_settings.cache_clear()
    with TestClient(create_app(load_model=False)) as com_cors:
        resp = com_cors.options("/detect", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origem
