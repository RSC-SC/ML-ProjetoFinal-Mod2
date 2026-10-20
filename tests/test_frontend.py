"""Testes do frontend estático servido em GET / (não carregam o modelo)."""
from __future__ import annotations

from typing import Any


def test_index_e_html(client: Any) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ScanVision" in resp.text


def test_assets_estaticos_sao_servidos(client: Any) -> None:
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]

    css = client.get("/style.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]


def test_rotas_da_api_nao_sao_sombreadas_pelo_frontend(client: Any) -> None:
    """/detect, /health e /docs precisam responder antes do mount em '/'."""
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
