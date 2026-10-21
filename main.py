"""Ponto de entrada da API.

Execução (como exigido no README do enunciado):
    uvicorn main:app --reload
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.core.config import get_settings
from app.services.detector import get_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scanvision")

DESCRIPTION = """
API de detecção de objetos para imagens de supermercado.

Envie uma imagem em `POST /detect` e receba as detecções em JSON
(classe, confiança e caixa delimitadora em pixels).

Uma interface de demonstração (que desenha as caixas no navegador) está
disponível em `/` — a API continua devolvendo apenas JSON.
"""


def create_app(*, load_model: bool = True) -> FastAPI:
    """Fabrica a aplicação.

    `load_model=False` é usado pelos testes unitários, que injetam um modelo
    falso e não precisam baixar pesos.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        if load_model:
            # O modelo é carregado UMA única vez aqui, nunca dentro de um request.
            logger.info("Carregando modelo '%s' (device=%s, scope=%s)...",
                        settings.model_path, settings.resolve_device(), settings.scope)
            get_detector().load()
        else:
            logger.warning("create_app(load_model=False): inferência desativada (testes).")
        yield

    app = FastAPI(
        title="ScanVision — Detecção de Objetos",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    # CORS só é habilitado quando explicitamente configurado (ALLOW_ORIGINS).
    # O frontend servido pela própria aplicação não precisa disso.
    settings = get_settings()
    if origins := settings.allowed_origins():
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["POST", "GET"],
            allow_headers=["*"],
        )
        logger.info("CORS habilitado para: %s", ", ".join(origins))

    @app.get("/health", tags=["utilidades"], summary="Saudabilidade da API e do modelo")
    def health() -> dict[str, str]:
        settings = get_settings()
        return {
            "status": "ok" if get_detector().loaded else "model_not_loaded",
            "model": settings.model_path,
            "device": settings.resolve_device(),
            "scope": settings.scope,
        }

    # Frontend de demonstração (HTML/JS puro) em GET /.
    # Montado por último: rotas registradas acima (/detect, /health, /docs)
    # continuam com prioridade na resolução de caminhos.
    frontend_dir = Path(__file__).resolve().parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
