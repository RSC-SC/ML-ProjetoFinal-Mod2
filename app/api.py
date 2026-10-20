"""Endpoint POST /detect — recebe uma imagem e devolve o contrato JSON do enunciado."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import UnidentifiedImageError

from app.core.config import get_settings
from app.schemas import DetectResponse
from app.services.detector import get_detector

logger = logging.getLogger(__name__)

router = APIRouter()

#: Content-Types que aceitamos sem questionar.
ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/gif",
}
#: Clientes que não sabem informar o tipo real do arquivo (a validação real é a decodificação).
LENIENT_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detecta objetos em uma imagem",
    response_description="Lista de detecções (classe, confiança e caixa em pixels)",
)
async def detect(
    request: Request,
    file: UploadFile = File(..., description="Imagem a ser analisada (JPEG, PNG, WEBP...)"),
) -> DetectResponse:
    """Recebe um arquivo de imagem e devolve as detecções.

    Erros:
    - 400: content-type não suportado, arquivo vazio ou imagem corrompida;
    - 413: arquivo acima de MAX_UPLOAD_MB;
    - 503: modelo não carregado.
    """
    settings = get_settings()
    content_type = (file.content_type or "").lower()

    # 1) Rejeição barata pelo header, antes de ler o corpo inteiro.
    declared = int(request.headers.get("content-length") or 0)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if declared > max_bytes + 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo acima do limite de {settings.max_upload_mb} MB.",
        )

    # 2) Tipo declarado.
    if content_type not in ALLOWED_TYPES and content_type not in LENIENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Content-Type não suportado: '{content_type}'. "
                "Envie uma imagem JPEG, PNG ou WEBP."
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo acima do limite de {settings.max_upload_mb} MB.",
        )

    # 3) Inferência (a decodificação real acontece dentro de detect()).
    try:
        detections = get_detector().detect(data)
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="O arquivo não é uma imagem válida ou está corrompido."
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Falha ao ler a imagem: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info("POST /detect -> %d detecção(ões) (%s)", len(detections), file.filename or "?")
    return DetectResponse(status="success", detections=detections)
