"""Inferência com YOLO: carrega os pesos uma única vez e traduz para o contrato JSON."""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from app.core.classes import SELECTED_SET
from app.core.config import Settings, get_settings
from app.schemas import Box, Detection

logger = logging.getLogger(__name__)


def to_numpy(value: Any) -> np.ndarray:
    """Converte tensor/escalar/lista do resultado do YOLO para numpy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def build_detection(
    names: dict[int, str], cls_id: int, conf: float, xyxy: np.ndarray
) -> Detection:
    """Monta uma detecção arredondando as coordenadas para pixels inteiros."""
    x_min, y_min, x_max, y_max = (int(round(float(v))) for v in xyxy[:4])
    return Detection(
        class_name=names.get(cls_id, str(cls_id)),
        confidence=round(float(conf), 4),
        box=Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
    )


class Detector:
    """Detecta objetos em imagens com um modelo YOLO da Ultralytics."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        """Indica se os pesos já foram carregados."""
        return self._model is not None

    @property
    def settings(self) -> Settings:
        """Configuração usada por este detector."""
        return self._settings

    def load(self) -> None:
        """Baixa (se necessário) e carrega os pesos. Chamado só no startup."""
        from ultralytics import YOLO  # import pesado: fica fora do caminho do startup

        path = self._settings.model_path
        self._model = YOLO(path)
        logger.info("Modelo '%s' carregado (device=%s)", path, self._settings.resolve_device())

    def detect(self, image_bytes: bytes) -> list[Detection]:
        """Executa a inferência e devolve as detecções ordenadas por confiança.

        Levanta `UnidentifiedImageError`/`OSError` para imagem inválida e
        `RuntimeError` quando o modelo não foi carregado.
        """
        if self._model is None:
            raise RuntimeError("Modelo de detecção não carregado.")

        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # força a decodificação agora, revelando arquivos corrompidos
        image = image.convert("RGB")

        s = self._settings
        results = self._model.predict(
            image,
            conf=s.conf,
            iou=s.iou,
            device=s.resolve_device(),
            imgsz=640,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = to_numpy(boxes.xyxy)
        if xyxy.size == 0:
            return []
        confs = to_numpy(boxes.conf)
        classes = to_numpy(boxes.cls)
        names = dict(getattr(result, "names", None) or self._model.names)

        detections: list[Detection] = []
        for box, conf, cls_id in zip(xyxy, confs, classes):
            cid = int(cls_id)
            # Escopo do dataset: por padrão só reportamos as 13 classes alvo.
            if s.scope == "dataset" and cid not in SELECTED_SET:
                continue
            detections.append(build_detection(names, cid, conf, box))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


_detector: Detector | None = None


def get_detector() -> Detector:
    """Singleton do detector — o modelo é carregado uma única vez no startup."""
    global _detector
    if _detector is None:
        _detector = Detector(get_settings())
    return _detector
