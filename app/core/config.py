"""Configuração da aplicação por variáveis de ambiente (carrega um `.env` se existir)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Exemplos: MODEL_PATH=yolo11n.pt, CONF=0.25, IOU=0.45, DEVICE=mps, SCOPE=dataset
#: MAX_UPLOAD_MB=10 (nenhum é obrigatório — todos têm padrão sensato).


class Settings(BaseSettings):
    """Parâmetros ajustáveis sem mudar código."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    #: Pesos do YOLO. `yolo11n.pt` é baixado automaticamente na primeira execução.
    model_path: str = "yolo11n.pt"
    #: Confiança mínima para reportar uma detecção (0..1).
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    #: IoU para supressão de não-máximos (0..1).
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    #: `auto` usa MPS quando disponível e cai para CPU.
    device: str = "auto"
    #: Tamanho máximo do upload, em megabytes.
    max_upload_mb: int = Field(default=10, gt=0)
    #: `dataset` = só as 13 classes do dataset; `all` = as 80 classes do COCO.
    scope: Literal["dataset", "all"] = "dataset"

    def resolve_device(self) -> str:
        """Resolve `auto` para `mps` (Apple Silicon) ou `cpu`."""
        if self.device not in ("auto", ""):
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001 - torch ausente não pode derrubar a API
            pass
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton das configurações (cacheado para não reler o `.env` a cada request)."""
    return Settings()
