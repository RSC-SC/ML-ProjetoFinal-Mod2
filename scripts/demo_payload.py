#!/usr/bin/env python3
"""Sobe a API com `uvicorn` de verdade e grava um payload real de exemplo.

Gera:
  - `examples/payload_real.json` → resposta real, colada no README;
  - `examples/<imagem>.jpg`      → imagem usada (vem do split de teste do COCO).

Uso:
    python scripts/demo_payload.py
    python scripts/demo_payload.py --port 8001 --limit 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "http://127.0.0.1:{port}"


def wait_for(url: str, timeout: float = 90.0) -> dict:
    """Espera a API responder, devolvendo o /health."""
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            return httpx.get(url + "/health", timeout=5.0).json()
        except Exception as exc:  # noqa: BLE001 - servidor ainda subindo
            last = exc
            time.sleep(1.0)
    raise SystemExit(f"API não subiu em {timeout}s: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--limit", type=int, default=25, help="imagens candidatas avaliadas")
    args = parser.parse_args()

    images = sorted(Path("data/coco-subset/images/test").glob("*.jpg"))[: args.limit]
    if not images:
        print("ERRO: rode antes python scripts/prepare_dataset.py --download")
        return 1

    log_path = Path(tempfile.gettempdir()) / f"scanvision-uvicorn-{args.port}.log"
    Path("examples").mkdir(exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")

    print(f"Subindo uvicorn na porta {args.port}...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(args.port)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        base = BASE.format(port=args.port)
        health = wait_for(base)
        print(f"  health: {health}")

        best: tuple[Path, dict] | None = None
        for path in images:
            data = path.read_bytes()
            resp = httpx.post(
                base + "/detect",
                files={"file": (path.name, data, "image/jpeg")},
                timeout=60.0,
            )
            if resp.status_code != 200:
                print(f"  {path.name}: HTTP {resp.status_code} {resp.text[:120]}")
                continue
            body = resp.json()
            names = [d["class_name"] for d in body["detections"]]
            print(f"  {path.name}: {len(names)} detecção(ões) {names}")
            if best is None or len(names) > len(best[1]["detections"]):
                best = (path, body)
            if len(names) >= 3:
                break

        if best is None or not best[1]["detections"]:
            print("AVISO: nenhuma imagem candidata com detecções.")
            return 1

        path, body = best
        exemplo = Path("examples") / path.name
        shutil.copyfile(path, exemplo)
        payload = {
            "request": {
                "method": "POST",
                "endpoint": "/detect",
                "content_type": "multipart/form-data",
                "field": "file",
                "file": path.name,
            },
            "response": body,
        }
        Path("examples/payload_real.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nImagem de exemplo : {exemplo}")
        print("Payload real      : examples/payload_real.json")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
