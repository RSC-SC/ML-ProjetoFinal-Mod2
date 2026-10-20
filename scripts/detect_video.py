#!/usr/bin/env python3
"""Testa a API com VÍDEO, sem tocar no contrato da API.

O enunciado exige `POST /detect` com imagem, então este script não cria endpoint de
vídeo: ele **extrai quadros do vídeo, envia cada quadro para a API rodando** (o mesmo
caminho que o avaliador usa) e grava um vídeo de saída com as caixas desenhadas.

Serve para demonstrar no vídeo da entrega que o sistema acompanha objetos em movimento
e para medir a latência real por quadro em `device=mps`.

Requisito: API rodando (`uvicorn main:app --reload`) e um arquivo de vídeo (.mp4/.mov).

Uso:
    python scripts/detect_video.py meufilme.mov
    python scripts/detect_video.py meufilme.mov --fps 4 --max-width 960
    python scripts/detect_video.py meufilme.mov --api http://127.0.0.1:8001 --save-frames
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PALETTE = [
    (220, 38, 38), (37, 99, 235), (22, 163, 74), (217, 119, 6), (124, 58, 237),
    (13, 148, 136), (192, 38, 211), (202, 138, 4), (71, 85, 105), (2, 132, 199),
]
COLOR_BY_CLASS: dict[str, tuple[int, int, int]] = {}


def color_for(class_name: str) -> tuple[int, int, int]:
    """Cor fixa por classe, para o objeto não trocar de cor entre quadros."""
    if class_name not in COLOR_BY_CLASS:
        COLOR_BY_CLASS[class_name] = PALETTE[len(COLOR_BY_CLASS) % len(PALETTE)]
    return COLOR_BY_CLASS[class_name]


def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Desenha caixas + rótulos (só para o arquivo de saída; a API não desenha nada)."""
    for det in detections:
        b = det["box"]
        color = color_for(det["class_name"])
        cv2.rectangle(frame, (b["x_min"], b["y_min"]), (b["x_max"], b["y_max"]), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        y0 = max(0, b["y_min"] - th - 6)
        cv2.rectangle(frame, (b["x_min"], y0), (b["x_min"] + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(
            frame, label, (b["x_min"] + 3, y0 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="arquivo de vídeo (.mp4, .mov, .avi...)")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="URL da API rodando")
    parser.add_argument("--fps", type=float, default=2.0, help="quadros por segundo enviados")
    parser.add_argument("--max-width", type=int, default=960, help="largura máxima do quadro")
    parser.add_argument("--max-frames", type=int, default=0, help="limite de quadros (0 = todos)")
    parser.add_argument("--out", default="runs/video", help="pasta do vídeo anotado (gitignored)")
    parser.add_argument("--save-frames", action="store_true", help="também salva os quadros .jpg")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERRO: vídeo não encontrado: {video_path}")
        return 1

    try:
        health = httpx.get(args.api + "/health", timeout=5.0).json()
    except httpx.HTTPError as exc:
        print(f"ERRO: API não está rodando em {args.api} ({exc}).")
        print("Rode antes:  uvicorn main:app --reload")
        return 1
    print(f"API: {health}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"ERRO: não consegui abrir {video_path}")
        return 1

    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / args.fps))  # pega 1 quadro a cada `step`
    print(
        f"Vídeo: {video_path.name} · {src_fps:.1f} fps · {total} quadros · "
        f"enviando {src_fps / step:.1f} quadros/s (1 a cada {step})"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "quadros"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    out_path = out_dir / f"{video_path.stem}_anotado.mp4"

    contagem: collections.Counter[str] = collections.Counter()
    latencias: list[float] = []
    enviados = 0
    sem_deteccao = 0

    with httpx.Client(timeout=60.0) as client:
        indice = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if indice % step:
                indice += 1
                continue
            indice += 1

            if args.max_frames and enviados >= args.max_frames:
                break

            h, w = frame.shape[:2]
            if w > args.max_width:
                scale = args.max_width / w
                frame = cv2.resize(frame, (args.max_width, int(h * scale)))

            ok_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok_enc:
                continue

            t0 = time.perf_counter()
            resp = client.post(
                args.api + "/detect",
                files={"file": (f"frame_{indice:05d}.jpg", buffer.tobytes(), "image/jpeg")},
            )
            latencias.append((time.perf_counter() - t0) * 1000)
            enviados += 1

            if resp.status_code != 200:
                print(f"  quadro {indice}: HTTP {resp.status_code} {resp.text[:120]}")
                continue
            dets = resp.json()["detections"]
            if not dets:
                sem_deteccao += 1
            for d in dets:
                contagem[d["class_name"]] += 1

            anotado = draw(frame.copy(), dets)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    src_fps / step,
                    (anotado.shape[1], anotado.shape[0]),
                )
            writer.write(anotado)
            if args.save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{indice:05d}.jpg"), anotado)

    capture.release()
    if writer is not None:
        writer.release()

    if not latencias:
        print("Nenhum quadro processado.")
        return 1

    print(f"\nQuadros enviados à API: {enviados}")
    print(
        f"Latência por quadro: média {statistics.mean(latencias):.0f} ms · "
        f"mediana {statistics.median(latencias):.0f} ms · "
        f"máx {max(latencias):.0f} ms → {1000 / statistics.mean(latencias):.1f} quadros/s"
    )
    print(f"Quadros sem nenhuma detecção: {sem_deteccao}/{enviados}")
    print("Classes vistas (soma de detecções):")
    for nome, n in contagem.most_common():
        print(f"  {nome:10s} {n}")
    print(f"\nVídeo anotado: {out_path}")
    if args.save_frames:
        print(f"Quadros anotados: {frames_dir}/")
    print("Obs: o desenho é só do script — a API em si continua devolvendo JSON puro.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
