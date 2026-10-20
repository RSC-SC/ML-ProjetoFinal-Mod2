#!/usr/bin/env python3
"""Fine-tuning OPCIONAL do enunciado — não é necessário para a entrega.

Deixado pronto no repo para que, com um único comando, o bônus da rubrica
("de forma opcional, realizou o fine-tuning") possa ser conquistado sem refazer nada.

    .venv/bin/python scripts/train.py              # 30 epochs (padrão)
    .venv/bin/python scripts/train.py --epochs 5   # smoke test rápido

Saída: runs/detect/<name>/ (gitignorado). Métricas em results.csv e results.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATA_YAML = Path("data/coco-subset/data.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="épocas de treino")
    parser.add_argument("--batch", type=int, default=16, help="tamanho do batch")
    parser.add_argument("--imgsz", type=int, default=640, help="lado da imagem")
    parser.add_argument("--seed", type=int, default=42, help="seed (reprodutibilidade)")
    parser.add_argument("--name", default="scanvision", help="nome da pasta em runs/detect/")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print(f"ERRO: {DATA_YAML} não existe. Rode antes: python scripts/prepare_dataset.py --download")
        return 1

    from ultralytics import YOLO

    # Começa dos pesos pré-treinados no COCO (transfer learning) — muito mais
    # rápido e estável do que treinar do zero com poucos dados.
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        device="mps",  # Apple Silicon — nunca device=0 (não existe CUDA aqui)
        patience=10,  # early stopping
        project="runs/detect",
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    metrics = results.results_dict
    print("\n=== Métricas do fine-tuning (registradas em runs/detect/{name}) ===")
    for key in (
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    print("\nPara usar os pesos treinados na API:  MODEL_PATH=runs/detect/"
          f"{args.name}/weights/best.pt uvicorn main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
