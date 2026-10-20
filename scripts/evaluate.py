#!/usr/bin/env python3
"""Avalia a API no split de teste do dataset — só métricas medidas, nada estimado.

Roda o MESMO caminho de inferência da API (`Detector.detect`, com `conf=0.25`,
`iou=0.45` e escopo das 13 classes) sobre `data/coco-subset/images/test/` e compara
com as anotações de referência (caixas derivadas dos polígonos do COCO).

Saída: `docs/evaluation-report.md` com precisão, recall e F1 por classe @IoU 0,5.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --limit 50
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.classes import COCO_NAMES, SELECTED_IDS, SELECTED_SET  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.detector import Detector  # noqa: E402

IOU_MIN = 0.5


def load_gt(label_path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """Lê um rótulo YOLO-segmentação e devolve (classe, x1, y1, x2, y2) em pixels."""
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if cls_id not in SELECTED_SET:
            continue
        coords = [float(v) for v in parts[1:]]
        # Formato YOLO-seg: pares (x, y) normalizados formando o polígono.
        xs, ys = coords[0::2], coords[1::2]
        if not xs or not ys:
            continue
        boxes.append(
            (cls_id, min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)
        )
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Interseção sobre união de duas caixas (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="processa só N imagens")
    parser.add_argument("--out", default="docs/evaluation-report.md")
    args = parser.parse_args()

    root = Path("data/coco-subset")
    images = sorted((root / "images" / args.split).glob("*.jpg"))
    if not images:
        print(f"ERRO: nenhum imagem em {root / 'images' / args.split}. Rode prepare_dataset.py.")
        return 1
    if args.limit:
        images = images[: args.limit]

    settings = get_settings()
    detector = Detector(settings)
    detector.load()

    id_by_name = {name: cid for cid, name in COCO_NAMES.items()}
    stats: dict[int, dict[str, float]] = collections.defaultdict(
        lambda: {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "conf": 0.0, "n_conf": 0}
    )
    images_with_gt: set[int] = set()
    images_with_det: set[int] = set()

    print(f"Avaliando {len(images)} imagens do split '{args.split}' "
          f"(model={settings.model_path}, conf={settings.conf}, device={settings.resolve_device()})")
    for path in images:
        data = path.read_bytes()
        with Image.open(path) as img:
            width, height = img.size
        gt = load_gt(root / "labels" / args.split / f"{path.stem}.txt", width, height)
        for cls_id, *_ in gt:
            stats[cls_id]["gt"] += 1
            images_with_gt.add(cls_id)

        detections = detector.detect(data)
        preds = [
            (id_by_name[d.class_name], (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max),
             d.confidence)
            for d in detections
            if d.class_name in id_by_name
        ]
        for cls_id, _, _ in preds:
            images_with_det.add(cls_id)

        # Casamento guloso: para cada classe, prioriza a maior IoU.
        unmatched_gt = list(range(len(gt)))
        for cls_id, pbox, conf in sorted(preds, key=lambda p: -p[2]):
            stats[cls_id]["fp"] += 1  # vira TP se casar
            stats[cls_id]["conf"] += conf
            stats[cls_id]["n_conf"] += 1
            best_iou, best_j = 0.0, -1
            for j in unmatched_gt:
                g = gt[j]
                if g[0] != cls_id:
                    continue
                v = iou(pbox, g[1:])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= IOU_MIN:
                unmatched_gt.remove(best_j)
                stats[cls_id]["tp"] += 1
                stats[cls_id]["fp"] -= 1
        for j in unmatched_gt:
            stats[gt[j][0]]["fn"] += 1

    # Tabela final.
    lines = [
        "# Relatório de avaliação (split de teste)",
        "",
        f"- Modelo: `{settings.model_path}` (pré-treinado no COCO, **sem fine-tuning**)",
        f"- Parâmetros da API: `conf={settings.conf}`, `iou={settings.iou}`, "
        f"`device={settings.resolve_device()}`, `scope={settings.scope}`",
        f"- Imagens avaliadas: **{len(images)}** (split `{args.split}`)",
        f"- Critério de acerto: IoU ≥ {IOU_MIN} na mesma classe",
        "",
        "| Classe | GT | TP | FP | FN | Precisão | Recall | F1 | Conf. média |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tot = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
    for cls_id in SELECTED_IDS:
        s = stats[cls_id]
        gt_n, tp, fp, fn = int(s["gt"]), int(s["tp"]), int(s["fp"]), int(s["fn"])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        conf = s["conf"] / s["n_conf"] if s["n_conf"] else 0.0
        tot["gt"] += gt_n
        tot["tp"] += tp
        tot["fp"] += fp
        tot["fn"] += fn
        lines.append(
            f"| {COCO_NAMES[cls_id]} | {gt_n} | {tp} | {fp} | {fn} | "
            f"{prec:.2f} | {rec:.2f} | {f1:.2f} | {conf:.2f} |"
        )

    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    lines += [
        "",
        f"**Total (micro):** GT {tot['gt']} · TP {tp} · FP {fp} · FN {fn} → "
        f"precisão **{prec:.2f}**, recall **{rec:.2f}**, F1 **{f1:.2f}**",
        "",
        f"- Classes presentes no GT: **{len(images_with_gt)}/13**",
        f"- Classes detectadas pela API: **{len(images_with_det)}/13** "
        f"({', '.join(sorted(COCO_NAMES[i] for i in images_with_det))})",
        "",
        "> Números medidos pela execução de `scripts/evaluate.py` — não estimados.",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRelatório gravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
