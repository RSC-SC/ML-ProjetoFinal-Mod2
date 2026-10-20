#!/usr/bin/env python3
"""Testa a API em imagens externas (fora do COCO) e gera relatório de generalização.

Para cada imagem de `examples/external/`, roda o MESMO caminho de inferência da API e
compara com a classe esperada (o prefixo do nome do arquivo: `pizza_1.jpg` → `pizza`).

Sem anotações humanas aqui, então **não** é mAP: é a taxa de acerto da classe esperada
e as confianças observadas — exatamente o que dá para afirmar sem estimar.

Também grava cópias anotadas (com caixas) em `runs/external/` para demonstração no
vídeo — a API em si nunca devolve imagem desenhada.

Uso:
    python scripts/eval_external.py
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.detector import Detector  # noqa: E402

PALETTE = [
    (220, 38, 38), (37, 99, 235), (22, 163, 74), (217, 119, 6), (124, 58, 237),
    (13, 148, 136), (192, 38, 211), (202, 138, 4), (71, 85, 105), (2, 132, 199),
]


def annotate(image: Image.Image, dets: list, out_path: Path) -> None:
    """Salva uma cópia da imagem com as caixas desenhadas (só para o vídeo)."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, det in enumerate(dets):
        color = PALETTE[idx % len(PALETTE)]
        box = (det.box.x_min, det.box.y_min, det.box.x_max, det.box.y_max)
        draw.rectangle(box, outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        x0, y0 = box[0], max(0, box[1] - 14)
        draw.rectangle((x0, y0, x0 + 8 * len(label), y0 + 14), fill=color)
        draw.text((x0 + 3, y0 + 2), label, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=85)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="examples/external")
    parser.add_argument("--out", default="docs/external-report.md")
    parser.add_argument("--annotated", default="runs/external", help="cópias anotadas (gitignored)")
    args = parser.parse_args()

    folder = Path(args.dir)
    images = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        print(f"ERRO: nenhuma imagem em {folder}. Rode scripts/fetch_external.py.")
        return 1

    settings = get_settings()
    detector = Detector(settings)
    detector.load()

    esperado_por_arquivo: dict[str, str] = {}
    sidecar = folder / "atribuicao.json"
    if sidecar.exists():
        import json

        esperado_por_arquivo = {
            k: v["classe"] for k, v in json.loads(sidecar.read_text(encoding="utf-8")).items()
        }

    por_classe: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"total": 0, "hit": 0, "conf": 0.0, "n_conf": 0}
    )
    sem_deteccao = 0
    classes_detectadas: set[str] = set()
    linhas = [
        "# Relatório de generalização (imagens externas)",
        "",
        f"- Modelo: `{settings.model_path}` (pré-treinado, **sem fine-tuning**)",
        f"- Parâmetros: `conf={settings.conf}`, `iou={settings.iou}`, `device={settings.resolve_device()}`",
        f"- Imagens: **{len(images)}**, baixadas do Wikimedia Commons, **fora da distribuição do COCO**",
        "- Métrica: acertou a classe esperada (pelo nome do arquivo) com pelo menos uma detecção",
        "",
        "| Imagem | Classe esperada | Acerto | Detecções (classe/confiança) |",
        "|---|---|---|---|",
    ]

    for path in images:
        esperado = esperado_por_arquivo.get(path.name, path.stem.rsplit("_", 1)[0])
        esperado = esperado.replace("_", " ")
        dets = detector.detect(path.read_bytes())

        with Image.open(path) as img:
            annotate(img.copy(), dets, Path(args.annotated) / path.name)

        nomes = [f"{d.class_name} {d.confidence:.2f}" for d in dets] or ["—"]
        acerto = any(d.class_name == esperado for d in dets)
        if not dets:
            sem_deteccao += 1
        for d in dets:
            classes_detectadas.add(d.class_name)
            por_classe[esperado]["n_conf"] += 1
            por_classe[esperado]["conf"] += d.confidence
        por_classe[esperado]["total"] += 1
        if acerto:
            por_classe[esperado]["hit"] += 1

        linhas.append(
            f"| `{path.name}` | {esperado} | {'✅' if acerto else '❌'} | {' · '.join(nomes[:6])} |"
        )

    total = sum(s["total"] for s in por_classe.values())
    hits = sum(s["hit"] for s in por_classe.values())
    linhas += [
        "",
        "## Resumo",
        "",
        "| Classe esperada | Imagens | Acertos | Taxa | Confiança média |",
        "|---|---|---|---|---|",
    ]
    for nome in sorted(por_classe):
        s = por_classe[nome]
        taxa = s["hit"] / s["total"] if s["total"] else 0.0
        conf = s["conf"] / s["n_conf"] if s["n_conf"] else 0.0
        linhas.append(f"| {nome} | {int(s['total'])} | {int(s['hit'])} | {taxa:.0%} | {conf:.2f} |")

    linhas += [
        "",
        f"**Taxa de acerto global: {hits}/{total} ({hits / total:.0%})**",
        f"- Imagens sem nenhuma detecção: **{sem_deteccao}/{total}**",
        f"- Classes distintas detectadas: **{len(classes_detectadas)}/13** "
        f"({', '.join(sorted(classes_detectadas))})",
        f"- Cópias anotadas (para o vídeo) em `{args.annotated}/` — a API não devolve imagem.",
        "",
        "> Números medidos por `scripts/eval_external.py`. Sem anotações humanas estas",
        "> imagens, por isso reportamos acerto da classe esperada e confianças — não mAP.",
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
