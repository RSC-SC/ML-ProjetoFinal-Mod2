#!/usr/bin/env python3
"""Prepara o dataset de supermercado a partir do COCO 2017 (val2017).

O que faz:
  1. Baixa os labels YOLO do COCO (`coco2017labels-segments.zip`, ~169 MB) e as
     imagens de validação (`val2017.zip`, ~815 MB) — apenas se ainda não existirem.
  2. Seleciona apenas as imagens que contêm uma das 13 classes de supermercado.
  3. Descarta das anotações as classes fora da seleção (fica um dataset limpo).
  4. Faz o split train/val/test (70/15/15) por imagem, com seed fixa.
  5. Gera o `data.yaml` e um relatório de contagem por classe.

Por que os IDs de classe do COCO são preservados (não são remapeados para 0..12):
o modelo pré-treinado devolve IDs do COCO (0..79). Se remapearmos para 0..12,
o `yolo val` compararia classes erradas e produziria métricas falsas, e a API
teria que traduzir a saída. Mantendo os IDs originais, a inferência zero-shot
e a avaliação ficam corretas sem nenhuma camada extra.

Uso:
    python scripts/prepare_dataset.py --download
    python scripts/prepare_dataset.py --download --limit 200   # smoke test
"""
from __future__ import annotations

import argparse
import collections
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

# URLs verificadas em 2026-10-20 (tamanhos confirmados por HEAD request).
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"  # 176.694.178 B
VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"  # 815.585.330 B

# Classes compartilhadas com a API — fonte única de verdade em app/core/classes.py
# (80 classes do COCO na posição original + as 13 de supermercado).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.classes import COCO_NAMES, SELECTED_IDS  # noqa: E402

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path) -> None:
    """Baixa `url` para `dest` de forma resumível (retoma se o arquivo parcial existir)."""
    import urllib.request

    if dest.exists():
        log(f"  já existe, pulando: {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        log(f"  retomando {dest.name} a partir de {existing / 1e6:.0f} MB")
    try:
        with urllib.request.urlopen(req) as r, open(tmp, "ab") as f:
            total = int(r.headers.get("Content-Length", 0)) + existing
            done = existing
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {dest.name}: {done / 1e6:6.0f}/{total / 1e6:.0f} MB ({pct:4.1f}%)", end="", flush=True)
        print()
    except Exception as exc:  # noqa: BLE001
        if existing and tmp.exists():
            log(f"\n  falha ao retomar ({exc}); reiniciando do zero")
            tmp.unlink(missing_ok=True)
            return download(url, dest)
        raise
    tmp.rename(dest)


def parse_label_lines(text: str) -> list[tuple[int, str]]:
    """Lê um arquivo .txt YOLO e devolve (id_classe, linha_original) por anotação."""
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            cls_id = int(line.split()[0])
        except (ValueError, IndexError):
            continue
        out.append((cls_id, line))
    return out


def assign_splits(stems: list[str], rng: random.Random) -> dict[str, str]:
    """Divide os stems em train/val/test respeitando as proporções de SPLITS."""
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * SPLITS["train"])
    n_val = round(n * SPLITS["val"])
    mapping: dict[str, str] = {}
    for i, stem in enumerate(shuffled):
        if i < n_train:
            mapping[stem] = "train"
        elif i < n_train + n_val:
            mapping[stem] = "val"
        else:
            mapping[stem] = "test"
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/_raw", help="onde baixar os zips do COCO")
    parser.add_argument("--out-dir", default="data/coco-subset", help="raiz do dataset gerado")
    parser.add_argument("--download", action="store_true", help="baixa os arquivos se não existirem")
    parser.add_argument("--limit", type=int, default=0, help="processa só N imagens (smoke test)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    labels_zip = raw / "coco2017labels-segments.zip"
    images_zip = raw / "val2017.zip"

    if args.download:
        raw.mkdir(parents=True, exist_ok=True)
        log("Baixando artefatos do COCO (só o que falta):")
        download(LABELS_URL, labels_zip)
        download(VAL_IMAGES_URL, images_zip)
    for path in (labels_zip, images_zip):
        if not path.exists():
            log(f"ERRO: {path} não existe. Rode com --download.")
            return 1

    rng = random.Random(args.seed)

    log("Lendo anotações do COCO...")
    selected = set(SELECTED_IDS)
    # stem -> linhas de anotação das classes selecionadas
    kept_labels: dict[str, list[str]] = {}
    total_label_files = 0
    with zipfile.ZipFile(labels_zip) as zf:
        # Estrutura real do zip: coco/labels/val2017/<id>.txt (não labels/val2017/...).
        label_names = [
            n for n in zf.namelist() if re.fullmatch(r".*labels/val2017/\d{12}\.txt", n)
        ]
        total_label_files = len(label_names)
        for name in label_names:
            stem = Path(name).stem
            anns = parse_label_lines(zf.read(name).decode("utf-8", "replace"))
            kept = [line for cls_id, line in anns if cls_id in selected]
            if kept:
                kept_labels[stem] = kept
            if args.limit and len(kept_labels) >= args.limit:
                break

    log(f"  {total_label_files} arquivos de anotação no COCO val2017")
    log(f"  {len(kept_labels)} imagens contêm ao menos uma das {len(SELECTED_IDS)} classes")

    stems = sorted(kept_labels)
    splits = assign_splits(stems, rng)
    log(f"  split {SPLITS} (seed={args.seed}): "
        + ", ".join(f"{k}={sum(1 for v in splits.values() if v == k)}" for k in SPLITS))

    # Estrutura de pastas no formato que o Ultralytics espera.
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    log("Extraindo imagens e gravando anotações filtradas...")
    extracted = 0
    with zipfile.ZipFile(images_zip) as zf:
        available = set(zf.namelist())
        for stem, split in splits.items():
            img_name = f"val2017/{stem}.jpg"
            if img_name not in available:
                continue
            with zf.open(img_name) as src, open(out / "images" / split / f"{stem}.jpg", "wb") as dst:
                shutil.copyfileobj(src, dst)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(kept_labels[stem]) + "\n", encoding="utf-8"
            )
            extracted += 1
    log(f"  {extracted} imagens extraídas para {out}")

    # data.yaml com os 80 names do COCO na posição original.
    names_block = "\n".join(f"  {i}: {COCO_NAMES[i]}" for i in sorted(COCO_NAMES))
    data_yaml = (
        "path: ./\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "# IDs de classe do COCO preservados (0..79) para casar com o modelo\n"
        "# pré-treinado. Só as imagens/linhas das 13 classes foram mantidas.\n"
        "names:\n"
        f"{names_block}\n"
    )
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    log(f"  data.yaml gravado em {out / 'data.yaml'}")

    # Relatório de contagem por classe (imagens e instâncias).
    img_count: collections.Counter[int] = collections.Counter()
    inst_count: collections.Counter[int] = collections.Counter()
    for stem, lines in kept_labels.items():
        seen: set[int] = set()
        for line in lines:
            cls_id = int(line.split()[0])
            inst_count[cls_id] += 1
            seen.add(cls_id)
        img_count.update(seen)

    report = [
        "# Relatório do dataset (supermercado / COCO subset)",
        "",
        f"- Fonte: COCO 2017 `val2017` ({total_label_files} imagens anotadas no total)",
        f"- Classes selecionadas: {len(SELECTED_IDS)}",
        f"- Imagens no subset: {len(kept_labels)} (extraídas: {extracted})",
        f"- Split: train {SPLITS['train']:.0%} / val {SPLITS['val']:.0%} / test {SPLITS['test']:.0%} (seed {args.seed})",
        "- IDs de classe: preservados do COCO (0..79) para compatibilidade com o modelo pré-treinado",
        "",
        "| ID | Classe | Imagens | Instâncias |",
        "|---|---|---|---|",
    ]
    for cls_id in SELECTED_IDS:
        report.append(f"| {cls_id} | {COCO_NAMES[cls_id]} | {img_count[cls_id]} | {inst_count[cls_id]} |")
    report += [
        "",
        f"**Total de instâncias**: {sum(inst_count.values())}",
        "",
        "Gerado por `scripts/prepare_dataset.py` — não editar manualmente.",
    ]
    (out / "dataset_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"  relatório gravado em {out / 'dataset_report.md'}")

    log("\nClasses sem imagens (verifique antes de treinar): "
        + (", ".join(COCO_NAMES[i] for i in SELECTED_IDS if not img_count[i]) or "nenhuma"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
