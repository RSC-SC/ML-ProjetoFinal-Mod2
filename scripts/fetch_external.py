#!/usr/bin/env python3
"""Baixa imagens de fonte livre (Wikimedia Commons) para testar generalização.

São imagens **fora da distribuição do COCO** — o modelo nunca as viu. Elas sustentam
a parte de generalização exigida no vídeo.

Fonte: Wikimedia Commons (CC0 / CC BY / CC BY-SA, variadas por arquivo). Toda atribuição
fica em `examples/external/ATTRIBUICAO.md`; os metadados (incluindo a posição escolhida
na categoria) ficam em `examples/external/atribuicao.json`, então a seleção é reproduzível.

Categorias do Commons são ruidosas (uma categoria "Pizzas" tem fotos de geladeira e de
cartaz), por isso a seleção é feita **pelo título do arquivo** com `--dry-run` + `--positions`.

Uso:
    python scripts/fetch_external.py --dry-run            # só lista candidatos
    python scripts/fetch_external.py                      # completa o que falta (posições 1..N)
    python scripts/fetch_external.py --only pizza donut --positions 5,12
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.classes import COCO_NAMES, SELECTED_IDS  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"

# Uma categoria do Commons por classe do projeto (nome em inglês).
CATEGORIES: dict[int, str] = {
    39: "Bottles",          # bottle
    41: "Cups",             # cup
    45: "Bowls",            # bowl
    46: "Bananas",          # banana
    47: "Apples",           # apple
    48: "Sandwiches",       # sandwich
    49: "Oranges",          # orange
    50: "Broccoli",         # broccoli
    51: "Carrots",          # carrot
    52: "Hot dogs",         # hot dog
    53: "Pizzas",           # pizza
    54: "Doughnuts",        # donut
    55: "Cakes",            # cake
}

TAG_RE = re.compile(r"<[^>]+>")


def strip_html(value: str) -> str:
    """Remove tags HTML do campo de crédito do Commons."""
    return TAG_RE.sub("", html.unescape(value or "")).strip()


def get_json(client: httpx.Client, params: dict, tries: int = 4) -> dict:
    """GET com retry/backoff — o Commons responde 429 se batermos rápido demais."""
    for attempt in range(1, tries + 1):
        resp = client.get(API, params=params, timeout=60.0)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        time.sleep(3 * attempt)
    resp.raise_for_status()
    return {}


def list_category(category: str, width: int, client: httpx.Client) -> list[dict]:
    """Lista os candidatos (só imagem raster) de uma categoria do Commons."""
    data = get_json(
        client,
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "40",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": str(width),
        },
    )
    pages = (data.get("query") or {}).get("pages") or {}

    candidatos: list[dict] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        mime = info.get("mime", "")
        if "jpeg" not in mime and "png" not in mime:
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        meta = info.get("extmetadata") or {}
        candidatos.append(
            {
                "title": page.get("title", ""),
                "url": url,
                "page": info.get("descriptionurl", ""),
                "author": strip_html((meta.get("Artist") or {}).get("value", "")) or "desconhecido",
                "license": strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
                or "desconhecida",
            }
        )
    return candidatos


def write_credits(out_dir: Path, atribuicao: dict[str, dict]) -> None:
    """Regenera ATTRIBUICAO.md a partir do sidecar (nunca perde crédito)."""
    lines = [
        "# Atribuição das imagens externas",
        "",
        "Baixadas do **Wikimedia Commons** pelo `scripts/fetch_external.py`. Cada imagem",
        "mantém a licença original do arquivo. Reuso deste material exige manter a",
        "atribuição abaixo (licenças CC BY / CC BY-SA).",
        "",
        "| Arquivo | Classe | Autor | Licença | Origem |",
        "|---|---|---|---|---|",
    ]
    for filename in sorted(atribuicao):
        a = atribuicao[filename]
        lines.append(
            f"| `{filename}` | {a['classe']} | {a['autor'] or 'desconhecido'} | {a['licenca']} "
            f"| [Commons]({a['pagina']}) |"
        )
    (out_dir / "ATTRIBUICAO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def next_index(out_dir: Path, prefix: str) -> int:
    """Próximo índice livre para `<classe>_<n>.jpg` (não sobrescreve nada)."""
    idxs = [
        int(m.group(1))
        for f in out_dir.glob(f"{prefix}_*.jpg")
        if (m := re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.jpg", f.name))
    ]
    return max(idxs, default=0) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--width", type=int, default=800, help="largura máxima da cópia baixada")
    parser.add_argument("--out", default="examples/external")
    parser.add_argument("--delay", type=float, default=1.5, help="pausa entre chamadas à API")
    parser.add_argument("--only", nargs="*", default=None, help="só estas classes (em inglês)")
    parser.add_argument(
        "--positions", default=None,
        help="posições (1-based) na categoria, separadas por vírgula; ex.: 5,12",
    )
    parser.add_argument("--dry-run", action="store_true", help="lista os candidatos sem baixar")
    parser.add_argument("--force", action="store_true", help="baixa mesmo com a classe completa")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = out_dir / "atribuicao.json"
    atribuicao: dict[str, dict] = (
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    )

    alvo = list(SELECTED_IDS)
    if args.only:
        desejadas = {n.lower().replace(" ", "_") for n in args.only}
        alvo = [i for i in SELECTED_IDS if COCO_NAMES[i].replace(" ", "_") in desejadas]
        if not alvo:
            print(f"ERRO: nenhuma classe corresponde a {args.only}")
            return 1

    posicoes = (
        [int(p) for p in args.positions.split(",")] if args.positions else None
    )

    baixadas, puladas, falhas = 0, 0, []
    with httpx.Client(
        headers={"User-Agent": "ScanVision-SCTEC/1.0 (projeto academico)"}
    ) as client:
        for cls_id in alvo:
            name = COCO_NAMES[cls_id]
            prefix = name.replace(" ", "_")
            ja_tem = [f for f in atribuicao if f.startswith(f"{prefix}_")]

            if args.dry_run:
                candidatos = list_category(CATEGORIES[cls_id], args.width, client)
                print(f"\n== {name} (Category:{CATEGORIES[cls_id]}) — {len(candidatos)} candidatos")
                for i, c in enumerate(candidatos[:20], start=1):
                    print(f"  {i:2d}. {c['title'][5:90]}")
                continue

            if not args.force and not posicoes and len(ja_tem) >= args.per_class:
                puladas += 1
                continue

            try:
                candidatos = list_category(CATEGORIES[cls_id], args.width, client)
            except httpx.HTTPError as exc:
                falhas.append(f"{name}: HTTP {exc}")
                continue
            if not candidatos:
                falhas.append(f"{name}: categoria '{CATEGORIES[cls_id]}' sem candidatos")
                continue

            if posicoes:
                escolhidos = [candidatos[p - 1] for p in posicoes if 0 < p <= len(candidatos)]
            else:
                faltando = max(0, args.per_class - len(ja_tem))
                escolhidos = candidatos[: max(faltando, 0)]

            idx = next_index(out_dir, prefix)
            for file in escolhidos:
                dest = out_dir / f"{prefix}_{idx}.jpg"
                try:
                    time.sleep(args.delay)
                    data = client.get(file["url"], timeout=120.0).content
                except httpx.HTTPError as exc:
                    falhas.append(f"{name}: falha ao baixar ({exc})")
                    continue
                if len(data) < 5_000:
                    falhas.append(f"{name}: arquivo pequeno demais ({file['title'][:50]})")
                    continue
                dest.write_bytes(data)
                atribuicao[dest.name] = {
                    "classe": name,
                    "titulo": file["title"],
                    "autor": file["author"],
                    "licenca": file["license"],
                    "pagina": file["page"],
                }
                baixadas += 1
                idx += 1
                print(f"  {dest.name:20s} <- {file['title'][:60]} [{file['license']}]")

            sidecar.write_text(
                json.dumps(atribuicao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            write_credits(out_dir, atribuicao)

    if not args.dry_run:
        sidecar.write_text(
            json.dumps(atribuicao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_credits(out_dir, atribuicao)
        print(f"\n{baixadas} baixadas · {puladas} classes já completas · {len(atribuicao)} no total")
    if falhas:
        print("Problemas:")
        for f in falhas:
            print(f"  - {f}")
    return 0 if (args.dry_run or atribuicao) else 1


if __name__ == "__main__":
    raise SystemExit(main())
