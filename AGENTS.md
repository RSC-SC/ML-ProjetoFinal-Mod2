# AGENTS.md

Projeto avaliativo do Módulo 2 (SCTEC) — detecção de objetos com YOLO + API FastAPI.
Especificação vigente: `ProjetoFinalModulo2-Descritivo.md` + `Machine Learning e Visão Computacional [T1] - Projeto Avaliativo-2.pdf` (oficial, mesmo diretório — o PDF não é legível por ferramentas do repo).

## Escopo (revisado em 2026-10-20 — substitui o enunciado anterior)
- **Fine-tuning é opcional.** O desafio aceita "utilizar um modelo pré-treinado, **ou** aplicar fine-tuning". Na rubrica, 3,00 = "construção do modelo YOLO com sucesso em um dataset com ≥10 classes e testado com imagens variadas"; o fine-tuning aparece como opcional. 1,00 = "não reconhece as 10 classes com clareza ou não justificou a escolha da arquitetura".
- Continua **obrigatório**: dataset com ≥10 classes, arquitetura YOLO, `POST /detect` com `UploadFile`, JSON estruturado (sem imagem desenhada), Kanban (backlog/todo/doing/blocked/review/done), GitFlow (`main`/`develop`/`feature/*`), README e vídeo de até 7 min.
- Prazo: **03/11/2026 22h** no AVA (era 02/11 15h). Não alterar o repo depois da entrega; plágio = 0.
- Trabalho **individual** → benchmark entre versões do YOLO não é obrigatório.

## Decisões tomadas (2026-10-20)
- **Software:** `ScanVision` (título do FastAPI, logger e `--name` do run de fine-tuning).
- **Trilha C — supermercado:** 13 classes do COCO (IDs 39, 41, 45–55) sobre o `val2017`.
- **Modelo:** `yolo11n.pt` pré-treinado, **sem fine-tuning** (`scripts/train.py` fica pronto para o opcional).
- **Dataset:** `data/coco-subset/` — 1.160 imagens, 5.323 instâncias, split 812/174/174 (seed 42, sem vazamento). Gerado por `scripts/prepare_dataset.py --download`; labels do zip vêm em `coco/labels/val2017/` (não `labels/val2017/`) e em formato **YOLO-segmentação** (polígono), não bbox.
- **Resultado medido:** 13/13 classes no split de teste (precisão 0,70 / recall 0,42 @`conf=0.25`, IoU ≥ 0,5) e 13/13 em 38 imagens externas do Wikimedia Commons (acerto da classe esperada 22/38). Relatórios: `docs/evaluation-report*.md` e `docs/external-report.md`.
- **Repositório:** <https://github.com/RSC-SC/ML-ProjetoFinal-Mod2> — GitFlow completo **local**; push pendente de `gh auth login` do aluno.
- **Kanban:** <https://trello.com/b/esPArbhZ/scanvision> — espelho dos cartões em `docs/trello-backlog.md`.

## O que falta
- `gh auth login` + push das branches (main, develop e 8 `feature/*`).
- Gravar o vídeo (até 7 min) e entregar no AVA até **03/11/2026 22h**.

## Ambiente (macOS, Apple Silicon, M5 Pro 24 GB)
- `python3` do PATH é 3.15 (homebrew) e **torch não publica wheel cp315** (2.14.1: cp310–cp314). Use `python3.12` (`/opt/homebrew/bin/python3.12`) para criar o venv.
- `uv` e `poetry` não instalados → `python3.12 -m venv .venv` + `pip`.
- Sem CUDA: treino/inferência com `device=mps` no ultralytics (nunca `device=0`).

## Armadilha do caminho sem fine-tuning
- Um YOLO pré-treinado no COCO só reconhece as **80 classes do COCO** (IDs 0–79, lista em `ultralytics/cfg/datasets/coco.yaml`). Se as 10 classes escolhidas não estiverem nessa lista, a API retorna `detections: []` e a nota cai ao nível 1,00 da rubrica.
- Ao criar um subconjunto do COCO, **mantenha os 80 `names` na posição original** no `data.yaml` e apenas filtre quais imagens entram. Remapear para 0..9 quebra a avaliação (`yolo val` compararia IDs errados → métricas falsas).
- Tamanhos verificados (2026-10-20): `val2017.zip` 815.585.330 B (~0,8 GB), `coco2017labels-segments.zip` 176.694.178 B (~169 MB, labels YOLO), `annotations_trainval2017.zip` 252.907.541 B (~241 MB), `train2017.zip` 19.336.861.798 B (~19,3 GB — evitar).

## Regras do entregável (é o que é avaliado)
- Dataset com **≥10 classes**, preparação própria: contagem de imagens/instâncias por classe, splits sem vazamento e `data.yaml` documentados.
- Reporte apenas métricas medidas; nunca estime valores de mAP/precisão.
- `POST /detect` recebendo `UploadFile` e devolvendo JSON exatamente com `status` + `detections[].{class_name, confidence, box{x_min,y_min,x_max,y_max}}`. Não retornar imagem anotada.
- README precisa conter `uvicorn main:app --reload` e um payload de exemplo real.
- `.gitignore` desde o primeiro commit: nunca commitar dataset, `runs/`, pesos (`*.pt`) nem `.venv`.

## Agente `ml-vision`
- Definição em `.opencode/agents/ml-vision.md`, com **cópia global** em `~/.config/opencode/agents/ml-vision.md`. Editar uma não atualiza a outra — sincronize com `cp .opencode/agents/ml-vision.md ~/.config/opencode/agents/`.
