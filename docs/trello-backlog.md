# Backlog do Kanban (Trello) — referência do quadro

Quadro público: <https://trello.com/b/esPArbhZ/scanvision>

Este arquivo guarda o conteúdo dos cartões (título com o nome do aluno, especificação e
critério de pronto). Mantenha-o sincronizado com o Trello: **depois de mover um card no
quadro, refleta o movimento na seção 2 abaixo)**. A rubrica
avalia a **movimentação correta** (nada fica eternamente em `doing`).

## 1. Como montar o quadro

1. Trello → **Criar quadro**: nome `ScanVision — Projeto Módulo 2 (SCTEC)`.
2. Adicione as **6 colunas nesta ordem**: `backlog` · `todo` · `doing` · `blocked` · `review` · `done`.
3. Torne o quadro **público** (Menu → Visibilidade → Público) e copie o link para o README.
4. Em cada cartão: **o nome do aluno no título** e a especificação na descrição.

> ✅ Quadro já criado: <https://trello.com/b/esPArbhZ/scanvision> — todos os cartões
> estão com o nome **Rafael Cunha** no título. Este arquivo é a **cópia de referência**
> dos cartões; se você for clonar o projeto, use a seção 3 como modelo.

## 2. Coluna atual (estado em 21/10/2026 — espelhe no Trello)

| Coluna | Cartões |
|---|---|
| backlog | Fine-tuning opcional · Otimização de latência · Publicação em nuvem |
| todo | Gravação do vídeo · Entrega no AVA |
| doing | _(vazia — nada parado)_ |
| blocked | _(vazia — mantenha a coluna mesmo assim)_ |
| review | _(vazia — tudo o que estava em revisão foi conferido)_ |
| done | Definição do escopo (trilha C, 13 classes) · Setup GitFlow · Ambiente Python 3.12 e dependências · Estrutura da API · Endpoint `POST /detect` · Testes pytest (18) · Quadro Kanban público · Baixar/preparar dataset (1.160 imagens) · Relatório de contagens e splits sem vazamento · Avaliação medida no split de teste (13/13 classes) · Validação em imagens variadas (38 imagens externas, 13/13 classes) · README completo · Demo com vídeo (`scripts/detect_video.py`) · Interface de demonstração (frontend que desenha as caixas) |

---

## 3. Cartões

### BACKLOG

**`[Rafael Cunha]` Fine-tuning opcional do YOLO**
- **Especificação:** rodar `scripts/train.py --epochs 30` sobre `data/coco-subset`, `device=mps`, `seed=42`, `batch=16`, `imgsz=640`, `patience=10`. Salvar em `runs/detect/` (gitignorado).
- **Critério de pronto:** `mAP50`, `mAP50-95`, precision e recall registrados no README; API apontando para `best.pt` via `MODEL_PATH` e detectando as mesmas 13 classes.
- **Por que está no backlog:** o enunciado tornou o fine-tuning **opcional**.

**`[Rafael Cunha]` Otimização de latência da inferência**
- **Especificação:** medir tempo por request (log em `/detect`), testar exportação ONNX e `imgsz` 320/640/1280.
- **Critério de pronto:** número médio de ms por inferência documentado (MPS).
- **Nota:** só depois da entrega — não bloqueia nada.

**`[Rafael Cunha]` Publicação da API em nuvem**
- **Especificação:** container/uvicorn gerenciado com upload de imagem.
- **Critério de pronto:** URL pública respondendo `/health`.
- **Nota:** o enunciado não exige deploy (backend-only) → opcional.

### TODO

**`[Rafael Cunha]` Gravação do vídeo (até 7 min)**
- **Especificação:** cobrir dataset → demonstração no Swagger/Postman → processo do YOLO e generalização → testes do dataset → Trello → branches do GitHub. O frontend (`GET /`) pode entrar como bônus depois do Swagger.
- **Critério de pronto:** vídeo publicado com link acessível sem login.

**`[Rafael Cunha]` Entrega no AVA**
- **Especificação:** submeter links de GitHub, vídeo e Kanban na tarefa *Módulo 2 - Projeto Avaliativo*.
- **Critério de pronto:** envio confirmado **até 03/11/2026 às 22h**; repositório não alterado depois disso.

### DOING

_Vazia. Cards concluídos passaram por `doing` → `review` → `done` com a evidência registrada._

### BLOCKED

_Quando algo travar (ex.: download do dataset caindo, espaço em disco), mova o card
para cá com o motivo no comentário. Volte para `doing` quando destravar._

### REVIEW

_Vazia. Nesta entrega individual a conferência é a própria suíte `pytest` (18 testes) +
validação no Swagger antes de mover para `done`._

### DONE

**`[Rafael Cunha]` Setup do repositório GitFlow**
- **Especificação:** `main`, `develop` e `feature/*` (uma por entrega), commits no padrão `tipo(escopo): descrição`, merge de tudo em `main` ao final.
- **Evidência:** 13 branches (`feature/project-scaffold`, `dataset-prep`, `dataset-validation`, `api-setup`, `yolo-finetuning`, `identity-links`, `external-validation`, `repo-rules`, `video-demo`, `docs-delivery`, `frontend`) + `main` e `develop`; `.gitignore` cobre `data/`, `runs/`, `*.pt`, `.venv`; tudo pushado.

**`[Rafael Cunha]` Ambiente Python 3.12 e dependências fixadas**
- **Especificação:** `python3.12 -m venv .venv` + `requirements.txt` pinado, `device=mps`.
- **Evidência:** `/health` responde `device: mps`; instalação limpa registrada no README.

**`[Rafael Cunha]` Estrutura da API e configuração por ambiente**
- **Especificação:** `main.py` + `app/{api,schemas}.py` + `app/core/config.py` (`pydantic-settings`), modelo carregado uma única vez no `lifespan`.
- **Evidência:** `GET /health` → `{"status":"ok","model":"yolo11n.pt","device":"mps","scope":"dataset"}`.

**`[Rafael Cunha]` Endpoint POST /detect com UploadFile**
- **Especificação:** `UploadFile`, validação de content-type/tamanho, JSON do enunciado em pixels, erros 400/413/503.
- **Evidência:** `examples/payload_real.json` gerado por `scripts/demo_payload.py`; contrato validado por teste (`set(body) == {"status","detections"}`) e demonstrado no Swagger.

**`[Rafael Cunha]` Testes automatizados (pytest)**
- **Especificação:** unitários com detector falso + integração com o modelo real.
- **Evidência:** **18 testes verdes** (`pytest -q`), incluindo contrato exato, escopo, content-type, arquivo vazio, limite de tamanho, modelo não carregado, frontend e CORS.

**`[Rafael Cunha]` Criação do quadro Kanban público**
- **Especificação:** 6 colunas obrigatórias, cartões com nome do aluno, quadro público com link no README.
- **Evidência:** <https://trello.com/b/esPArbhZ/scanvision> referenciado no README.

**`[Rafael Cunha]` Baixar e preparar o dataset (13 classes)**
- **Especificação:** `scripts/prepare_dataset.py --download` (COCO 2017 `val2017` + labels), filtro das 13 classes, split 70/15/15 com seed 42.
- **Evidência:** 1.160 imagens e 5.323 instâncias em `data/coco-subset/`, `data.yaml` com os 80 `names` na posição original.

**`[Rafael Cunha]` Relatório de contagens e splits sem vazamento**
- **Especificação:** `dataset_report.md` com imagens/instâncias por classe e verificação de vazamento.
- **Evidência:** `data/coco-subset/dataset_report.md`; splits 812/174/174 sem imagem repetida entre splits.

**`[Rafael Cunha]` Validação do modelo em imagens variadas**
- **Especificação:** imagens externas ao COCO rodando em `POST /detect`.
- **Evidência:** 38 imagens do Wikimedia Commons com atribuição (`examples/external/ATTRIBUICAO.md`), 13/13 classes detectadas, acerto da classe esperada 22/38 → `docs/external-report.md`.

**`[Rafael Cunha]` README completo**
- **Especificação:** nome, dataset e classes, justificativa do YOLO11n, instalação, `uvicorn main:app --reload`, payload real, variáveis de ambiente, Kanban e GitFlow.
- **Evidência:** `README.md` com todas as seções (inclui avaliação medida, generalização e a seção do frontend).

**`[Rafael Cunha]` Avaliação medida no split de teste**
- **Evidência:** 13/13 classes, precisão 0,70 / recall 0,42 / F1 0,52 @ `conf=0.25` (IoU ≥ 0,5) → `docs/evaluation-report.md`; sensibilidade a `conf=0.10` em `docs/evaluation-report-conf010.md`.

**`[Rafael Cunha]` Demo com vídeo (`scripts/detect_video.py`)**
- **Evidência:** vídeo sintético de 24 quadros → 8 enviados, detecções em 7, 6 classes, latência mediana 11 ms/quadro no MPS.

**`[Rafael Cunha]` Interface de demonstração (frontend que desenha as caixas)**
- **Especificação:** página estática em `frontend/` (HTML/CSS/JS sem build) servida pela própria API em `GET /`; envia a imagem em `multipart/form-data` para `POST /detect` e desenha as bounding boxes no `<canvas>` a partir do JSON, sem alterar o contrato do endpoint.
- **Critério de pronto:** upload por drag&drop e por seletor; caixas com cor por classe e rótulo `classe + confiança`; tabela de detecções e JSON bruto exibidos; rotas da API não sombreadas pelo mount (`tests/test_frontend.py`).
- **Evidência medida (20/10/2026):** imagem `examples/external/orange_2.jpg` → 16 detecções, tabela e JSON renderizados, 44 ms (primeira execução) e 56 ms (repetição) de ida e volta com o modelo aquecido, zero erros no console.

**`[Rafael Cunha]` Definição do escopo e da trilha de classes**
- **Especificação:** decisão de usar as **13 classes de supermercado do COCO** (IDs 39–55) com modelo **pré-treinado YOLO11n**, sem fine-tuning obrigatório (opcional no enunciado), trabalho individual.
- **Evidência:** `app/core/classes.py` (fonte única) + `AGENTS.md`.

---

## 4. Regra de movimentação (o que a rubrica olha)

| Transição | Quando |
|---|---|
| `backlog` → `todo` | entrou na prioridade desta semana |
| `todo` → `doing` | alguém começou a trabalhar nele |
| `doing` → `blocked` | surgiu impedimento (descreva o motivo no card) |
| `blocked` → `doing` | impedimento resolvido |
| `doing` → `review` | entrega feita, aguardando conferência |
| `review` → `done` | conferido e aceito (com evidência na descrição) |

Nada deve pular direto de `todo` para `done`.
