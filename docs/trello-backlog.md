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

## 2. Coluna atual (estado em 20/10/2026 — espelhe no Trello)

| Coluna | Cartões |
|---|---|
| backlog | Fine-tuning opcional · Otimização de latência · Publicação em nuvem |
| todo | Validação em imagens variadas (fotos externas) · Gravação do vídeo · Entrega no AVA |
| doing | _(nada parado — mova o card assim que iniciar)_ |
| blocked | _(vazia — mantenha a coluna mesmo assim)_ |
| review | README completo (aguardando revisão final) |
| done | Definição do escopo (trilha C, 13 classes) · Setup GitFlow · Ambiente Python 3.12 e dependências · Estrutura da API · Endpoint `POST /detect` · Testes pytest (14) · Quadro Kanban público · Baixar/preparar dataset (1.160 imagens) · Relatório de contagens e splits sem vazamento · Avaliação medida no split de teste |

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

**`[Rafael Cunha]` Baixar e preparar o dataset (13 classes)**
- **Especificação:** `python scripts/prepare_dataset.py --download` — baixa `coco2017labels-segments.zip` (169 MB) + `val2017.zip` (778 MB), filtra as 13 classes de supermercado (IDs 39–55 do COCO), descarta anotações fora da seleção e monta `data/coco-subset/`.
- **Critério de pronto:** pastas `images/{train,val,test}` e `labels/{train,val,test}` existem; `data.yaml` gerado com os 80 `names` do COCO na posição original.
- **Risco conhecido:** download de ~1 GB (retomável).

**`[Rafael Cunha]` Relatório de contagens e splits sem vazamento**
- **Especificação:** gerar `dataset_report.md` com nº de imagens e instâncias por classe; confirmar split 70/15/15 com seed 42 e que nenhuma imagem aparece em dois splits.
- **Critério de pronto:** tabela das 13 classes preenchida no README e nenhuma classe com 0 imagens.

**`[Rafael Cunha]` Validação do modelo em imagens variadas**
- **Especificação:** coletar ~20 imagens externas ao COCO (celular, luz baixa, ângulo, oclusão, fundo de prateleira) e rodar `POST /detect` em todas.
- **Critério de pronto:** tabela no README/video com classe → confiança observada; evidência de que as 13 classes são reconhecidas com clareza.
- **Por que:** rubrica 7 exige "testou com imagens variadas" — é exigido no vídeo.

**`[Rafael Cunha]` README completo**
- **Especificação:** nome do software, dataset + 13 classes, justificativa do YOLO11n, instalação, execução (`uvicorn main:app --reload`), payload de exemplo real, variáveis de ambiente, link do Kanban.
- **Critério de pronto:** todas as seções exigidas preenchidas com dados reais (sem placeholder).

**`[Rafael Cunha]` Gravação do vídeo (até 7 min)**
- **Especificação:** cobrir dataset → demonstração no Swagger/Postman → processo do YOLO e generalização → testes do dataset → Trello → branches do GitHub.
- **Critério de pronto:** vídeo publicado com link acessível sem login.

**`[Rafael Cunha]` Entrega no AVA**
- **Especificação:** submeter links de GitHub, vídeo e Kanban na tarefa *Módulo 2 - Projeto Avaliativo*.
- **Critério de pronto:** envio confirmado **até 03/11/2026 às 22h**; repositório não alterado depois disso.

### DOING

**`[Rafael Cunha]` Setup do repositório GitFlow**
- **Especificação:** `git init` em `main`, branch `develop`, features `feature/project-scaffold`, `feature/dataset-prep`, `feature/api-setup`, `feature/docs-delivery`; commits no padrão `tipo(escopo): descrição`; merge de tudo em `main` ao final.
- **Critério de pronto:** log com branches e commits curtos; `.gitignore` cobrindo `data/`, `runs/`, `*.pt`, `.venv`.

**`[Rafael Cunha]` Ambiente Python 3.12 e dependências fixadas**
- **Especificação:** `python3.12 -m venv .venv` (o `python3` padrão é 3.15 e o torch não tem wheel `cp315`) + `requirements.txt` com versões pinadas.
- **Critério de pronto:** `pip install -r requirements.txt` limpo e `import torch` com `mps available: True`.

**`[Rafael Cunha]` Estrutura da API e configuração por ambiente**
- **Especificação:** `main.py` + `app/{api,schemas}.py`, `app/core/config.py` com `pydantic-settings` (`MODEL_PATH`, `CONF`, `IOU`, `DEVICE`, `SCOPE`, `MAX_UPLOAD_MB`), modelo carregado **uma única vez** no `lifespan`.
- **Critério de pronto:** `/health` responde `status: ok` e nenhum request recarrega os pesos.

**`[Rafael Cunha]` Endpoint POST /detect com UploadFile**
- **Especificação:** recebe `UploadFile`, valida content-type/tamanho, devolve `{status, detections[{class_name, confidence, box{x_min,y_min,x_max,y_max}}]}` em pixels da imagem original, sem imagem desenhada; erros 400/413/503 via `HTTPException`.
- **Critério de pronto:** contrato exato validado por teste e demonstrado no Swagger.

**`[Rafael Cunha]` Testes automatizados (pytest)**
- **Especificação:** `tests/test_api.py` (contrato, escopo, content-type, bytes inválidos, arquivo vazio, limite de tamanho, modelo não carregado, sem detecções) + `tests/test_integration.py` (modelo real).
- **Critério de pronto:** `pytest` verde.

**`[Rafael Cunha]` Criação do quadro Kanban público**
- **Especificação:** 6 colunas obrigatórias, cartões com nome do aluno e especificações, quadro público com link no README.
- **Critério de pronto:** link do quadro funcionando em aba anônima.

### BLOCKED

_Quando algo travar (ex.: download do dataset caindo, espaço em disco), mova o card
para cá com o motivo no comentário. Volte para `doing` quando destravar._

### REVIEW

_Cartões prontos e aguardando conferência (sua ou de um colega) antes de ir para `done`._

### DONE

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
