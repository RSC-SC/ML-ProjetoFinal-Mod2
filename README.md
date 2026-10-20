# ScanVision — Detecção de Objetos em Imagens de Supermercado

> **Nome do software:** `ScanVision` — detecta produtos em imagens de supermercado.

Sistema backend-only de detecção de objetos: um modelo **YOLO** identifica produtos em
imagens e devolve as caixas delimitadoras em **JSON**, consumíveis por qualquer
front-end ou sistema de PDV/estoque.

## Sumário

- [Dataset e classes](#dataset-e-classes)
- [Justificativa da arquitetura (por que YOLO11n)](#justificativa-da-arquitetura-por-que-yolo11n)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Preparação do dataset](#preparação-do-dataset)
- [Execução da API](#execução-da-api)
- [Payload de exemplo](#payload-de-exemplo)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Testes automatizados](#testes-automatizados)
- [Fine-tuning (opcional)](#fine-tuning-opcional)
- [Estrutura do projeto](#estrutura-do-projeto)
- [GitFlow (branches)](#gitflow-branches)
- [Kanban](#kanban)

## Dataset e classes

**Fonte:** COCO 2017 (`val2017`), conjunto de anotações do COCO — licença
[CC BY 4.0](https://cocodataset.org/#termsofuse), imagens com licenças originais
preservadas no dataset de origem.

**Preparação própria** (`scripts/prepare_dataset.py`): filtramos apenas as imagens que
contêm ao menos uma das **13 classes** abaixo, descartamos as anotações fora da seleção,
fizemos split `train/val/test` (70/15/15) **sem vazamento** (uma imagem nunca aparece em
dois splits) e geramos o `data.yaml` + um relatório de contagens
(`data/coco-subset/dataset_report.md`).

| ID (COCO) | Classe | Instâncias | Imagens |
|---|---|---|---|
| 39 | bottle | 1013 | 379 |
| 41 | cup | 895 | 390 |
| 45 | bowl | 623 | 314 |
| 46 | banana | 370 | 103 |
| 47 | apple | 236 | 76 |
| 48 | sandwich | 177 | 98 |
| 49 | orange | 285 | 85 |
| 50 | broccoli | 312 | 71 |
| 51 | carrot | 365 | 81 |
| 52 | hot dog | 125 | 51 |
| 53 | pizza | 284 | 153 |
| 54 | donut | 328 | 62 |
| 55 | cake | 310 | 124 |

**Totais: 1.160 imagens e 5.323 instâncias** — split `train` 812 · `val` 174 · `test` 174
(interseção entre splits = 0, ou seja, **sem vazamento**). Nenhuma classe ficou sem
imagens; a menor é `hot dog` (51 imagens) e a maior `cup` (390).

> **Por que os IDs do COCO são preservados?** O modelo pré-treinado devolve o ID da
> classe na posição original dele (0..79). Remapear para 0..12 faria o `yolo val`
> comparar classes erradas e produziria métricas falsas.

## Justificativa da arquitetura (por que YOLO11n)

- **As 13 classes já existem no COCO** (IDs 39–55, região "comida/mesa"), então o
  modelo pré-treinado as reconhece **sem nenhum treino** — isso é o que garante que ele
  "reconheça as 10+ classes com clareza" logo de cara.
- **`yolo11n` é o menor da família** (≈5,6 MB): carrega em menos de 1 s, roda em
  **MPS** (Apple Silicon) ou CPU e mantém a latência da API baixa — critério "tempos de
  inferência" da rubrica.
- Arquitetura moderna da Ultralytics (2024/2025): atenção C3k2 + SPPF, melhor relação
  acurácia/velocidade que o YOLOv5/v8 do mesmo porte.
- Trabalho **individual** → não há comparação entre versões (benchmark só vale para squad).

_Obs.: se o trabalho passar a ser em squad, acrescente aqui a tabela de mAP e velocidade
comparando `yolo11n/s/m`._

## Requisitos

- macOS com Apple Silicon (testado em M5 Pro, 24 GB) — funciona também em Linux/Windows com CPU.
- **Python 3.12** (obrigatório: o `python3` padrão da máquina é 3.15 e o `torch` ainda
  não publica wheel para `cp315`).
- ~2 GB de espaço livres (dataset + pesos).

## Instalação

```bash
# 1. clonar
git clone <URL_DO_REPOSITORIO>
cd <pasta_do_repositorio>

# 2. ambiente virtual com Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 3. dependências (versões fixadas)
pip install -r requirements.txt
```

## Preparação do dataset

```bash
# baixa (~1 GB) e processa — pode ser retomado se interromper
python scripts/prepare_dataset.py --download
```

Saída: `data/coco-subset/` com `images/{train,val,test}`, `labels/{train,val,test}`,
`data.yaml` e `dataset_report.md`. A pasta é ignorada pelo Git (`.gitignore`).

## Execução da API

```bash
uvicorn main:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Saúde do serviço: <http://127.0.0.1:8000/health>

Os pesos `yolo11n.pt` (~5,6 MB) são baixados automaticamente na primeira subida do
serviço.

## Payload de exemplo

**Requisição** (Swagger, Postman ou cURL):

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -H "accept: application/json" \
  -F "file=@exemplo.jpg;type=image/jpeg"
```

**Resposta** (contrato exigido pelo enunciado — sem imagem desenhada):

```json
{
  "status": "success",
  "detections": [
    {
      "class_name": "pizza",
      "confidence": 0.92,
      "box": { "x_min": 150, "y_min": 200, "x_max": 250, "y_max": 300 }
    },
    {
      "class_name": "bottle",
      "confidence": 0.87,
      "box": { "x_min": 40, "y_min": 60, "x_max": 95, "y_max": 210 }
    }
  ]
}
```

> Este é o formato exato. Os valores acima são o **modelo do contrato**; o payload
> real obtido na demonstração está em [examples/payload_real.json](examples/payload_real.json).

**Erros possíveis:** `400` content-type/arquivo inválido · `413` acima de
`MAX_UPLOAD_MB` · `503` modelo não carregado.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_PATH` | `yolo11n.pt` | pesos do modelo (use `runs/detect/.../best.pt` após fine-tuning) |
| `CONF` | `0.25` | confiança mínima |
| `IOU` | `0.45` | IoU da supressão de não-máximos |
| `DEVICE` | `auto` | `auto` (MPS se disponível) · `mps` · `cpu` |
| `SCOPE` | `dataset` | `dataset` = só as 13 classes · `all` = as 80 do COCO |
| `MAX_UPLOAD_MB` | `10` | limite do upload |

## Testes automatizados

```bash
pytest                 # unitários + integração (13+ testes)
pytest -m integration  # só os que carregam o modelo real
```

Cobrem: contrato JSON exato, filtros de escopo, content-type inválido, bytes que não são
imagem, arquivo vazio, limite de tamanho, modelo não carregado, imagem sem detecções e
o caminho completo de inferência com o modelo real.

## Fine-tuning (opcional)

Não é necessário para a entrega (o enunciado aceita modelo pré-treinado **ou**
fine-tuning). Se quiser o bônus:

```bash
python scripts/train.py --epochs 30
MODEL_PATH=runs/detect/scanvision/weights/best.pt uvicorn main:app --reload
```

## Estrutura do projeto

```
.
├── main.py                  # app FastAPI (uvicorn main:app --reload)
├── app/
│   ├── api.py               # POST /detect
│   ├── schemas.py           # contrato JSON
│   ├── core/
│   │   ├── config.py        # configuração por variáveis de ambiente
│   │   └── classes.py       # fonte única das 13 classes
│   └── services/
│       └── detector.py      # modelo carregado UMA única vez no startup
├── scripts/
│   ├── prepare_dataset.py   # download + filtro + split + data.yaml
│   └── train.py             # fine-tuning opcional
├── tests/                   # pytest
├── data/                    # dataset (gitignorado)
└── docs/                    # roteiro do vídeo, backlog do Kanban
```

## GitFlow (branches)

| Branch | Uso |
|---|---|
| `main` | entrega final (tudo mergeado aqui ao final) |
| `develop` | integração das features |
| `feature/project-scaffold` | estrutura, dependências e testes |
| `feature/dataset-prep` | download/preparação do dataset |
| `feature/api-setup` | endpoint `POST /detect` e validações |
| `feature/docs-delivery` | README, vídeo e Kanban |

Commits no padrão `tipo(escopo): descrição` (ex.: `feat(api): adiciona endpoint /detect`).

## Kanban

Quadro público no Trello (colunas **backlog · todo · doing · blocked · review · done**):

🔗 <https://trello.com/b/esPArbhZ/scanvision>

Quadro público com as 6 colunas obrigatórias. Os cartões de referência (título com o
nome do aluno + especificações + critério de pronto) estão em
[`docs/trello-backlog.md`](docs/trello-backlog.md).

## Créditos e licença

Dataset: COCO 2017 (Microsoft) — [CC BY 4.0](https://cocodataset.org/#termsofuse).
Modelos: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) (AGPL-3.0).
