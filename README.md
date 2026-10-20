# ScanVision — Detecção de Objetos em Imagens de Supermercado

> **Autor:** Rafael Cunha · **Curso:** SCTEC — Módulo 2 (Machine Learning e Visão Computacional)
> **Nome do software:** `ScanVision` — detecta produtos em imagens de supermercado.
> **Repositório:** <https://github.com/RSC-SC/ML-ProjetoFinal-Mod2> ·
> **Kanban:** <https://trello.com/b/esPArbhZ/scanvision>

Sistema backend-only de detecção de objetos: um modelo **YOLO** identifica produtos em
imagens e devolve as caixas delimitadoras em **JSON**, consumíveis por qualquer
front-end ou sistema de PDV/estoque.

## Sumário

- [Dataset e classes](#dataset-e-classes)
- [Justificativa da arquitetura (por que YOLO11n)](#justificativa-da-arquitetura-por-que-yolo11n)
- [Avaliação (resultados medidos)](#avaliação-resultados-medidos)
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

## Avaliação (resultados medidos)

Executado com `python scripts/evaluate.py` sobre as **174 imagens do split `teste`**,
usando **exatamente** o mesmo caminho de inferência da API (`conf=0.25`, `iou=0.45`,
`device=mps`, escopo das 13 classes) e casamento predito×referência com **IoU ≥ 0,5**:

| Ponto de operação | Precisão | Recall | F1 | Classes detectadas |
|---|---|---|---|---|
| `conf=0.25` (padrão da API) | **0,70** | **0,42** | **0,52** | **13/13** |
| `conf=0.10` | 0,48 | 0,52 | 0,50 | **13/13** |

- **O modelo pré-treinado reconhece as 13 classes sem nenhum treino** — comportamento
  esperado, pois todas existem entre as 80 classes do COCO em que o `yolo11n` foi
  treinado. É esse o ponto que sustenta o critério "reconhece as 10+ classes com clareza".
- Baixar o limiar de confiança troca precisão por recall (aqui, medido: de 0,70/0,42 para
  0,48/0,52) — por isso o padrão é `conf=0.25`.
- Tabelas por classe (GT, TP, FP, FN, confiança média):
  [`docs/evaluation-report.md`](docs/evaluation-report.md) e
  [`docs/evaluation-report-conf010.md`](docs/evaluation-report-conf010.md).
- Reproduzir: `python scripts/evaluate.py` e `CONF=0.10 python scripts/evaluate.py --out docs/evaluation-report-conf010.md`.

### Generalização (imagens fora do COCO)

O split de teste acima vem do mesmo COCO em que o modelo foi treinado. Para provar
comportamento em imagens que o modelo **nunca viu**, baixamos **38 imagens do
Wikimedia Commons** (2 a 4 por classe, fora do COCO, licenças livres registradas em
[`examples/external/ATTRIBUICAO.md`](examples/external/ATTRIBUICAO.md)):

| Métrica | Resultado |
|---|---|
| Imagens avaliadas | 38 (fora do COCO) |
| Classes detectadas | **13/13** |
| Acerto da classe esperada | 22/38 (**58%**) |
| Imagens sem nenhuma detecção | 12/38 |

Leitura honesta dos números:
- **As 13 classes aparecem em imagens nunca vistas** — inclusive `pizza` (0,88),
  `donut` (0,93), `broccoli` (0,94), `carrot` (0,91) e `cup` (0,89) com confiança alta.
- As falhas se concentram em fotos onde o objeto **não é o assunto** (cena de rua,
  artefato de museu, foto de produto em fundo branco): as categorias do Commons são
  ruidosas, e essa é uma limitação da fonte, não só do modelo.
- Os erros restantes são pares naturalmente confundíveis: `donut`↔`cake`,
  `cup`↔`bottle`, `orange`↔`apple`, `carrot`↔`bowl` — comportamento esperado do
  `yolo11n` (o menor modelo da família).
- Relatório imagem a imagem: [`docs/external-report.md`](docs/external-report.md).
- Reproduzir: `python scripts/fetch_external.py` (baixa/atribui) e
  `python scripts/eval_external.py` (gera o relatório e as cópias anotadas para o vídeo).

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

**Requisição** (Swagger, Postman ou cURL) — usando a imagem de exemplo do repositório:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -H "accept: application/json" \
  -F "file=@examples/000000002149.jpg;type=image/jpeg"
```

**Resposta real** (contrato exigido pelo enunciado — sem imagem desenhada):

```json
{
  "status": "success",
  "detections": [
    {
      "class_name": "apple",
      "confidence": 0.7957,
      "box": { "x_min": 78, "y_min": 74, "x_max": 341, "y_max": 329 }
    },
    {
      "class_name": "apple",
      "confidence": 0.6672,
      "box": { "x_min": 335, "y_min": 71, "x_max": 586, "y_max": 325 }
    },
    {
      "class_name": "apple",
      "confidence": 0.6225,
      "box": { "x_min": 0, "y_min": 236, "x_max": 250, "y_max": 399 }
    },
    {
      "class_name": "apple",
      "confidence": 0.2705,
      "box": { "x_min": 7, "y_min": 66, "x_max": 635, "y_max": 405 }
    },
    {
      "class_name": "banana",
      "confidence": 0.2581,
      "box": { "x_min": 574, "y_min": 161, "x_max": 640, "y_max": 330 }
    }
  ]
}
```

> Esta resposta foi obtida executando a API real (`uvicorn main:app`) e fazendo o
> upload da imagem — não foi digitada à mão. Arquivo original:
> [`examples/payload_real.json`](examples/payload_real.json) ·
> regenerar com `python scripts/demo_payload.py`.
> A imagem `examples/000000002149.jpg` vem do split de teste do COCO 2017.

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
