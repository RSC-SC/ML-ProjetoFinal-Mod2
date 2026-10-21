# Evidências para os cartões movidos para `done`

Textos prontos para colar na descrição/comentário de cada cartão do Trello
(<https://trello.com/b/esPArbhZ/scanvision>) antes de arrastá-lo para `done`.
Todos os números foram **medidos** no repositório — nenhum é estimativa.

> Fluxo recomendado por cartão: `doing` → `review` → `done`, colando o texto
> correspondente como comentário antes do último movimento.

---

## 1. Setup do repositório GitFlow

```
Concluído em 21/10/2026.

O que foi feito:
- Repositório público github.com/RSC-SC/ML-ProjetoFinal-Mod2 com GitFlow completo:
  main (entrega), develop (integração) e 11 branches feature/*, uma por entrega,
  todas criadas a partir de develop e mergeadas com --no-ff.
- Branches: project-scaffold, dataset-prep, dataset-validation, api-setup,
  yolo-finetuning, identity-links, external-validation, repo-rules, video-demo,
  docs-delivery e frontend.
- Commits pequenos no padrão tipo(escopo): descrição
  (ex.: feat(api): adiciona endpoint /detect; docs(kanban): atualiza espelho).
- .gitignore cobre .venv/, data/, runs/ e *.pt: nenhum dataset, peso ou virtualenv
  versionado.
- Tudo mergeado em main e pushado (13 branches no remoto).

Como verificar: git log --oneline --graph --all e a aba Branches do GitHub.
```

## 2. Ambiente Python 3.12 e dependências fixadas

```
Concluído em 21/10/2026.

O que foi feito:
- O python3 do PATH é 3.15 (homebrew) e o torch não publica wheel cp315, então o
  venv foi criado com python3.12 (/opt/homebrew/bin/python3.12).
- requirements.txt com versões fixadas (ultralytics, fastapi, uvicorn,
  pydantic-settings, pillow, numpy, python-multipart, pytest, httpx).
- Inferência em device=mps (Apple Silicon M5 Pro 24 GB); não há CUDA na máquina.

Evidência:
- GET /health -> {"status":"ok","model":"yolo11n.pt","device":"mps","scope":"dataset"}
- import torch -> torch.backends.mps.is_available() == True

Como verificar: .venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

## 3. Estrutura da API e configuração por ambiente

```
Concluído em 21/10/2026.

O que foi feito:
- main.py com create_app() e lifespan: o modelo é carregado UMA única vez no
  startup, nunca dentro de um request.
- app/api.py (router POST /detect), app/schemas.py (contrato JSON),
  app/core/config.py (pydantic-settings + .env) e app/services/detector.py
  (singleton do modelo com filtro de escopo).
- Configuração por ambiente: MODEL_PATH, CONF, IOU, DEVICE, SCOPE, MAX_UPLOAD_MB
  e CORS_ORIGINS.

Evidência:
- log de startup "Modelo 'yolo11n.pt' carregado (device=mps)";
- GET /health informa modelo, device e escopo em uso.
```

## 4. Endpoint POST /detect com UploadFile

```
Concluído em 21/10/2026.

O que foi feito:
- POST /detect recebe UploadFile via multipart/form-data e devolve exatamente
  {status, detections[{class_name, confidence, box{x_min,y_min,x_max,y_max}}]}
  em pixels da imagem original, SEM imagem desenhada.
- Validações: content-type na allowlist (application/octet-stream tolerado),
  arquivo vazio (400), acima de MAX_UPLOAD_MB (413), imagem corrompida (400) e
  modelo não carregado (503), sempre via HTTPException com mensagem clara.
- Escopo de saída configurável: SCOPE=dataset (13 classes) ou all (80 do COCO).

Evidência:
- examples/payload_real.json, gerado por scripts/demo_payload.py a partir de uma
  requisição real;
- endpoint documentado no Swagger (/docs) com os códigos de erro.
```

## 5. Testes automatizados (pytest)

```
Concluído em 21/10/2026.

O que foi feito: 18 testes, todos verdes.

Cobertura:
- tests/test_api.py: contrato JSON exato (chaves obrigatórias), filtro de escopo,
  coordenadas inteiras em pixels, content-type inválido, arquivo vazio, limite de
  tamanho (413), modelo não carregado (503), imagem sem detecções e CORS.
- tests/test_frontend.py: index servido em /, assets estáticos e precedência das
  rotas da API sobre o mount do frontend.
- tests/test_integration.py: caminho completo de inferência com o yolo11n.pt real.

Evidência: .venv/bin/python -m pytest -q -> 18 passed (2,1 s com o modelo real).
```

## 6. Criação do quadro Kanban público

```
Concluído em 21/10/2026.

O que foi feito:
- Quadro público com as 6 colunas obrigatórias, nesta ordem: backlog, todo, doing,
  blocked, review, done.
- Todos os cartões com o nome do aluno no título, especificação e critério de
  pronto na descrição.
- Link referenciado no README e espelho dos cartões versionado em
  docs/trello-backlog.md.

Link: https://trello.com/b/esPArbhZ/scanvision
Como verificar: abrir o link em aba anônima (acesso público).
```

## 7. Baixar e preparar o dataset (13 classes)

```
Concluído em 21/10/2026.

O que foi feito:
- scripts/prepare_dataset.py --download baixa val2017.zip (~778 MB) e
  coco2017labels-segments.zip (~169 MB) e monta data/coco-subset/.
- 13 classes do COCO (IDs 39, 41, 45-55), mantendo os 80 names na posição
  original no data.yaml (remap para 0..12 invalidaria a avaliação).
- Split 70/15/15 com seed 42, por nome de arquivo, sem vazamento: nenhuma imagem
  aparece em dois splits.

Números medidos: 1.160 imagens e 5.323 instâncias.
Instâncias por classe: bottle 1013, cup 895, bowl 623, banana 370, carrot 365,
donut 328, broccoli 312, cake 310, orange 285, pizza 284, apple 236,
sandwich 177, hot dog 125.

Splits: train 812 / val 174 / test 174.
O dataset não é versionado (gitignored) — é reproduzível pelo script.
```

## 8. Relatório de contagens e splits sem vazamento

```
Concluído em 21/10/2026.

O que foi feito:
- data/coco-subset/dataset_report.md gerado automaticamente pelo script de
  preparação, com imagens e instâncias por classe e o total.
- Conferência de vazamento: cada imagem é sorteada uma única vez para um split
  (train/val/test), então a mesma cena não aparece em splits diferentes.
- Nenhuma classe ficou com 0 imagens; a menor é hot dog (51 imagens / 125
  instâncias), suficiente para avaliação.

Evidência: tabela das 13 classes replicada no README (seção "Dataset e classes")
e arquivo dataset_report.md.
```

## 9. Validação do modelo em imagens variadas

```
Concluído em 21/10/2026.

O que foi feito:
- 38 imagens do Wikimedia Commons, fora da distribuição do COCO (fundo, ângulo,
  iluminação e oclusão variados), baixadas por scripts/fetch_external.py com
  licença e atribuição registradas em examples/external/ATTRIBUICAO.md e
  atribuicao.json.
- Cada imagem enviada ao POST /detect com conf=0.25, iou=0.45, device=mps.

Resultado medido:
- 13/13 classes detectadas ao menos uma vez;
- a classe esperada apareceu em 22 das 38 imagens (58%);
- relatório por imagem, com as confianças observadas, em docs/external-report.md.

Limitações registradas no relatório: cake_2 foi classificada como donut; algumas
imagens com oclusão forte (bottle_2, broccoli_1, cup_1) não geraram detecção.
```

## 10. README completo

```
Concluído em 21/10/2026.

O que foi feito: README.md com todas as seções exigidas e dados reais.
- Nome do software (ScanVision) e autor;
- dataset e as 13 classes, com contagens de imagens e instâncias;
- justificativa da arquitetura (YOLO11n pré-treinado, sem fine-tuning — opcional
  no enunciado);
- requisitos e instalação (python3.12 + requirements.txt);
- execução: uvicorn main:app --reload (Swagger em /docs, /health);
- payload de exemplo real (requisição curl + resposta JSON completa);
- tabela de variáveis de ambiente (inclui CORS_ORIGINS);
- avaliação medida e generalização, com links para os relatórios;
- testes, estrutura do projeto, GitFlow e link do Kanban;
- créditos e licenças (COCO CC BY 4.0, Ultralytics AGPL-3.0).
```

## 11. Interface de demonstração (frontend que desenha as caixas)

```
Concluído em 21/10/2026.

O que foi feito:
- frontend/ com index.html, app.js e style.css (HTML/CSS/JS puro, sem build e sem
  dependências), servido pela própria API em GET /.
- Fluxo: drag&drop ou seletor de arquivo -> multipart/form-data para POST /detect
  -> as caixas do JSON são desenhadas no <canvas>, com cor determinística por
  classe e rótulo "classe + confiança"; a página também mostra a tabela de
  detecções e o JSON bruto.
- O contrato do endpoint NÃO mudou: a API continua devolvendo apenas JSON; o
  desenho é feito no cliente.
- CORS_ORIGINS (opcional) permite servir o frontend em outro host.

Evidência (20/10/2026): orange_2.jpg -> 16 detecções, tabela e JSON renderizados,
44 ms e 56 ms de ida e volta com o modelo aquecido, zero erros no console.
Cobertura: tests/test_frontend.py (index, assets e precedência das rotas da API).
```

## 12. Demonstração com vídeo (scripts/detect_video.py)

```
Concluído em 21/10/2026.

O que foi feito:
- scripts/detect_video.py extrai quadros de um arquivo de vídeo e envia cada
  quadro para o POST /detect rodando — o mesmo caminho que o avaliador usa — e
  grava um vídeo anotado em runs/video/ (gitignored).
- Não foi criado endpoint de vídeo: o enunciado pede upload de imagem, então o
  vídeo é apenas uma camada de demonstração, como o frontend.

Resultado medido (vídeo sintético de 24 quadros, 3 fps):
- 8 quadros enviados, detecções em 7 deles, 6 classes distintas;
- latência mediana de 11 ms por quadro no MPS (primeiro quadro ~860 ms por
  aquecimento do modelo), cerca de 8 quadros/s.
```

## 13. Avaliação medida no split de teste (se o cartão existir)

```
Concluído em 21/10/2026.

O que foi feito: scripts/evaluate.py rodou os 174 pares imagem/anotação do split
test com conf=0.25, iou=0.45, device=mps e critério de acerto IoU >= 0,5 na mesma
classe.

Resultado micro: GT 878, TP 368, FP 159, FN 510
-> precisão 0,70 · recall 0,42 · F1 0,52.
13/13 classes presentes no GT e 13/13 detectadas.

Destaques: pizza F1 0,76 (P 0,87 / R 0,68), bowl F1 0,70, hot dog F1 0,65.
Piores: carrot F1 0,29 e apple F1 0,34 (itens pequenos e muito ocluídos).
Sensibilidade: com conf=0.10 -> P 0,48 / R 0,52 / F1 0,50.

Relatórios: docs/evaluation-report.md e docs/evaluation-report-conf010.md.
Números medidos pela execução do script, não estimados.
```
