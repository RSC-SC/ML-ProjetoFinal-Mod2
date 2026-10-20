# Relatório de generalização (imagens externas)

- Modelo: `yolo11n.pt` (pré-treinado, **sem fine-tuning**)
- Parâmetros: `conf=0.25`, `iou=0.45`, `device=mps`
- Imagens: **38**, baixadas do Wikimedia Commons, **fora da distribuição do COCO**
- Métrica: acertou a classe esperada (pelo nome do arquivo) com pelo menos uma detecção

| Imagem | Classe esperada | Acerto | Detecções (classe/confiança) |
|---|---|---|---|
| `apple_1.jpg` | apple | ✅ | apple 0.87 · apple 0.85 · apple 0.80 · orange 0.68 · apple 0.56 · banana 0.27 |
| `apple_2.jpg` | apple | ✅ | apple 0.85 · apple 0.83 · apple 0.78 · apple 0.29 |
| `banana_1.jpg` | banana | ✅ | banana 0.87 |
| `banana_2.jpg` | banana | ✅ | banana 0.90 |
| `bottle_1.jpg` | bottle | ✅ | bottle 0.44 · bottle 0.41 |
| `bottle_2.jpg` | bottle | ❌ | — |
| `bottle_3.jpg` | bottle | ❌ | — |
| `bowl_1.jpg` | bowl | ✅ | bowl 0.57 |
| `bowl_2.jpg` | bowl | ✅ | bowl 0.43 · bowl 0.32 |
| `broccoli_1.jpg` | broccoli | ❌ | — |
| `broccoli_2.jpg` | broccoli | ✅ | broccoli 0.94 |
| `broccoli_3.jpg` | broccoli | ✅ | broccoli 0.75 · broccoli 0.31 |
| `cake_1.jpg` | cake | ✅ | cake 0.74 · cake 0.61 |
| `cake_2.jpg` | cake | ❌ | donut 0.88 · donut 0.79 · donut 0.78 |
| `cake_3.jpg` | cake | ❌ | — |
| `carrot_1.jpg` | carrot | ✅ | carrot 0.91 · carrot 0.89 · carrot 0.86 · carrot 0.59 · carrot 0.58 · carrot 0.36 |
| `carrot_2.jpg` | carrot | ❌ | bowl 0.71 · bowl 0.66 · bowl 0.41 · bowl 0.35 |
| `carrot_3.jpg` | carrot | ✅ | carrot 0.76 · carrot 0.62 · carrot 0.49 |
| `cup_1.jpg` | cup | ❌ | — |
| `cup_2.jpg` | cup | ❌ | bottle 0.34 |
| `cup_3.jpg` | cup | ❌ | — |
| `cup_4.jpg` | cup | ✅ | cup 0.89 |
| `donut_1.jpg` | donut | ❌ | — |
| `donut_2.jpg` | donut | ✅ | cake 0.74 · cake 0.73 · cake 0.70 · cake 0.65 · cake 0.56 · cake 0.53 |
| `donut_3.jpg` | donut | ✅ | donut 0.93 · donut 0.92 · cup 0.91 |
| `hot_dog_1.jpg` | hot dog | ❌ | — |
| `hot_dog_2.jpg` | hot dog | ❌ | — |
| `hot_dog_3.jpg` | hot dog | ✅ | cup 0.73 · sandwich 0.41 · hot dog 0.30 |
| `hot_dog_4.jpg` | hot dog | ✅ | hot dog 0.40 |
| `orange_1.jpg` | orange | ❌ | — |
| `orange_2.jpg` | orange | ✅ | orange 0.84 · orange 0.84 · orange 0.82 · orange 0.77 · apple 0.63 · bowl 0.60 |
| `orange_3.jpg` | orange | ✅ | orange 0.79 · orange 0.73 · orange 0.72 · orange 0.66 · orange 0.63 · orange 0.53 |
| `pizza_1.jpg` | pizza | ❌ | — |
| `pizza_2.jpg` | pizza | ❌ | — |
| `pizza_3.jpg` | pizza | ✅ | pizza 0.88 |
| `pizza_4.jpg` | pizza | ❌ | cup 0.71 · cup 0.60 |
| `sandwich_1.jpg` | sandwich | ✅ | sandwich 0.53 |
| `sandwich_2.jpg` | sandwich | ✅ | bottle 0.84 · sandwich 0.77 · cup 0.47 · bottle 0.40 |

## Resumo

| Classe esperada | Imagens | Acertos | Taxa | Confiança média |
|---|---|---|---|---|
| apple | 2 | 2 | 100% | 0.68 |
| banana | 2 | 2 | 100% | 0.89 |
| bottle | 3 | 1 | 33% | 0.43 |
| bowl | 2 | 2 | 100% | 0.44 |
| broccoli | 3 | 2 | 67% | 0.67 |
| cake | 3 | 1 | 33% | 0.76 |
| carrot | 3 | 2 | 67% | 0.58 |
| cup | 4 | 1 | 25% | 0.62 |
| donut | 3 | 2 | 67% | 0.53 |
| hot dog | 4 | 2 | 50% | 0.46 |
| orange | 3 | 2 | 67% | 0.55 |
| pizza | 4 | 1 | 25% | 0.73 |
| sandwich | 2 | 2 | 100% | 0.60 |

**Taxa de acerto global: 22/38 (58%)**
- Imagens sem nenhuma detecção: **12/38**
- Classes distintas detectadas: **13/13** (apple, banana, bottle, bowl, broccoli, cake, carrot, cup, donut, hot dog, orange, pizza, sandwich)
- Cópias anotadas (para o vídeo) em `runs/external/` — a API não devolve imagem.

> Números medidos por `scripts/eval_external.py`. Sem anotações humanas estas
> imagens, por isso reportamos acerto da classe esperada e confianças — não mAP.
