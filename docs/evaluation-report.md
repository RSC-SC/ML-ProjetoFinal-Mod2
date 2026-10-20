# Relatório de avaliação (split de teste)

- Modelo: `yolo11n.pt` (pré-treinado no COCO, **sem fine-tuning**)
- Parâmetros da API: `conf=0.25`, `iou=0.45`, `device=mps`, `scope=dataset`
- Imagens avaliadas: **174** (split `test`)
- Critério de acerto: IoU ≥ 0.5 na mesma classe

| Classe | GT | TP | FP | FN | Precisão | Recall | F1 | Conf. média |
|---|---|---|---|---|---|---|---|---|
| bottle | 154 | 58 | 17 | 96 | 0.77 | 0.38 | 0.51 | 0.52 |
| cup | 174 | 72 | 33 | 102 | 0.69 | 0.41 | 0.52 | 0.59 |
| bowl | 100 | 65 | 22 | 35 | 0.75 | 0.65 | 0.70 | 0.59 |
| banana | 56 | 31 | 11 | 25 | 0.74 | 0.55 | 0.63 | 0.50 |
| apple | 37 | 9 | 7 | 28 | 0.56 | 0.24 | 0.34 | 0.52 |
| sandwich | 23 | 8 | 3 | 15 | 0.73 | 0.35 | 0.47 | 0.49 |
| orange | 61 | 20 | 13 | 41 | 0.61 | 0.33 | 0.43 | 0.56 |
| broccoli | 80 | 28 | 14 | 52 | 0.67 | 0.35 | 0.46 | 0.44 |
| carrot | 55 | 11 | 10 | 44 | 0.52 | 0.20 | 0.29 | 0.48 |
| hot dog | 22 | 11 | 1 | 11 | 0.92 | 0.50 | 0.65 | 0.60 |
| pizza | 19 | 13 | 2 | 6 | 0.87 | 0.68 | 0.76 | 0.75 |
| donut | 68 | 29 | 20 | 39 | 0.59 | 0.43 | 0.50 | 0.60 |
| cake | 29 | 13 | 6 | 16 | 0.68 | 0.45 | 0.54 | 0.62 |

**Total (micro):** GT 878 · TP 368 · FP 159 · FN 510 → precisão **0.70**, recall **0.42**, F1 **0.52**

- Classes presentes no GT: **13/13**
- Classes detectadas pela API: **13/13** (apple, banana, bottle, bowl, broccoli, cake, carrot, cup, donut, hot dog, orange, pizza, sandwich)

> Números medidos pela execução de `scripts/evaluate.py` — não estimados.
