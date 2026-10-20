# Relatório de avaliação (split de teste)

- Modelo: `yolo11n.pt` (pré-treinado no COCO, **sem fine-tuning**)
- Parâmetros da API: `conf=0.1`, `iou=0.45`, `device=mps`, `scope=dataset`
- Imagens avaliadas: **174** (split `test`)
- Critério de acerto: IoU ≥ 0.5 na mesma classe

| Classe | GT | TP | FP | FN | Precisão | Recall | F1 | Conf. média |
|---|---|---|---|---|---|---|---|---|
| bottle | 154 | 71 | 68 | 83 | 0.51 | 0.46 | 0.48 | 0.35 |
| cup | 174 | 89 | 85 | 85 | 0.51 | 0.51 | 0.51 | 0.42 |
| bowl | 100 | 72 | 55 | 28 | 0.57 | 0.72 | 0.63 | 0.46 |
| banana | 56 | 38 | 39 | 18 | 0.49 | 0.68 | 0.57 | 0.34 |
| apple | 37 | 10 | 23 | 27 | 0.30 | 0.27 | 0.29 | 0.33 |
| sandwich | 23 | 8 | 19 | 15 | 0.30 | 0.35 | 0.32 | 0.30 |
| orange | 61 | 26 | 30 | 35 | 0.46 | 0.43 | 0.44 | 0.39 |
| broccoli | 80 | 39 | 51 | 41 | 0.43 | 0.49 | 0.46 | 0.29 |
| carrot | 55 | 17 | 35 | 38 | 0.33 | 0.31 | 0.32 | 0.28 |
| hot dog | 22 | 12 | 1 | 10 | 0.92 | 0.55 | 0.69 | 0.56 |
| pizza | 19 | 15 | 5 | 4 | 0.75 | 0.79 | 0.77 | 0.61 |
| donut | 68 | 41 | 62 | 27 | 0.40 | 0.60 | 0.48 | 0.37 |
| cake | 29 | 15 | 17 | 14 | 0.47 | 0.52 | 0.49 | 0.43 |

**Total (micro):** GT 878 · TP 453 · FP 490 · FN 425 → precisão **0.48**, recall **0.52**, F1 **0.50**

- Classes presentes no GT: **13/13**
- Classes detectadas pela API: **13/13** (apple, banana, bottle, bowl, broccoli, cake, carrot, cup, donut, hot dog, orange, pizza, sandwich)

> Números medidos pela execução de `scripts/evaluate.py` — não estimados.
