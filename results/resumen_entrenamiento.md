# Resumen de entrenamiento y prueba — Identificación de especies de mosquitos
Generado automáticamente el 2026-10-12 13:33 a partir de `results/`, `logs/` y `models/` (`python app/summary.py`). Todas las cifras de test se calcularon sobre imágenes que ningún modelo vio al entrenar.

## 1. Datos y partición
- Conjunto original (`train.csv` del MosquitoAlert Challenge 2023): **10,357 imágenes**, una caja por imagen.
- Limpieza: se descartaron **3** filas con cajas geométricamente inválidas (xbr ≤ xtl, ybr ≤ ytl o fuera de la imagen). Quedan **10,354** imágenes.
- Partición estratificada por clase 70 / 15 / 15 (entrenamiento / validación / prueba), semilla 42, hecha **antes** de cualquier sobremuestreo para no filtrar copias entre particiones.
- Validación: selección de hiperparámetros y de la mejor época. Prueba: solo para las cifras finales.

| clase | train | val | test | total | % del total |
|---|---|---|---|---|---|
| aegypti | 33 | 7 | 7 | 47 | 0.45 |
| albopictus | 3227 | 692 | 692 | 4611 | 44.53 |
| anopheles | 58 | 13 | 13 | 84 | 0.81 |
| culex | 3193 | 684 | 684 | 4561 | 44.05 |
| culiseta | 436 | 93 | 93 | 622 | 6.01 |
| japonicus/koreicus | 301 | 64 | 64 | 429 | 4.14 |
| Total | 7248 | 1553 | 1553 | 10354 | 100.00 |

Desbalance: la clase mayoritaria (albopictus, 4,611) tiene 98 veces más imágenes que la minoritaria (aegypti, 47). Por eso la métrica principal es el **F1 macro** (métrica oficial del reto), que pesa igual a cada clase.

## 2. Preprocesamiento (`src/prepare_data.py`)
- Recorte cuadrado centrado en la caja anotada con **15 % de margen**, redimensionado a **256 × 256 px** (entrada de todos los clasificadores). La app aplica exactamente el mismo recorte a la caja del detector.
- Para el detector: imagen completa reescalada a 640 px en el lado mayor y etiqueta YOLO de una sola clase («mosquito»).
- Decodificación JPEG a media resolución (`draft`) para acelerar ~4× el preprocesamiento de ~9 GB de imágenes.
- CNN: aumento de datos (RandomResizedCrop 0.6–1.0, volteos horizontal y vertical, rotación ±20°, ColorJitter) y **sobremuestreo con WeightedRandomSampler hasta clases equiprobables** (Buda et al., 2018); evaluación con recorte central de 224 px.
- DINOv2: características congeladas de 384 dimensiones (ViT-S/14, entrada 224 px) extraídas una sola vez; clasificadores con `class_weight='balanced'` y estandarización.

## 3. Recursos de cómputo y herramientas
- GPU: NVIDIA GeForce GTX 1650, 4096 MiB. Entrenamiento local (sin Colab); una tarea de GPU a la vez, los modelos sobre DINOv2 en CPU en paralelo.
- Software: Python 3.11.17; torch 2.14.1+cu126, torchvision 0.29.1+cu126, timm 1.0.30, ultralytics 8.4.175, scikit-learn 1.9.1, numpy 2.4.6, pandas 3.0.6, streamlit 1.65.0, plotly 7.1.0.
- Medición previa de rendimiento en la GPU (imágenes/s de entrenamiento), que justificó usar AMP sin `channels_last` y reducir ConvNeXt-Tiny a 176 px con las dos primeras etapas congeladas:

| arch | amp | channels_last | img_per_s |
|---|---|---|---|
| efficientnet_b0 | False | False | 115.9 |
| efficientnet_b0 | True | False | 120.6 |
| efficientnet_b0 | True | True | 16.1 |
| convnext_tiny | False | False | 8.1 |
| convnext_tiny | True | False | 13.0 |
| convnext_tiny | True | True | 13.0 |

## 4. Modelos entrenados y ajuste de hiperparámetros
| modelo | familia | configuración elegida | F1 macro val | entrenamiento (min) | inferencia (ms/img) | parámetros entrenables |
|---|---|---|---|---|---|---|
| dinov2_svm | DINOv2 congelado + clasificador plano | C = 10.0 | 0.692 | 1.923 | 37.322 | — |
| dinov2_hier_cascade | DINOv2 congelado + jerárquico (2 niveles) | C = 0.1 | 0.640 | 0.158 | 35.976 | — |
| dinov2_logreg | DINOv2 congelado + clasificador plano | C = 0.1 | 0.633 | 0.133 | 35.972 | — |
| dinov2_hier_taxonomic | DINOv2 congelado + jerárquico (2 niveles) | C = 0.1 | 0.610 | 0.257 | 35.990 | — |
| convnext_tiny | CNN fine-tune | 3 épocas, batch 32, lr 0.0003, 176 px, 2 etapas congeladas, AdamW + OneCycle, label smoothing 0.1 | 0.633 | 17.590 | 19.864 | 26,589,702 |
| efficientnet_b0 | CNN fine-tune | 6 épocas, batch 32, lr 0.0005, 224 px, AdamW + OneCycle, label smoothing 0.1 | 0.565 | 8.744 | 12.269 | 4,015,234 |

Notas: el tiempo de inferencia de los modelos DINOv2 incluye la extracción de características en GPU; los jerárquicos reutilizan el C de la regresión logística plana.

**Búsqueda de C (F1 macro en validación, ✔ = elegido):**

- `dinov2_logreg`: C=0.01 → 0.6195, C=0.1 → 0.6335 ✔, C=1 → 0.6234, C=10 → 0.5809
- `dinov2_svm`: C=1 → 0.6836, C=3 → 0.6912, C=10 → 0.6917 ✔

**Curva de entrenamiento `convnext_tiny`** (se guarda la época con mejor F1 en validación):

| época | pérdida train | F1 macro val | segundos |
|---|---|---|---|
| 1 | 1.0123 | 0.5696 | 477 |
| 2 | 0.6126 | 0.6326 | 288 |
| 3 | 0.5060 | 0.6297 | 290 |

**Curva de entrenamiento `efficientnet_b0`** (se guarda la época con mejor F1 en validación):

| época | pérdida train | F1 macro val | segundos |
|---|---|---|---|
| 1 | 1.9115 | 0.3788 | 165 |
| 2 | 0.8277 | 0.4844 | 72 |
| 3 | 0.6490 | 0.5404 | 71 |
| 4 | 0.5827 | 0.5368 | 72 |
| 5 | 0.5550 | 0.5600 | 72 |
| 6 | 0.5362 | 0.5646 | 72 |

## 5. Resultados en el conjunto de prueba
1,553 imágenes de prueba. Ordenado por F1 macro.

| modelo | F1 macro | exactitud balanceada | exactitud |
|---|---|---|---|
| dinov2_svm | 0.664 | 0.618 | 0.940 |
| dinov2_hier_cascade | 0.662 | 0.689 | 0.907 |
| dinov2_logreg | 0.630 | 0.707 | 0.900 |
| dinov2_hier_taxonomic | 0.623 | 0.684 | 0.896 |
| convnext_tiny | 0.565 | 0.606 | 0.892 |
| efficientnet_b0 | 0.540 | 0.568 | 0.876 |

**Mejor modelo: `dinov2_svm`** (F1 macro 0.664). La exactitud global es alta para todos porque las dos clases mayoritarias dominan; el F1 macro muestra la diferencia en las clases raras.

**F1 por clase:**

| modelo | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| dinov2_svm | 0.000 | 0.961 | 0.632 | 0.971 | 0.805 | 0.618 |
| dinov2_hier_cascade | 0.111 | 0.942 | 0.667 | 0.949 | 0.738 | 0.568 |
| dinov2_logreg | 0.000 | 0.931 | 0.579 | 0.951 | 0.792 | 0.526 |
| dinov2_hier_taxonomic | 0.000 | 0.924 | 0.588 | 0.953 | 0.762 | 0.512 |
| convnext_tiny | 0.000 | 0.948 | 0.300 | 0.937 | 0.642 | 0.565 |
| efficientnet_b0 | 0.000 | 0.919 | 0.190 | 0.938 | 0.667 | 0.528 |

**Recall por clase** (proporción de cada clase real que el modelo reconoce):

| modelo | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| dinov2_svm | 0.000 | 0.980 | 0.462 | 0.982 | 0.753 | 0.531 |
| dinov2_hier_cascade | 0.143 | 0.926 | 0.692 | 0.942 | 0.774 | 0.656 |
| dinov2_logreg | 0.000 | 0.905 | 0.846 | 0.931 | 0.839 | 0.719 |
| dinov2_hier_taxonomic | 0.000 | 0.893 | 0.769 | 0.940 | 0.828 | 0.672 |
| convnext_tiny | 0.000 | 0.931 | 0.231 | 0.898 | 0.828 | 0.750 |
| efficientnet_b0 | 0.000 | 0.883 | 0.154 | 0.927 | 0.710 | 0.734 |

Clase más difícil para el mejor modelo: **aegypti** (F1 0.00, 7 imágenes en test).

**Matrices de confusión** (filas = clase real, columnas = predicción):

`dinov2_svm`

| real \ pred | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| aegypti | 0 | 6 | 0 | 0 | 0 | 1 |
| albopictus | 0 | 678 | 0 | 5 | 2 | 7 |
| anopheles | 0 | 0 | 6 | 6 | 1 | 0 |
| culex | 0 | 4 | 0 | 672 | 7 | 1 |
| culiseta | 0 | 5 | 0 | 15 | 70 | 3 |
| japonicus/koreicus | 1 | 26 | 0 | 2 | 1 | 34 |

`dinov2_hier_cascade`

| real \ pred | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| aegypti | 1 | 5 | 0 | 1 | 0 | 0 |
| albopictus | 8 | 641 | 0 | 8 | 4 | 31 |
| anopheles | 0 | 0 | 9 | 4 | 0 | 0 |
| culex | 1 | 7 | 4 | 644 | 22 | 6 |
| culiseta | 0 | 2 | 1 | 13 | 72 | 5 |
| japonicus/koreicus | 1 | 14 | 0 | 3 | 4 | 42 |

`dinov2_logreg`

| real \ pred | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| aegypti | 0 | 6 | 0 | 0 | 0 | 1 |
| albopictus | 4 | 626 | 1 | 6 | 3 | 52 |
| anopheles | 0 | 0 | 11 | 2 | 0 | 0 |
| culex | 0 | 7 | 12 | 637 | 21 | 7 |
| culiseta | 0 | 1 | 1 | 8 | 78 | 5 |
| japonicus/koreicus | 1 | 13 | 0 | 2 | 2 | 46 |

`dinov2_hier_taxonomic`

| real \ pred | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| aegypti | 0 | 5 | 0 | 0 | 1 | 1 |
| albopictus | 6 | 618 | 1 | 8 | 7 | 52 |
| anopheles | 0 | 0 | 10 | 3 | 0 | 0 |
| culex | 1 | 7 | 8 | 643 | 20 | 5 |
| culiseta | 0 | 1 | 2 | 10 | 77 | 3 |
| japonicus/koreicus | 1 | 14 | 0 | 2 | 4 | 43 |

`convnext_tiny`

| real \ pred | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| aegypti | 0 | 4 | 0 | 0 | 2 | 1 |
| albopictus | 0 | 644 | 2 | 4 | 4 | 38 |
| anopheles | 0 | 1 | 3 | 0 | 7 | 2 |
| culex | 0 | 6 | 1 | 614 | 52 | 11 |
| culiseta | 0 | 2 | 1 | 7 | 77 | 6 |
| japonicus/koreicus | 0 | 10 | 0 | 1 | 5 | 48 |

`efficientnet_b0`

| real \ pred | aegypti | albopictus | anopheles | culex | culiseta | japonicus/koreicus |
|---|---|---|---|---|---|---|
| aegypti | 0 | 5 | 0 | 0 | 1 | 1 |
| albopictus | 10 | 611 | 1 | 9 | 7 | 54 |
| anopheles | 0 | 2 | 2 | 4 | 4 | 1 |
| culex | 7 | 8 | 3 | 634 | 26 | 6 |
| culiseta | 2 | 2 | 0 | 18 | 66 | 5 |
| japonicus/koreicus | 1 | 10 | 2 | 3 | 1 | 47 |

## 6. Análisis complementario: combinación y acuerdo entre modelos
- Promedio de las probabilidades de **todos** los modelos (dinov2_svm, dinov2_hier_cascade, dinov2_logreg, dinov2_hier_taxonomic, convnext_tiny, efficientnet_b0), sin elegir miembros con el conjunto de prueba: F1 macro **0.672**, exactitud balanceada 0.691, exactitud 0.921 (mejor modelo individual: 0.664).
  F1 por clase del promedio: aegypti 0.00, albopictus 0.95, anopheles 0.67, culex 0.96, culiseta 0.80, japonicus/koreicus 0.66.
- Todos los modelos coinciden en el **76.6%** de las imágenes de prueba; cuando coinciden aciertan el 98.3% de las veces, y cuando discrepan cada modelo acierta en promedio solo el 63.5%. El desacuerdo entre modelos es una buena señal de imagen dudosa (la app lo muestra como «consenso»).

## 7. Detector YOLOv8n (localización)
- Una sola clase («mosquito»), entrenado con las cajas de las seis clases. mAP@50 = **0.979**, mAP@50-95 = 0.723 en test.
- IoU medio de la caja de mayor confianza: **0.831**; imágenes con IoU ≥ 0.75 (umbral del reto): **83.3%** de 1,553.
- Entrenamiento: 20.0 min, YOLOv8n preentrenado (COCO), imagen de 512 px, batch 16, AMP, paciencia 5; tope de 12 épocas y de 0.3 h → se completaron **6 épocas**. Inferencia: 32.5 ms por imagen.

**Curva de entrenamiento del detector** (validación por época):

| epoch | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 0.898 | 0.874 | 0.945 | 0.613 |
| 2 | 0.890 | 0.843 | 0.922 | 0.606 |
| 3 | 0.912 | 0.873 | 0.940 | 0.623 |
| 4 | 0.906 | 0.907 | 0.960 | 0.659 |
| 5 | 0.933 | 0.915 | 0.972 | 0.688 |
| 6 | 0.948 | 0.948 | 0.982 | 0.721 |

## 8. Figuras estáticas para el informe (`reports/figures/`)
- `res_01_comparacion_modelos.png`: F1 macro, exactitud balanceada y exactitud de cada modelo en test
- `res_02_f1_por_clase.png`: Mapa de calor del F1 por clase y modelo
- `res_03_confusion_mejor_modelo.png`: Matriz de confusión normalizada del mejor modelo
- `res_04_costo_vs_f1.png`: F1 macro frente al costo de inferencia por imagen
- `res_05_curvas_entrenamiento.png`: Pérdida y F1 de validación por época de las CNN
- `res_06_iou_detector.png`: Distribución del IoU del detector frente al umbral 0.75 del reto
