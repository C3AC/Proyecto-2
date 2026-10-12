# Identificación de Especies de Mosquitos

Proyecto 2 del curso CC3084 (Data Science), UVG, semestre II 2026. El grupo trabaja sobre el
reto MosquitoAlert Challenge 2023 (AIcrowd): localizar el mosquito en fotos tomadas por
ciudadanos y clasificarlo en una de seis especies/géneros, con cajas anotadas por entomólogos.

Página del reto: https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023

El sistema tiene dos etapas, según la revisión bibliográfica (`Revision_bibliografica_Proyecto2.pdf`):
un **detector YOLOv8n** de una sola clase localiza el mosquito y un **clasificador** trabaja sobre
el recorte. Se comparan seis clasificadores y una app en Streamlit permite usarlos.

## Resultados principales

Partición de prueba estratificada (15 %, 1,553 imágenes que ningún modelo vio al entrenar).
La métrica principal es el **F1 macro**, que pesa igual a las seis clases sin importar su tamaño.

| Modelo | Familia | F1 macro | Exactitud balanceada | Exactitud |
|---|---|---|---|---|
| DINOv2 + SVM (RBF) | DINOv2 congelado, plano | **0.664** | 0.618 | 0.940 |
| Jerárquico en cascada | DINOv2 congelado, jerárquico | 0.662 | 0.689 | 0.907 |
| DINOv2 + Regresión logística | DINOv2 congelado, plano | 0.630 | 0.707 | 0.900 |
| Jerárquico taxonómico | DINOv2 congelado, jerárquico | 0.623 | 0.684 | 0.896 |
| ConvNeXt-Tiny | CNN con ajuste fino | 0.565 | 0.606 | 0.892 |
| EfficientNet-B0 | CNN con ajuste fino | 0.540 | 0.568 | 0.876 |

| Detector YOLOv8n (test) | Valor |
|---|---|
| mAP@50 / mAP@50-95 | 0.979 / 0.723 |
| IoU medio de la caja de mayor confianza | 0.831 |
| Imágenes con IoU ≥ 0.75 (umbral del reto) | 83.3 % |

Hallazgos:
- Las características de DINOv2 congelado con un clasificador simple superan a las dos CNN con
  ajuste fino, con una fracción del tiempo de entrenamiento.
- El desempeño cae con el tamaño de la clase: *Ae. aegypti* (33 imágenes de entrenamiento, 7 de
  prueba) queda cerca de F1 = 0 en casi todos los modelos. Con tan pocas imágenes de prueba, las
  métricas de *aegypti* y *Anopheles* tienen mucha varianza.
- La cascada {albopictus, Culex, otro} es la única variante que recupera algo de *aegypti* y mejora
  *Anopheles* frente al modelo plano con el mismo backbone.

El detalle completo (hiperparámetros, métricas por clase, matrices de confusión, curvas y
detector) está en `results/resumen_entrenamiento.md`, el archivo de resultados para el informe.

## Estructura del repositorio

```
README.md
environment.yml, requirements.txt
run_training.sh                    pipeline de entrenamiento completo
Revision_bibliografica_Proyecto2.pdf
Proyecto 2. Resultados. 2026.pdf   guía del curso (rúbrica)
data/
  raw/                             train.csv e images/ del reto (no versionado, ~10 GB)
  processed/                       splits.csv, recortes, dataset YOLO, características (no versionado)
notebooks/
  01_EDA_mosquito_species.ipynb    análisis exploratorio
  02_yolo_detector_colab.ipynb     entrenamiento del detector en Google Colab (alternativa)
src/
  prepare_data.py                  limpieza, partición estratificada, recortes y dataset YOLO
  common.py                        clases, particiones, métricas y modelo jerárquico compartidos
  extract_features.py              características DINOv2 ViT-S/14 congelado (GPU)
  train_feature_models.py          regresión logística, SVM y variantes jerárquicas (CPU)
  benchmark_gpu.py                 rendimiento de entrenamiento en la GPU
  train_cnn.py                     ajuste fino de EfficientNet-B0 y ConvNeXt-Tiny
  train_detector.py                detector YOLOv8n de una clase
  summarize_results.py             results/summary.csv y figuras base
  make_figures.py                  figuras de presentación con temas personalizables
app/
  streamlit_app.py                 aplicación (ver abajo)
  inference.py                     detector + recorte + clasificadores; autoverificación
  summary.py                       genera results/resumen_entrenamiento.md
models/                            modelos entrenados (.joblib, .pt) y detector/ de YOLO
results/                           métricas por modelo (json), predicciones de test (csv),
                                   historiales, summary.csv, benchmark y resumen para el informe
logs/                              salida de cada paso del entrenamiento
reports/
  figures/                         figuras base, presentacion/ y presentacion_oscuro/
  themes/                          temas de las figuras (base.json, dark.json)
```

## Entorno de trabajo

```
conda env create -f environment.yml
conda activate mosquito-ml
```

`environment.yml` instala PyTorch con CUDA desde el índice de pytorch.org (cu126), además de
timm, ultralytics, scikit-learn, Streamlit y Plotly. `requirements.txt` es la alternativa con pip.
Recursos usados: laptop con GPU GTX 1650 (4 GB), 16 GB de RAM y procesador i5 de 8 hilos.

## Obtención de los datos

1. Crear una cuenta en AIcrowd y unirse al reto MosquitoAlert Challenge 2023.
2. Descargar `train.csv` y la carpeta de imágenes desde la pestaña Resources del reto.
3. Ubicar los archivos en `data/raw/train.csv` y `data/raw/images/`.

El conjunto real tiene **10,357 imágenes** (no las 8,025 que reporta la página del reto). Tras
descartar 3 cajas inválidas quedan **10,354**. Si los archivos no existen, el notebook de EDA
genera un conjunto sintético con la distribución oficial para poder probar el análisis.

## Análisis exploratorio

`notebooks/01_EDA_mosquito_species.ipynb` sigue el orden de la rúbrica: situación problemática,
problema científico, objetivos, descripción de los datos, limpieza y preprocesamiento, análisis
exploratorio y hallazgos. Antes de subir cambios, correrlo completo (Kernel > Restart & Run All).

## Entrenamiento

Desde la raíz del repositorio:

```
python src/prepare_data.py      # una vez: splits.csv, recortes 256 px y dataset YOLO (~4 min)
bash run_training.sh            # todos los modelos, resumen y figuras base
```

`run_training.sh` corre los trabajos de GPU uno a la vez (4 GB no alcanzan para dos) y los
modelos sobre DINOv2 en CPU en paralelo con las CNN. Los pasos ya completados se omiten, así que
se puede reanudar. Tiempos medidos en la GTX 1650:

| Paso | Tiempo |
|---|---|
| Preprocesamiento (7 procesos) | ~4 min |
| Extracción de características DINOv2 | ~6 min |
| Regresión logística, SVM y jerárquicos (CPU) | ~2.5 min |
| EfficientNet-B0, 6 épocas | ~9 min |
| ConvNeXt-Tiny, 3 épocas | ~18 min |
| YOLOv8n, 6 épocas (tope de 18 min) | ~20 min |

Decisiones de entrenamiento:
- **Partición** estratificada 70/15/15 con semilla 42, hecha antes de cualquier sobremuestreo.
- **Desbalance** (Buda et al., 2018): pesos balanceados en los modelos sobre DINOv2; sobremuestreo
  hasta clases equiprobables, aumento de datos y label smoothing en las CNN.
- **Selección de modelo** por F1 macro en validación; test se usa solo para el reporte final.
- **Cómputo** (`results/gpu_benchmark.csv`): en la GTX 1650, `channels_last` bajó EfficientNet-B0
  de 121 a 16 img/s, así que se entrena con AMP en formato NCHW y batch 32. ConvNeXt-Tiny no
  supera 13 img/s a 224 px, por eso se ajusta a 176 px con el stem y las dos primeras etapas
  congeladas.

## Figuras para la presentación

```
python src/make_figures.py                                    # tema claro -> reports/figures/presentacion/
python src/make_figures.py --theme reports/themes/dark.json   # tema oscuro -> presentacion_oscuro/
python src/make_figures.py --only ranking,confusion           # solo algunas figuras
python src/make_figures.py --list                             # nombres disponibles
```

Diez figuras en PNG y SVG (texto editable): distribución de clases, ranking por F1 macro, F1 por
clase, recall según tamaño de clase, matriz de confusión del mejor modelo, costo frente a
desempeño, curvas de entrenamiento, plano frente a jerárquico, IoU del detector y benchmark de GPU.

Todo lo visual vive en `reports/themes/base.json`: colores, tipografía, tamaños, títulos, nombres
de modelos y clases, formatos, dpi y fondo transparente. Para adaptarlo a una plantilla se crea un
JSON con `"extends": "base.json"` y solo las claves que cambian (ver `dark.json`). La paleta de
tres familias está validada para daltonismo en ambos temas; si se cambia, conviene mantener tres
colores bien distinguibles.

## Aplicación local

App en Streamlit (backend y frontend en el mismo proceso de Python, sin despliegue web). Usa los
modelos de `models/` y los resultados de `results/`, así que primero debe haber corrido el
entrenamiento (`run_training.sh`).

```
conda activate mosquito-ml
streamlit run app/streamlit_app.py      # abre http://localhost:8501
```

- **Cómo funciona:** el flujo detector → recorte → clasificadores y los hallazgos principales.
- **Clasificar una imagen:** subir una foto (o usar una de prueba); el preprocesamiento (detección
  con YOLOv8n y recorte idéntico al del entrenamiento) es invisible para el usuario. Se elige un
  modelo, varios o todos, y se ve el consenso.
- **Explorar los datos:** distribución de clases y variables del conjunto limpio, recortes por clase.
- **Rendimiento de los modelos:** gráficas interactivas con los datos de prueba; se ocultan con el
  interruptor de la barra lateral.

`python app/inference.py` verifica que la app reproduce las predicciones de test guardadas.

## Resumen de resultados para el informe

```
python app/summary.py                   # -> results/resumen_entrenamiento.md
```

Reúne datos y partición, preprocesamiento, recursos de cómputo, hiperparámetros y búsqueda de C,
curvas de entrenamiento, métricas de test (globales, por clase, matrices de confusión), acuerdo
entre modelos y resultados del detector. La app ofrece el mismo resumen más métricas y figuras
como `.zip` (botón en la barra lateral). `results/summary.csv` tiene la tabla de métricas para
armar tablas propias.

## Flujo de trabajo en git

Un commit por avance concreto (un paso del pipeline, un modelo, una corrección), no un único
commit al final; los mensajes explican la decisión y su medición. La evaluación individual
depende del historial de contribuciones de cada integrante en el repositorio.

## Entregables

- Informe en PDF con la investigación y la discusión de resultados
- Código en Python (este repositorio)
- Enlace a este repositorio
- Presentación con los resultados
