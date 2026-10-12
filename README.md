# Identificacion de Especies de Mosquitos - Analisis Exploratorio

Proyecto 2 del curso CC3084 (Data Science), UVG, semestre II 2026. El grupo trabaja sobre el
reto MosquitoAlert Challenge 2023 (AIcrowd): clasificacion de especies/generos de mosquitos a
partir de imagenes capturadas por ciudadanos, con anotaciones de bounding box provistas por
entomologos.

Pagina del reto: https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023

## Estructura del repositorio

```
proyecto2_reto17_mosquitos/
  README.md
  environment.yml
  requirements.txt
  data/
    raw/            train.csv y carpeta de imagenes del dataset original (no versionado)
    processed/      datasets intermedios generados durante la limpieza
  notebooks/
    01_EDA_mosquito_species.ipynb
  reports/
    figures/        graficos exportados para el informe y la presentacion
```

Los directorios `data/raw` y `data/processed` se excluyen del control de versiones por su
tamano (el dataset completo pesa cerca de 9 GB); se mantienen en el repositorio solo como
estructura vacia mediante archivos `.gitkeep`.

## Entorno de trabajo

El proyecto usa un entorno de conda. El archivo `environment.yml` fija las versiones de las
librerias principales; `requirements.txt` se deja como alternativa para quien prefiera pip.

## Obtencion de los datos

1. Crear una cuenta en AIcrowd y unirse al reto MosquitoAlert Challenge 2023.
2. Descargar `train.csv` y la carpeta de imagenes desde la pestana Resources del reto.
3. Ubicar los archivos en:
   ```
   data/raw/train.csv
   data/raw/images/
   ```

El notebook detecta automaticamente si estos archivos existen. Mientras no esten disponibles,
genera un conjunto de datos sintetico que respeta la distribucion de clases reportada en la
pagina oficial del reto, de forma que el resto del analisis se pueda desarrollar y probar sin
esperar la descarga completa.

## Notebook principal

`notebooks/01_EDA_mosquito_species.ipynb` sigue el orden de la rubrica del curso: situacion
problematica, problema cientifico, objetivos, descripcion de los datos, limpieza y
preprocesamiento, analisis exploratorio y hallazgos. Cada seccion indica el responsable y marca
con `# TODO` los puntos donde falta investigacion, interpretacion o una decision documentada. El
codigo de carga de datos, limpieza y analisis exploratorio ya esta implementado y probado.

Antes de subir cambios, correr el notebook completo de inicio a fin (Kernel > Restart & Run All)
para confirmar que no quedan celdas rotas.

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
como `.zip` (botón en la barra lateral). Las figuras estáticas salen de `python src/summarize_results.py`.

## Flujo de trabajo en git

Se espera un commit por avance concreto (una seccion del notebook, una correccion, un dataset
procesado), no un unico commit al final. La evaluacion individual depende del historial de
contribuciones de cada integrante en el repositorio.

## Entregables

- Informe de analisis exploratorio en PDF
- Enlace a este repositorio
- Presentacion en PowerPoint con los resultados
