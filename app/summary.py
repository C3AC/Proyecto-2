"""Resumen de entrenamiento y prueba para el informe. Lee todo lo que dejó el pipeline (results/, logs/,
models/detector/, reports/figures/) y escribe un solo archivo Markdown; se puede correr las veces que se quiera.
    python app/summary.py            ->  results/resumen_entrenamiento.md
La app ofrece el mismo resumen (más summary.csv y las figuras) como .zip descargable.
"""
import glob
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES, compute_metrics  # noqa: E402

OUT = ROOT / "results" / "resumen_entrenamiento.md"
FIGURES = {
    "res_01_comparacion_modelos.png": "F1 macro, exactitud balanceada y exactitud de cada modelo en test",
    "res_02_f1_por_clase.png": "Mapa de calor del F1 por clase y modelo",
    "res_03_confusion_mejor_modelo.png": "Matriz de confusión normalizada del mejor modelo",
    "res_04_costo_vs_f1.png": "F1 macro frente al costo de inferencia por imagen",
    "res_05_curvas_entrenamiento.png": "Pérdida y F1 de validación por época de las CNN",
    "res_06_iou_detector.png": "Distribución del IoU del detector frente al umbral 0.75 del reto",
}


def table(df, floatfmt="{:.3f}"):
    """DataFrame -> tabla Markdown (sin depender de tabulate)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load_runs():
    runs = [json.load(open(p, encoding="utf-8")) for p in glob.glob(str(ROOT / "results" / "metrics_*.json"))]
    return sorted(runs, key=lambda r: -r["macro_f1"])


def tuning_from_log():
    """{modelo: [(C, F1 val)]} a partir de logs/feature_models.log."""
    path, out, cur = ROOT / "logs" / "feature_models.log", {}, None
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if m := re.match(r"\[(\w+)\] tuning", line):
            cur = m.group(1)
            out[cur] = []
        elif cur and (m := re.match(r"\s+C=(\S+)\s+val F1=(\S+)", line)):
            out[cur].append((float(m.group(1)), float(m.group(2))))
    return out


def environment():
    pkgs = ["torch", "torchvision", "timm", "ultralytics", "scikit-learn", "numpy", "pandas", "streamlit", "plotly"]
    vers = []
    for p in pkgs:
        try:
            vers.append(f"{p} {version(p)}")
        except PackageNotFoundError:
            pass
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = ""
    return f"Python {sys.version.split()[0]}; " + ", ".join(vers), gpu


def ensemble_and_agreement(runs):
    """Promedio de probabilidades de TODOS los modelos (sin elegir miembros con test) y acuerdo entre modelos."""
    preds = {r["model"]: pd.read_csv(p) for r in runs if (p := ROOT / "results" / f"test_preds_{r['model']}.csv").exists()}
    if len(preds) < 2:
        return None
    ref = next(iter(preds.values()))
    if not all((d["stem"].to_numpy() == ref["stem"].to_numpy()).all() for d in preds.values()):
        return None
    y_true = ref["y_true"].to_numpy()
    ens = compute_metrics(pd.Series(y_true).map({c: i for i, c in enumerate(CLASSES)}).to_numpy(),
                          np.mean([d[CLASSES].to_numpy() for d in preds.values()], 0).argmax(1))
    labels = np.array([d["y_pred"].to_numpy() for d in preds.values()])
    all_agree = (labels == labels[0]).all(0)
    acc_agree = (labels[0] == y_true)[all_agree].mean()
    acc_disagree = (labels == y_true).mean(0)[~all_agree].mean()
    return ens, list(preds), all_agree.mean(), acc_agree, acc_disagree


def build():
    runs = load_runs()
    if not runs:
        raise SystemExit("No hay results/metrics_*.json todavía")
    splits = pd.read_csv(ROOT / "data" / "processed" / "splits.csv")
    raw_path = ROOT / "data" / "raw" / "train.csv"
    n_raw = len(pd.read_csv(raw_path)) if raw_path.exists() else None
    tuning = tuning_from_log()
    env, gpu = environment()
    best = runs[0]
    md = []
    w = md.append

    w("# Resumen de entrenamiento y prueba — Identificación de especies de mosquitos")
    w(f"Generado automáticamente el {datetime.now():%Y-%m-%d %H:%M} a partir de `results/`, `logs/` y `models/` "
      "(`python app/summary.py`). Todas las cifras de test se calcularon sobre imágenes que ningún modelo vio al entrenar.\n")

    w("## 1. Datos y partición")
    if n_raw:
        w(f"- Conjunto original (`train.csv` del MosquitoAlert Challenge 2023): **{n_raw:,} imágenes**, una caja por imagen.")
        w(f"- Limpieza: se descartaron **{n_raw - len(splits)}** filas con cajas geométricamente inválidas "
          f"(xbr ≤ xtl, ybr ≤ ytl o fuera de la imagen). Quedan **{len(splits):,}** imágenes.")
    w("- Partición estratificada por clase 70 / 15 / 15 (entrenamiento / validación / prueba), semilla 42, "
      "hecha **antes** de cualquier sobremuestreo para no filtrar copias entre particiones.")
    w("- Validación: selección de hiperparámetros y de la mejor época. Prueba: solo para las cifras finales.\n")
    ct = pd.crosstab(splits["class_label"], splits["split"]).reindex(CLASSES)[["train", "val", "test"]]
    ct["total"] = ct.sum(axis=1)
    ct["% del total"] = ct["total"] / ct["total"].sum() * 100
    ct.loc["Total"] = ct.sum()
    ct = ct.astype({c: int for c in ["train", "val", "test", "total"]})
    w(table(ct.reset_index().rename(columns={"class_label": "clase"}), "{:.2f}"))
    counts = splits["class_label"].value_counts()
    w(f"\nDesbalance: la clase mayoritaria ({counts.index[0]}, {counts.iloc[0]:,}) tiene "
      f"{counts.iloc[0] / counts.iloc[-1]:.0f} veces más imágenes que la minoritaria ({counts.index[-1]}, {counts.iloc[-1]}). "
      "Por eso la métrica principal es el **F1 macro** (métrica oficial del reto), que pesa igual a cada clase.\n")

    w("## 2. Preprocesamiento (`src/prepare_data.py`)")
    w("- Recorte cuadrado centrado en la caja anotada con **15 % de margen**, redimensionado a **256 × 256 px** "
      "(entrada de todos los clasificadores). La app aplica exactamente el mismo recorte a la caja del detector.")
    w("- Para el detector: imagen completa reescalada a 640 px en el lado mayor y etiqueta YOLO de una sola clase («mosquito»).")
    w("- Decodificación JPEG a media resolución (`draft`) para acelerar ~4× el preprocesamiento de ~9 GB de imágenes.")
    w("- CNN: aumento de datos (RandomResizedCrop 0.6–1.0, volteos horizontal y vertical, rotación ±20°, ColorJitter) "
      "y **sobremuestreo con WeightedRandomSampler hasta clases equiprobables** (Buda et al., 2018); evaluación con recorte central de 224 px.")
    w("- DINOv2: características congeladas de 384 dimensiones (ViT-S/14, entrada 224 px) extraídas una sola vez; "
      "clasificadores con `class_weight='balanced'` y estandarización.\n")

    w("## 3. Recursos de cómputo y herramientas")
    if gpu:
        w(f"- GPU: {gpu}. Entrenamiento local (sin Colab); una tarea de GPU a la vez, los modelos sobre DINOv2 en CPU en paralelo.")
    w(f"- Software: {env}.")
    bench = ROOT / "results" / "gpu_benchmark.csv"
    if bench.exists():
        w("- Medición previa de rendimiento en la GPU (imágenes/s de entrenamiento), que justificó usar AMP sin `channels_last` "
          "y reducir ConvNeXt-Tiny a 176 px con las dos primeras etapas congeladas:\n")
        w(table(pd.read_csv(bench), "{:.1f}"))
    w("")

    w("## 4. Modelos entrenados y ajuste de hiperparámetros")
    rows = []
    for r in runs:
        hp = r.get("hyperparams", {})
        if "C" in r:
            conf = f"C = {r['C']}"
        else:
            conf = (f"{hp.get('epochs')} épocas, batch {hp.get('batch')}, lr {hp.get('lr')}, {hp.get('img_size')} px"
                    + (f", {hp['freeze_stages']} etapas congeladas" if hp.get("freeze_stages") else "")
                    + ", AdamW + OneCycle, label smoothing 0.1")
        rows.append({"modelo": r["model"], "familia": r.get("family", ""), "configuración elegida": conf,
                     "F1 macro val": r.get("best_val_macro_f1"),
                     "entrenamiento (min)": (r.get("train_seconds") or np.nan) / 60,
                     "inferencia (ms/img)": r.get("infer_ms_per_img"),
                     "parámetros entrenables": f"{r['trainable_params']:,}" if "trainable_params" in r else "—"})
    w(table(pd.DataFrame(rows)))
    w("\nNotas: el tiempo de inferencia de los modelos DINOv2 incluye la extracción de características en GPU; "
      "los jerárquicos reutilizan el C de la regresión logística plana.")
    if tuning:
        w("\n**Búsqueda de C (F1 macro en validación, ✔ = elegido):**\n")
        for name, grid in tuning.items():
            best_c = max(grid, key=lambda g: g[1])[0]
            w(f"- `{name}`: " + ", ".join(f"C={c:g} → {f:.4f}" + (" ✔" if c == best_c else "") for c, f in grid))
    for p in sorted(glob.glob(str(ROOT / "results" / "history_*.csv"))):
        h = pd.read_csv(p)
        h["sec"] = h["sec"].round().astype(int)
        w(f"\n**Curva de entrenamiento `{Path(p).stem[8:]}`** (se guarda la época con mejor F1 en validación):\n")
        w(table(h.rename(columns={"epoch": "época", "train_loss": "pérdida train", "val_macro_f1": "F1 macro val",
                                  "sec": "segundos"}), "{:.4f}"))
    w("")

    w("## 5. Resultados en el conjunto de prueba")
    w(f"{int(np.sum(best['confusion_matrix'])):,} imágenes de prueba. Ordenado por F1 macro.\n")
    w(table(pd.DataFrame([{"modelo": r["model"], "F1 macro": r["macro_f1"], "exactitud balanceada": r["balanced_accuracy"],
                           "exactitud": r["accuracy"]} for r in runs])))
    w(f"\n**Mejor modelo: `{best['model']}`** (F1 macro {best['macro_f1']:.3f}). La exactitud global es alta para todos "
      "porque las dos clases mayoritarias dominan; el F1 macro muestra la diferencia en las clases raras.\n")
    w("**F1 por clase:**\n")
    w(table(pd.DataFrame([{"modelo": r["model"], **r["f1_per_class"]} for r in runs])))
    w("\n**Recall por clase** (proporción de cada clase real que el modelo reconoce):\n")
    w(table(pd.DataFrame([{"modelo": r["model"], **r["recall_per_class"]} for r in runs])))
    worst = min(best["f1_per_class"], key=best["f1_per_class"].get)
    w(f"\nClase más difícil para el mejor modelo: **{worst}** (F1 {best['f1_per_class'][worst]:.2f}, "
      f"{int(np.sum(best['confusion_matrix'][CLASSES.index(worst)]))} imágenes en test).\n")
    w("**Matrices de confusión** (filas = clase real, columnas = predicción):\n")
    for r in runs:
        cm = pd.DataFrame(r["confusion_matrix"], columns=CLASSES)
        cm.insert(0, "real \\ pred", CLASSES)
        w(f"`{r['model']}`\n")
        w(table(cm))
        w("")

    ea = ensemble_and_agreement(runs)
    if ea:
        ens, members, agree, acc_agree, acc_disagree = ea
        w("## 6. Análisis complementario: combinación y acuerdo entre modelos")
        w(f"- Promedio de las probabilidades de **todos** los modelos ({', '.join(members)}), sin elegir miembros con "
          f"el conjunto de prueba: F1 macro **{ens['macro_f1']:.3f}**, exactitud balanceada {ens['balanced_accuracy']:.3f}, "
          f"exactitud {ens['accuracy']:.3f} (mejor modelo individual: {best['macro_f1']:.3f}).")
        w("  F1 por clase del promedio: " + ", ".join(f"{c} {v:.2f}" for c, v in ens["f1_per_class"].items()) + ".")
        w(f"- Todos los modelos coinciden en el **{agree:.1%}** de las imágenes de prueba; cuando coinciden aciertan el "
          f"{acc_agree:.1%} de las veces, y cuando discrepan cada modelo acierta en promedio solo el {acc_disagree:.1%}. "
          "El desacuerdo entre modelos es una buena señal de imagen dudosa (la app lo muestra como «consenso»).\n")

    w("## 7. Detector YOLOv8n (localización)")
    det = ROOT / "results" / "detector_summary.json"
    if det.exists():
        d = json.loads(det.read_text())
        w(f"- Una sola clase («mosquito»), entrenado con las cajas de las seis clases. "
          f"mAP@50 = **{d.get('map50', float('nan')):.3f}**, mAP@50-95 = {d.get('map50_95', float('nan')):.3f} en test.")
        w(f"- IoU medio de la caja de mayor confianza: **{d['mean_iou']:.3f}**; imágenes con IoU ≥ 0.75 (umbral del reto): "
          f"**{d['iou_ge_075']:.1%}** de {d['n_test']:,}.")
        if "train_seconds" in d:
            w(f"- Entrenamiento: {d['train_seconds'] / 60:.1f} min; inferencia: {d['infer_ms_per_img']:.1f} ms por imagen. "
              f"Hiperparámetros: {d.get('hyperparams')}.")
    else:
        w("- Pendiente: todavía no existe `results/detector_summary.json` (correr `src/train_detector.py`).")
    w("")

    w("## 8. Figuras estáticas para el informe (`reports/figures/`)")
    figs = sorted(glob.glob(str(ROOT / "reports" / "figures" / "*.png")))
    for f in figs:
        w(f"- `{Path(f).name}`: {FIGURES.get(Path(f).name, '')}")
    if not figs:
        w("- Aún no generadas: correr `python src/summarize_results.py`.")
    return "\n".join(md) + "\n"


def write():
    OUT.write_text(build(), encoding="utf-8")
    return OUT


def zip_bytes():
    """Resumen + summary.csv + métricas + figuras, para descargar desde la app."""
    files = [ROOT / "results" / "summary.csv", ROOT / "results" / "detector_summary.json",
             *(Path(p) for pat in ("results/metrics_*.json", "results/history_*.csv", "reports/figures/*.png")
               for p in glob.glob(str(ROOT / pat)))]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("resumen_entrenamiento.md", build())
        for p in files:
            if p.exists():
                z.write(p, f"{p.parent.name}/{p.name}")
    return buf.getvalue()


if __name__ == "__main__":
    print("escrito:", write())
