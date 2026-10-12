"""Junta los resultados de todos los modelos y genera las figuras estáticas del informe.
    python src/summarize_results.py

Lee results/metrics_*.json, results/history_*.csv y (si existe) models/detector/det_test_boxes.csv.
Escribe results/summary.csv y reports/figures/*.png. Se puede correr las veces que se quiera.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import CLASSES, RESULTS_DIR

FIG = "reports/figures"
DET = "models/detector/det_test_boxes.csv"
sns.set_theme(style="whitegrid", palette="colorblind")


def load():
    rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(f"{RESULTS_DIR}/metrics_*.json"))]
    if not rows:
        raise SystemExit("No hay results/metrics_*.json todavía")
    return sorted(rows, key=lambda r: r["macro_f1"], reverse=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=160)
    plt.close(fig)
    print("  ", name)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load()
    summary = pd.DataFrame([{
        "model": r["model"], "family": r.get("family", ""), "macro_f1": r["macro_f1"],
        "balanced_accuracy": r["balanced_accuracy"], "accuracy": r["accuracy"],
        "best_val_macro_f1": r.get("best_val_macro_f1"), "train_seconds": r.get("train_seconds"),
        "infer_ms_per_img": r.get("infer_ms_per_img"),
        **{f"f1_{c}": r["f1_per_class"][c] for c in CLASSES},
        **{f"recall_{c}": r["recall_per_class"][c] for c in CLASSES},
    } for r in runs])
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print(summary[["model", "macro_f1", "balanced_accuracy", "accuracy", "train_seconds", "infer_ms_per_img"]].to_string(index=False))

    # 1. Comparación global
    m = summary.melt(id_vars="model", value_vars=["macro_f1", "balanced_accuracy", "accuracy"], var_name="métrica")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=m, y="model", x="value", hue="métrica", ax=ax)
    ax.set(xlabel="Valor en test", ylabel="", xlim=(0, 1), title="Comparación de modelos (test)")
    save(fig, "res_01_comparacion_modelos.png")

    # 2. F1 por clase: muestra dónde fallan las clases minoritarias
    f1 = summary.set_index("model")[[f"f1_{c}" for c in CLASSES]]
    f1.columns = CLASSES
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(f1) + 2))
    sns.heatmap(f1, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set(title="F1 por clase en test", xlabel="", ylabel="")
    save(fig, "res_02_f1_por_clase.png")

    # 3. Matriz de confusión normalizada del mejor modelo
    best = runs[0]
    cm = np.array(best["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm.astype(int), fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar_kws={"label": "proporción de la fila"})
    ax.set(title=f"Matriz de confusión — {best['model']}", xlabel="Predicción", ylabel="Real")
    save(fig, "res_03_confusion_mejor_modelo.png")

    # 4. Costo vs desempeño
    s = summary.dropna(subset=["infer_ms_per_img"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=s, x="infer_ms_per_img", y="macro_f1", hue="family", s=120, ax=ax)
    for _, r in s.iterrows():
        ax.annotate(r["model"], (r["infer_ms_per_img"], r["macro_f1"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set(xlabel="Inferencia (ms por imagen, GTX 1650)", ylabel="F1 macro (test)", title="Desempeño vs costo de inferencia")
    save(fig, "res_04_costo_vs_f1.png")

    # 5. Curvas de entrenamiento de las CNN
    hist = {os.path.basename(p)[8:-4]: pd.read_csv(p) for p in glob.glob(f"{RESULTS_DIR}/history_*.csv")}
    if hist:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for name, h in hist.items():
            a1.plot(h["epoch"], h["train_loss"], marker="o", label=name)
            a2.plot(h["epoch"], h["val_macro_f1"], marker="o", label=name)
        a1.set(title="Pérdida de entrenamiento", xlabel="Época")
        a2.set(title="F1 macro en validación", xlabel="Época", ylim=(0, 1))
        a2.legend()
        save(fig, "res_05_curvas_entrenamiento.png")

    # 6. Detector (Colab): distribución del IoU frente al umbral del reto
    if os.path.exists(DET):
        d = pd.read_csv(DET)
        ok = (d["iou"] >= 0.75).mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(d["iou"], bins=40, ax=ax)
        ax.axvline(0.75, color="crimson", ls="--", label=f"umbral 0.75 ({ok:.1%} lo supera)")
        ax.set(title="IoU de la caja predicha por YOLOv8n en test", xlabel="IoU", ylabel="imágenes")
        ax.legend()
        save(fig, "res_06_iou_detector.png")
        summary_path = os.path.join(RESULTS_DIR, "detector_summary.json")
        if not os.path.exists(summary_path):  # train_detector.py ya escribe uno más completo
            with open(summary_path, "w") as f:
                json.dump({"n_test": len(d), "mean_iou": d["iou"].mean(), "iou_ge_075": ok}, f, indent=2)


if __name__ == "__main__":
    main()
