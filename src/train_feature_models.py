"""Clasificadores sobre características DINOv2 congeladas (CPU). Uso:
    python src/train_feature_models.py --jobs 2

Modelos: regresión logística y SVM planos (6 clases), y las dos variantes jerárquicas de la
revisión bibliográfica (Silla y Freitas, 2011), combinando probabilidades de ambos niveles:
  taxonómica: {Aedes, Anopheles, Culex, Culiseta} -> especie dentro de Aedes
  cascada:    {albopictus, Culex, otro}           -> {aegypti, anopheles, culiseta, japonicus/koreicus}
C se elige por F1 macro en validación; el grid corre en paralelo con joblib.
"""
import argparse
import os
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, MODELS_DIR, TwoLevel, load_splits, logreg, save_results

FEATS = "data/processed/dinov2_feats.npz"
AEDES = ["aegypti", "albopictus", "japonicus/koreicus"]
GENUS = {c: ("aedes" if c in AEDES else c) for c in CLASSES}
CASCADE_TOP = {c: (c if c in ("albopictus", "culex") else "otro") for c in CLASSES}


def svm(C):
    return make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", class_weight="balanced", probability=True, random_state=42))


def fit_eval(make, C, Xtr, ytr, Xva, yva):
    m = make(C).fit(Xtr, ytr)
    return C, f1_score(yva, m.predict(Xva), average="macro"), m


def tune(make, grid, Xtr, ytr, Xva, yva, jobs):
    res = Parallel(n_jobs=jobs)(delayed(fit_eval)(make, C, Xtr, ytr, Xva, yva) for C in grid)
    for C, f1, _ in res:
        print(f"   C={C:<6} val F1={f1:.4f}")
    return max(res, key=lambda r: r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    df = load_splits()
    npz = np.load(FEATS, allow_pickle=True)
    assert (npz["stem"] == df["stem"].to_numpy()).all(), "features y splits.csv desalineados"
    X, extract_ms = npz["feats"], float(npz["seconds"]) / len(df) * 1000
    idx = {s: (df.split == s).to_numpy() for s in ("train", "val", "test")}
    y, names = df["y"].to_numpy(), df["class_label"].to_numpy()
    Xtr, Xva, Xte = X[idx["train"]], X[idx["val"]], X[idx["test"]]
    te = df[idx["test"]].reset_index(drop=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    def finish(name, model, C, val_f1, train_sec, family):
        t = time.time()
        proba = model.predict_proba(Xte)
        infer_ms = (time.time() - t) / len(Xte) * 1000 + extract_ms
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
        save_results(name, te, proba, {"family": family, "C": C, "best_val_macro_f1": val_f1,
                                       "train_seconds": train_sec, "infer_ms_per_img": infer_ms})

    for name, make, grid in (("dinov2_logreg", logreg, [0.01, 0.1, 1.0, 10.0]),
                             ("dinov2_svm", svm, [1.0, 3.0, 10.0])):
        print(f"[{name}] tuning")
        t = time.time()
        C, f1, model = tune(make, grid, Xtr, y[idx["train"]], Xva, y[idx["val"]], args.jobs)
        finish(name, model, C, f1, time.time() - t, "DINOv2 congelado + clasificador plano")
        if name == "dinov2_logreg":
            best_C = C

    # Variantes jerárquicas con el mismo backbone y el C del modelo plano
    for name, groups in (("dinov2_hier_taxonomic", GENUS), ("dinov2_hier_cascade", CASCADE_TOP)):
        t = time.time()
        model = TwoLevel(groups, best_C).fit(Xtr, names[idx["train"]])
        val_f1 = f1_score(y[idx["val"]], model.predict_proba(Xva).argmax(1), average="macro")
        finish(name, model, best_C, val_f1, time.time() - t, "DINOv2 congelado + jerárquico (2 niveles)")


if __name__ == "__main__":
    main()
