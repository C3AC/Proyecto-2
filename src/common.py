"""Utilidades compartidas: clases, particiones y métricas. Cada modelo escribe su propio
archivo en results/ para que entrenamientos en paralelo nunca escriban el mismo archivo."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["aegypti", "albopictus", "anopheles", "culex", "culiseta", "japonicus/koreicus"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SPLITS_CSV = "data/processed/splits.csv"
CROPS_DIR = "data/processed/crops"
RESULTS_DIR = "results"
MODELS_DIR = "models"


def load_splits():
    df = pd.read_csv(SPLITS_CSV)
    df["stem"] = df["img_fName"].str.replace(r"\.[^.]+$", "", regex=True)
    df["crop_path"] = CROPS_DIR + "/" + df["stem"] + ".jpg"
    df["y"] = df["class_label"].map(CLASS_TO_IDX)
    return df


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_per_class": dict(zip(CLASSES, f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist())),
        "recall_per_class": dict(zip(CLASSES, recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_results(name, test_df, proba, extra):
    """Guarda métricas de test (json) y probabilidades por imagen (csv) del modelo `name`."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred = np.asarray(proba).argmax(1)
    out = {"model": name, **compute_metrics(test_df["y"].to_numpy(), pred), **extra}
    with open(os.path.join(RESULTS_DIR, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    preds = pd.DataFrame(proba, columns=CLASSES)
    preds.insert(0, "y_true", test_df["class_label"].to_numpy())
    preds.insert(0, "stem", test_df["stem"].to_numpy())
    preds["y_pred"] = [CLASSES[i] for i in pred]
    preds.to_csv(os.path.join(RESULTS_DIR, f"test_preds_{name}.csv"), index=False)
    print(f"[{name}] macro F1 = {out['macro_f1']:.4f} | balanced acc = {out['balanced_accuracy']:.4f}")
    return out


# En common.py (no en el script) para que joblib pueda cargar los modelos desde la app
def logreg(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


class TwoLevel:
    """Modelo jerárquico: P(clase) = P(grupo) * P(clase | grupo); los grupos de una sola clase pasan directo."""

    def __init__(self, groups, C):
        self.groups, self.C = groups, C

    def fit(self, X, y_names):
        g = np.array([self.groups[c] for c in y_names])
        self.top = logreg(self.C).fit(X, g)
        self.sub = {}
        for grp in set(self.groups.values()):
            members = [c for c in CLASSES if self.groups[c] == grp]
            if len(members) > 1:
                mask = g == grp
                self.sub[grp] = logreg(self.C).fit(X[mask], y_names[mask])
        return self

    def predict_proba(self, X):
        p_top = dict(zip(self.top.classes_, self.top.predict_proba(X).T))
        out = np.zeros((len(X), len(CLASSES)))
        for i, c in enumerate(CLASSES):
            grp = self.groups[c]
            if grp in self.sub:
                m = self.sub[grp]
                out[:, i] = p_top[grp] * m.predict_proba(X)[:, list(m.classes_).index(c)]
            else:
                out[:, i] = p_top[grp]
        return out
