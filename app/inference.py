"""Backend de la app: preprocesamiento (detector -> recorte) y predicción con los modelos entrenados.
Reutiliza el código de src/ para que la app procese las imágenes exactamente igual que el entrenamiento.

Autoverificación (compara contra las predicciones de test guardadas por cada modelo):
    python app/inference.py
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # los módulos de src/ usan rutas relativas a la raíz del repo

from common import CLASSES  # noqa: E402
from extract_features import build_model as build_dinov2  # noqa: E402
from prepare_data import CROP_SIZE, MARGIN  # noqa: E402
from train_cnn import transforms as cnn_transforms  # noqa: E402

DEVICE = "cpu"  # la GPU queda libre para entrenar; una imagen en CPU tarda < 1 s
DETECTOR = ROOT / "models" / "detector" / "weights" / "best.pt"

LABELS = {
    "dinov2_svm": "DINOv2 + SVM",
    "dinov2_logreg": "DINOv2 + Regresión logística",
    "dinov2_hier_taxonomic": "DINOv2 + Jerárquico taxonómico",
    "dinov2_hier_cascade": "DINOv2 + Jerárquico en cascada",
    "efficientnet_b0": "EfficientNet-B0 (ajuste fino)",
    "convnext_tiny": "ConvNeXt-Tiny (ajuste fino)",
}


def load_metrics():
    """{modelo: métricas de test} para los modelos cuyo archivo entrenado existe, del mejor al peor."""
    out = {}
    for p in sorted((ROOT / "results").glob("metrics_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        name = m["model"]
        if (ROOT / "models" / f"{name}.joblib").exists() or (ROOT / "models" / f"{name}.pt").exists():
            out[name] = m
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["macro_f1"]))


def square_crop(im, box):
    """Mismo recorte que src/prepare_data.py: cuadrado centrado en la caja + 15 % de margen, 256 px."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * MARGIN / 2
    crop = im.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))
    return crop.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


class Pipeline:
    def __init__(self):
        self.metrics = load_metrics()
        self.detector = None
        if DETECTOR.exists():
            from ultralytics import YOLO
            self.detector = YOLO(str(DETECTOR))
        self.dinov2 = self.dinov2_tf = None
        self.models = {}
        for name, m in self.metrics.items():
            if m.get("family", "").startswith("DINOv2"):
                self.models[name] = joblib.load(ROOT / "models" / f"{name}.joblib")
                if self.dinov2 is None:
                    self.dinov2, self.dinov2_tf = build_dinov2()
            else:
                hp = m["hyperparams"]
                import timm
                net = timm.create_model(hp["arch"], pretrained=False, num_classes=len(CLASSES))
                state = torch.load(ROOT / "models" / f"{name}.pt", map_location=DEVICE)
                net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
                self.models[name] = (net.eval(), cnn_transforms(hp["img_size"])[1])

    def locate(self, im):
        """Caja (x1, y1, x2, y2) del mosquito y confianza. Sin detector, o sin detecciones, usa la imagen completa."""
        if self.detector is not None:
            r = self.detector.predict(im, imgsz=512, conf=0.05, device=DEVICE, verbose=False)[0]
            if len(r.boxes):
                k = int(r.boxes.conf.argmax())
                return tuple(r.boxes.xyxy[k].tolist()), float(r.boxes.conf[k])
        return (0, 0, im.width, im.height), None

    def preprocess(self, file):
        """Imagen subida -> (imagen orientada, caja, confianza, recorte de 256 px)."""
        im = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        box, conf = self.locate(im)
        return im, box, conf, square_crop(im, box)

    @torch.no_grad()
    def predict(self, crop, names=None):
        """{modelo: (probabilidades por clase, ms)} sobre un recorte ya preprocesado."""
        out, feats = {}, None
        for name in names or self.models:
            t = time.perf_counter()
            model = self.models[name]
            if isinstance(model, tuple):
                net, tf = model
                proba = net(tf(crop)[None]).softmax(1)[0].numpy()
            else:
                if feats is None:
                    feats = self.dinov2(self.dinov2_tf(crop)[None]).numpy()
                proba = np.asarray(model.predict_proba(feats))[0]
            out[name] = (proba, (time.perf_counter() - t) * 1000)
        return out


if __name__ == "__main__":
    import pandas as pd

    pipe = Pipeline()
    print("modelos:", list(pipe.models), "| detector:", pipe.detector is not None)
    for name in pipe.models:
        ref = pd.read_csv(ROOT / "results" / f"test_preds_{name}.csv").sample(8, random_state=0)
        agree, diffs = 0, []
        for _, row in ref.iterrows():
            crop = Image.open(ROOT / "data" / "processed" / "crops" / f"{row['stem']}.jpg").convert("RGB")
            proba = pipe.predict(crop, [name])[name][0]
            agree += CLASSES[int(proba.argmax())] == row["y_pred"]
            diffs.append(np.abs(proba - row[CLASSES].to_numpy(float)).max())
        print(f"{name:24s} coincide {agree}/8 | máx |Δp| = {max(diffs):.3f}")
        # Entrenamiento en GPU con fp16 vs app en CPU fp32: diferencias pequeñas, misma clase
        assert agree >= 7 and max(diffs) < 0.15, f"{name}: el preprocesamiento de la app no coincide con el de test"
    print("OK: la app reproduce las predicciones de test")
