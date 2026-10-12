"""Detector YOLOv8n de una clase ("mosquito") entrenado en local; equivale a
notebooks/02_yolo_detector_colab.ipynb. Uso:
    python src/train_detector.py --imgsz 512 --epochs 12 --batch 16

Escribe models/detector/ (weights/best.pt, results.csv, curvas) y
models/detector/det_test_boxes.csv con la caja de mayor confianza y su IoU por imagen de test.
"""
import argparse
import csv
import glob
import json
import os
import time

from ultralytics import YOLO

DATA = os.path.abspath("data/processed/yolo")
OUT = os.path.abspath("models")


def iou(a, b):  # cajas (cx, cy, w, h) normalizadas
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hours", type=float, default=None, help="tope de tiempo de entrenamiento (ultralytics `time`)")
    args = ap.parse_args()

    # Ultralytics resuelve rutas relativas contra su propia carpeta de datasets: usar ruta absoluta
    yaml_path = os.path.join(DATA, "data_local.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: mosquito\n")

    t0 = time.time()
    YOLO("yolov8n.pt").train(data=yaml_path, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                             workers=args.workers, patience=5, seed=42, amp=True, time=args.hours,
                             project=OUT, name="detector", exist_ok=True)
    train_sec = time.time() - t0

    best = YOLO(os.path.join(OUT, "detector", "weights", "best.pt"))
    m = best.val(data=yaml_path, split="test", imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                 project=OUT, name="detector_test", exist_ok=True)

    rows, t1 = [], time.time()
    paths = sorted(glob.glob(os.path.join(DATA, "images", "test", "*.jpg")))
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt = [float(v) for v in open(os.path.join(DATA, "labels", "test", stem + ".txt")).read().split()[1:]]
        r = best.predict(path, imgsz=args.imgsz, conf=0.001, verbose=False)[0]
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())
            pred = r.boxes.xywhn[k].tolist()
            rows.append([stem, float(r.boxes.conf[k]), *pred, iou(pred, gt)])
        else:
            rows.append([stem, 0.0, 0, 0, 0, 0, 0.0])
    infer_ms = (time.time() - t1) / len(paths) * 1000

    with open(os.path.join(OUT, "detector", "det_test_boxes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "conf", "cx", "cy", "w", "h", "iou"])
        w.writerows(rows)
    ious = [r[-1] for r in rows]
    summary = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "n_test": len(ious),
               "mean_iou": sum(ious) / len(ious), "iou_ge_075": sum(i >= 0.75 for i in ious) / len(ious),
               "train_seconds": train_sec, "infer_ms_per_img": infer_ms, "hyperparams": vars(args)}
    os.makedirs("results", exist_ok=True)
    with open("results/detector_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
