"""Preprocesamiento único del dataset (correr desde la raíz del repo):
    python src/prepare_data.py

Genera en data/processed/:
  splits.csv        partición estratificada 70/15/15 (antes de cualquier sobremuestreo)
  crops/            recorte cuadrado de la caja anotada + 15 % de margen, 256 px (clasificadores)
  yolo/ + yolo.zip  imágenes a 640 px y etiquetas de una sola clase "mosquito" (detector en Colab)
"""
import csv
import os
import random
import shutil
from collections import defaultdict
from multiprocessing import Pool

from PIL import Image

RAW = "data/raw"
OUT = "data/processed"
SEED = 42
MARGIN = 1.15
CROP_SIZE = 256
YOLO_SIZE = 640


def load_clean_rows():
    rows = list(csv.DictReader(open(os.path.join(RAW, "train.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("img_w", "img_h"):
            r[k] = int(r[k])
        for k in ("bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr"):
            r[k] = float(r[k])
    # Misma regla de limpieza que el notebook de EDA (sección 6)
    return [r for r in rows
            if r["bbx_xbr"] > r["bbx_xtl"] and r["bbx_ybr"] > r["bbx_ytl"]
            and r["bbx_xbr"] <= r["img_w"] and r["bbx_ybr"] <= r["img_h"]]


def stratified_split(rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_label"]].append(r)
    rng = random.Random(SEED)
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        for i, r in enumerate(items):
            r["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return rows


def process(r):
    name = r["img_fName"]
    stem = os.path.splitext(name)[0]
    crop_path = os.path.join(OUT, "crops", stem + ".jpg")
    yolo_img = os.path.join(OUT, "yolo", "images", r["split"], stem + ".jpg")
    if os.path.exists(crop_path) and os.path.exists(yolo_img):
        return None
    im = Image.open(os.path.join(RAW, "images", name))
    im.draft("RGB", (im.width // 2, im.height // 2))  # decodificación JPEG reducida: ~4x más rápida
    im = im.convert("RGB")
    # Las cajas están en coordenadas de img_w/img_h; reescalar al tamaño decodificado
    sx, sy = im.width / r["img_w"], im.height / r["img_h"]
    x1, y1, x2, y2 = r["bbx_xtl"] * sx, r["bbx_ytl"] * sy, r["bbx_xbr"] * sx, r["bbx_ybr"] * sy

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * MARGIN / 2
    crop = im.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))
    crop.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC).save(crop_path, quality=92)

    scale = YOLO_SIZE / max(im.size)
    im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR).save(yolo_img, quality=90)
    W, H = im.size
    label = f"0 {cx / W:.6f} {cy / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
    with open(os.path.join(OUT, "yolo", "labels", r["split"], stem + ".txt"), "w") as f:
        f.write(label)
    return None


def main():
    rows = stratified_split(load_clean_rows())
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "yolo", "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT, "yolo", "labels", s), exist_ok=True)

    fields = ["img_fName", "img_w", "img_h", "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr", "class_label", "split"]
    with open(os.path.join(OUT, "splits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for i, _ in enumerate(pool.imap_unordered(process, rows, chunksize=16), 1):
            if i % 1000 == 0:
                print(f"{i}/{len(rows)}", flush=True)

    with open(os.path.join(OUT, "yolo", "data.yaml"), "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: mosquito\n")
    shutil.make_archive(os.path.join(OUT, "yolo"), "zip", os.path.join(OUT, "yolo"))
    print("listo:", len(rows), "imágenes")


if __name__ == "__main__":
    main()
