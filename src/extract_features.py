"""Extrae características DINOv2 ViT-S/14 congelado (Oquab et al., 2024) de todos los recortes.
    python src/extract_features.py   ->  data/processed/dinov2_feats.npz
"""
import os
import time

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import load_splits

OUT = "data/processed/dinov2_feats.npz"
ARCH = "vit_small_patch14_dinov2.lvd142m"


class Crops(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def build_model():
    model = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**{**cfg, "input_size": (3, 224, 224), "crop_pct": 224 / 256})
    return model, tf


@torch.no_grad()
def main():
    df = load_splits()
    model, tf = build_model()
    model = model.cuda()
    dl = DataLoader(Crops(df["crop_path"].tolist(), tf), batch_size=128, num_workers=6, pin_memory=True)
    feats, t0 = [], time.time()
    for x in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(model(x.cuda(non_blocking=True)).float().cpu())
    feats = torch.cat(feats).numpy()
    sec = time.time() - t0
    np.savez(OUT + ".tmp.npz", feats=feats, stem=df["stem"].to_numpy(), seconds=sec)
    os.replace(OUT + ".tmp.npz", OUT)
    print(f"{feats.shape} en {sec:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
