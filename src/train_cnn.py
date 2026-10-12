"""Ajuste fino de una CNN preentrenada (ImageNet) sobre los recortes. Uso:
    python src/train_cnn.py --arch efficientnet_b0 --epochs 12 --batch 64 --lr 3e-4
    python src/train_cnn.py --arch convnext_tiny  --epochs 5  --batch 32 --lr 2e-4 --img-size 176 --freeze-stages 2

Desbalance (Buda et al., 2018): sobremuestreo con WeightedRandomSampler hasta clases equiprobables
+ aumento de datos. Se elige la época con mejor F1 macro en validación.

Nota de rendimiento (medido en la GTX 1650, sin tensor cores): AMP en formato NCHW. channels_last
bajó EfficientNet-B0 de 121 a 16 img/s, por eso no se usa.
"""
import argparse
import os
import time

import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from common import CLASSES, MODELS_DIR, RESULTS_DIR, load_splits, save_results

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def transforms(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    # Recortes guardados a 256 px: centro de 224 (87.5 %) y luego al tamaño de entrada
    evaluation = T.Compose([T.CenterCrop(224), T.Resize(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evaluation


class Crops(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.tf = df["crop_path"].tolist(), df["y"].tolist(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.to(device, non_blocking=True)).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--freeze-stages", type=int, default=0, help="congela stem + las primeras N etapas (ConvNeXt)")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    df = load_splits()
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))

    counts = tr["y"].value_counts().sort_index().to_numpy()
    sampler = WeightedRandomSampler(torch.tensor(1.0 / counts[tr["y"]], dtype=torch.double), len(tr), replacement=True)
    train_tf, eval_tf = transforms(args.img_size)
    # En Windows cada worker es un proceso con su propia copia de torch (~600 MB): solo el loader de
    # entrenamiento usa workers; validación/test cargan en el proceso principal (~5 s por pasada)
    train_dl = DataLoader(Crops(tr, train_tf), batch_size=args.batch, sampler=sampler, drop_last=True,
                          num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(Crops(va, eval_tf), batch_size=args.batch * 2, pin_memory=True)
    test_dl = DataLoader(Crops(te, eval_tf), batch_size=args.batch * 2, pin_memory=True)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(CLASSES)).to(device)
    if args.freeze_stages:
        for module in [model.stem, *model.stages[:args.freeze_stages]]:
            module.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt = os.path.join(MODELS_DIR, f"{args.arch}.pt")
    history, best_f1, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        te0, total = time.time(), 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val_f1 = f1_score(va["y"], predict(model, val_dl, device).argmax(1), average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(train_dl), "val_macro_f1": val_f1, "sec": time.time() - te0})
        print(f"[{args.arch}] epoch {epoch}/{args.epochs} loss {history[-1]['train_loss']:.4f} val F1 {val_f1:.4f} ({history[-1]['sec']:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            # fp16 para que el archivo quepa en GitHub (<100 MB)
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)  # escritura atómica: nunca queda un checkpoint a medias
    train_sec = time.time() - t0
    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, f"history_{args.arch}.csv"), index=False)

    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in torch.load(ckpt).items()})
    t1 = time.time()
    proba = predict(model, test_dl, device)
    infer_ms = (time.time() - t1) / len(te) * 1000
    save_results(args.arch, te, proba, {
        "family": "CNN fine-tune", "train_seconds": train_sec, "infer_ms_per_img": infer_ms,
        "best_val_macro_f1": best_f1, "hyperparams": vars(args),
        "trainable_params": sum(p.numel() for p in params), "total_params": sum(p.numel() for p in model.parameters()),
    })


if __name__ == "__main__":
    main()
