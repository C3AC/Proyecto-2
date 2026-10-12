"""Figuras estáticas de resultados para la presentación (versión personalizable).

    python src/make_figures.py                                   # tema claro (reports/themes/base.json)
    python src/make_figures.py --theme reports/themes/dark.json  # tema oscuro
    python src/make_figures.py --only ranking,confusion          # solo algunas figuras
    python src/make_figures.py --list                            # nombres disponibles

Todo lo visual (colores, tipografía, tamaños, textos, nombres de modelos y clases, formato de
salida, fondo transparente) vive en el JSON del tema; este archivo solo tiene la lógica de dibujo.
Para adaptar a la plantilla de la presentación: crear un JSON con "extends": "base.json" y
sobrescribir solo las claves necesarias (ver dark.json). Las figuras sin datos se omiten.
Independiente de summarize_results.py: escribe en otra carpeta y no lo modifica.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import CLASSES, RESULTS_DIR, SPLITS_CSV

DEFAULT_THEME = "reports/themes/base.json"
DETECTOR_CSV = "models/detector/det_test_boxes.csv"


# ----------------------------------------------------------------------------- tema

def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_theme(path):
    with open(path, encoding="utf-8") as f:
        theme = json.load(f)
    parent = theme.pop("extends", None)
    if parent:
        theme = deep_merge(load_theme(os.path.join(os.path.dirname(path), parent)), theme)
    return theme


class Painter:
    """Aplica el tema y resuelve colores, marcadores y etiquetas por modelo y por clase."""

    def __init__(self, theme):
        self.t = theme
        self.c, self.f, self.layout, self.txt = theme["colors"], theme["font"], theme["layout"], theme["texts"]
        cat = self.c["categorical"]
        # El color sigue a la familia del modelo, nunca a su posición en el ranking
        self.family_color = {fam["key"]: cat[i] if i < len(cat) else self.c["fallback"]
                             for i, fam in enumerate(theme["families"])}
        self.family_label = {fam["key"]: fam["label"] for fam in theme["families"]}
        self.cmap = LinearSegmentedColormap.from_list("theme_seq", self.c["sequential"])
        plt.rcParams.update({
            "font.family": "sans-serif", "font.sans-serif": self.f["family"], "font.size": self.f["size"],
            "figure.facecolor": self.c["surface"], "axes.facecolor": self.c["surface"],
            "savefig.facecolor": self.c["surface"], "text.color": self.c["text_primary"],
            "axes.labelcolor": self.c["text_secondary"], "axes.edgecolor": self.c["baseline"],
            "axes.linewidth": 1, "axes.labelsize": self.f["label_size"],
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.color": self.c["baseline"], "ytick.color": self.c["baseline"],
            "xtick.labelcolor": self.c["text_secondary"], "ytick.labelcolor": self.c["text_secondary"],
            "xtick.labelsize": self.f["tick_size"], "ytick.labelsize": self.f["tick_size"],
            "grid.color": self.c["grid"], "grid.linewidth": 1, "grid.linestyle": "-",
            "legend.frameon": False, "legend.fontsize": self.f["tick_size"],
            "legend.labelcolor": self.c["text_secondary"],
            "lines.linewidth": self.layout["line_width"], "lines.markersize": self.layout["marker_size"],
            "svg.fonttype": "none",  # texto editable al importar el SVG en PowerPoint/Figma
        })

    # --- entidades
    def model(self, name):
        return self.t["models"].get(name, {"label": name, "family": None, "marker": "o"})

    def label(self, name):
        return self.model(name)["label"]

    def color(self, name):
        return self.family_color.get(self.model(name)["family"], self.c["fallback"])

    def marker(self, name):
        return self.model(name).get("marker", "o")

    def cls(self, c):
        return self.t["classes"].get(c, c)

    def ink_on(self, fill):
        """Texto legible sobre un relleno: tinta oscura o clara según la luminancia del fondo."""
        r, g, b = to_rgb(fill)[:3]
        is_light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5
        dark_ink, light_ink = sorted([self.c["text_primary"], self.c["surface"]], key=lambda h: sum(to_rgb(h)))
        return dark_ink if is_light else light_ink

    # --- lienzo
    def figure(self, nrows=1, ncols=1, figsize=None, **kw):
        return plt.subplots(nrows, ncols, figsize=figsize or self.layout["figsize"], **kw)

    def header(self, fig, key, **fmt):
        texts = self.txt[key]
        has_sub = self.layout["show_subtitles"] and texts.get("subtitle")
        has_note = self.layout["show_source_note"] and self.txt.get("source_note")
        fig.tight_layout(rect=[0, 0.05 if has_note else 0, 1, 0.86 if has_sub else 0.91])
        fig.text(0.012, 0.965, texts["title"].format(**fmt), ha="left", va="top",
                 fontsize=self.f["title_size"], fontweight=self.f["title_weight"], color=self.c["text_primary"])
        if has_sub:
            fig.text(0.012, 0.905, texts["subtitle"].format(**fmt), ha="left", va="top",
                     fontsize=self.f["subtitle_size"], color=self.c["text_secondary"])
        if has_note:
            fig.text(0.012, 0.012, self.txt["source_note"], ha="left", va="bottom",
                     fontsize=self.f["annotation_size"] - 1, color=self.c["muted"])

    def save(self, fig, name):
        out = self.t["output"]
        os.makedirs(out["dir"], exist_ok=True)
        for fmt in out["formats"]:
            fig.savefig(os.path.join(out["dir"], f"{out['prefix']}{name}.{fmt}"),
                        dpi=out["dpi"], transparent=out["transparent"])
        plt.close(fig)
        print("  ", name)

    def legend_families(self, ax, families, **kw):
        handles = [Patch(color=self.family_color[k], label=self.family_label[k]) for k in families]
        ax.legend(handles=handles, **kw)

    def model_handle(self, name, label=None):
        return Line2D([], [], ls="", marker=self.marker(name), color=self.color(name),
                      markeredgecolor=self.c["surface"], markersize=self.layout["marker_size"],
                      label=label or self.label(name))

    def dot(self, ax, x, y, name):
        ax.scatter(x, y, marker=self.marker(name), color=self.color(name), s=self.layout["marker_size"] ** 2,
                   edgecolors=self.c["surface"], linewidths=2, zorder=3)  # anillo del color de fondo

    def clean_barh(self, ax):
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    def colorbar(self, fig, im, ax, fraction):
        cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(color=self.c["baseline"], labelcolor=self.c["text_secondary"])


# ----------------------------------------------------------------------------- datos

def load_data():
    runs = [json.load(open(p, encoding="utf-8")) for p in glob.glob(f"{RESULTS_DIR}/metrics_*.json")]
    runs.sort(key=lambda r: r["macro_f1"], reverse=True)
    splits = pd.read_csv(SPLITS_CSV) if os.path.exists(SPLITS_CSV) else None
    support = splits[splits.split == "train"]["class_label"].value_counts() if splits is not None else None
    bench_path = os.path.join(RESULTS_DIR, "gpu_benchmark.csv")
    return {
        "runs": runs, "by_name": {r["model"]: r for r in runs}, "splits": splits, "support": support,
        "history": {os.path.basename(p)[len("history_"):-4]: pd.read_csv(p)
                    for p in glob.glob(f"{RESULTS_DIR}/history_*.csv")},
        "bench": pd.read_csv(bench_path) if os.path.exists(bench_path) else None,
        "detector": pd.read_csv(DETECTOR_CSV) if os.path.exists(DETECTOR_CSV) else None,
    }


def families_present(p, names):
    present = {p.model(n)["family"] for n in names}
    return [f["key"] for f in p.t["families"] if f["key"] in present]


def best_per_family(p, runs):
    best = {}
    for r in runs:  # runs ya viene ordenado por F1 macro
        best.setdefault(p.model(r["model"])["family"], r)
    return [best[f["key"]] for f in p.t["families"] if f["key"] in best]


def spread_labels(ys, min_gap):
    """Posiciones verticales para etiquetas sin solaparse, lo más cerca posible de su punto."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for prev, cur in zip(order, order[1:]):
        out[cur] = max(out[cur], out[prev] + min_gap)
    return out


# ----------------------------------------------------------------------------- figuras

def class_distribution(p, d):
    if d["splits"] is None:
        return None
    counts = d["splits"]["class_label"].value_counts().sort_values()
    total = counts.sum()
    fig, ax = p.figure()
    ax.barh([p.cls(c) for c in counts.index], counts.values, height=p.layout["bar_height"],
            color=p.c["categorical"][0], zorder=2)
    for i, v in enumerate(counts.values):
        ax.text(v + total * 0.006, i, f"{v:,}  ({v / total:.1%})", va="center",
                fontsize=p.f["annotation_size"], color=p.c["text_secondary"])
    ax.set_xlim(0, counts.max() * 1.22)
    ax.set_xlabel(p.txt["class_distribution"]["x"])
    p.clean_barh(ax)
    p.header(fig, "class_distribution")
    return fig


def ranking(p, d):
    runs = d["runs"][::-1]  # el mejor queda arriba
    if not runs:
        return None
    fig, ax = p.figure()
    ys = list(range(len(runs)))
    ax.barh(ys, [r["macro_f1"] for r in runs], height=p.layout["bar_height"],
            color=[p.color(r["model"]) for r in runs], zorder=2)
    for y, r in zip(ys, runs):
        ax.text(r["macro_f1"] + 0.01, y, f"{r['macro_f1']:.3f}", va="center",
                fontsize=p.f["annotation_size"], color=p.c["text_primary"])
    ax.set_yticks(ys, [p.label(r["model"]) for r in runs])
    ax.set_xlim(0, 1)
    ax.set_xlabel(p.txt["ranking"]["x"])
    p.clean_barh(ax)
    p.legend_families(ax, families_present(p, [r["model"] for r in runs]), loc="lower right")
    p.header(fig, "ranking")
    return fig


def f1_by_class(p, d):
    if not d["runs"] or d["support"] is None:
        return None
    classes = list(d["support"].sort_values(ascending=False).index)
    runs = d["runs"]
    m = np.array([[r["f1_per_class"][c] for c in classes] for r in runs])
    fig, ax = p.figure(figsize=(p.layout["figsize"][0], max(4.8, 0.62 * len(runs) + 2.8)))
    im = ax.imshow(m, cmap=p.cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=p.f["annotation_size"],
                    color=p.ink_on(p.cmap(m[i, j])))
    ax.set_xticks(range(len(classes)), [f"{p.cls(c)}\nn = {d['support'][c]:,}" for c in classes])
    ax.set_yticks(range(len(runs)), [p.label(r["model"]) for r in runs])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    p.colorbar(fig, im, ax, 0.025)
    p.header(fig, "f1_by_class")
    return fig


def recall_vs_support(p, d):
    if not d["runs"] or d["support"] is None:
        return None
    classes = list(d["support"].sort_values().index)
    best = best_per_family(p, d["runs"])
    fig, ax = p.figure()
    offsets = np.linspace(-0.18, 0.18, len(best)) if len(best) > 1 else [0.0]
    for off, r in zip(offsets, best):
        p.dot(ax, np.arange(len(classes)) + off, [r["recall_per_class"][c] for c in classes], r["model"])
    ax.set_xticks(range(len(classes)), [f"{p.cls(c)}\nn = {d['support'][c]:,}" for c in classes])
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel(p.txt["recall_vs_support"]["y"])
    ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[p.model_handle(r["model"]) for r in best], loc="lower right")
    p.header(fig, "recall_vs_support")
    return fig


def confusion(p, d):
    if not d["runs"]:
        return None
    best = d["runs"][0]
    cm = np.array(best["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = p.figure(figsize=(9.5, 7.8))
    im = ax.imshow(norm, cmap=p.cmap, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=p.f["annotation_size"],
                    color=p.ink_on(p.cmap(norm[i, j])))
    labels = [p.cls(c) for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels)
    ax.set_xlabel(p.txt["confusion"]["x"])
    ax.set_ylabel(p.txt["confusion"]["y"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    p.colorbar(fig, im, ax, 0.04)
    p.header(fig, "confusion", model=p.label(best["model"]))
    return fig


def cost_vs_f1(p, d):
    runs = [r for r in d["runs"] if r.get("infer_ms_per_img")]
    if not runs:
        return None
    fig, ax = p.figure()
    xs = np.array([r["infer_ms_per_img"] for r in runs])
    ys = np.array([r["macro_f1"] for r in runs])
    for x, y, r in zip(xs, ys, runs):
        p.dot(ax, x, y, r["model"])
    ax.set_xscale("log")
    ax.set_xlim(xs.min() / 1.6, xs.max() * 4)
    ax.set_ylim(max(0, ys.min() - 0.12), min(1, ys.max() + 0.08))
    # Etiquetas a la derecha, separadas verticalmente y unidas a su punto con una guía fina
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.065
    for x, y, ly, r in zip(xs, ys, spread_labels(ys, gap), runs):
        ax.annotate(p.label(r["model"]), (x, y), xytext=(x * 1.5, ly), va="center",
                    fontsize=p.f["annotation_size"], color=p.c["text_secondary"],
                    arrowprops=dict(arrowstyle="-", color=p.c["baseline"], lw=1, shrinkA=0, shrinkB=5))
    ax.set_xlabel(p.txt["cost_vs_f1"]["x"])
    ax.set_ylabel(p.txt["cost_vs_f1"]["y"])
    ax.grid(zorder=0)
    p.legend_families(ax, families_present(p, [r["model"] for r in runs]), loc="lower left")
    p.header(fig, "cost_vs_f1")
    return fig


def training_curves(p, d):
    if not d["history"]:
        return None
    fig, (a1, a2) = p.figure(1, 2)
    for (name, h), ls in zip(sorted(d["history"].items()), ["-", "--", ":", "-."]):
        kw = dict(color=p.color(name), ls=ls, marker=p.marker(name), markeredgecolor=p.c["surface"],
                  markeredgewidth=2, label=p.label(name))
        a1.plot(h["epoch"], h["train_loss"], **kw)
        a2.plot(h["epoch"], h["val_macro_f1"], **kw)
        a2.annotate(f"{h['val_macro_f1'].iloc[-1]:.3f}", (h["epoch"].iloc[-1], h["val_macro_f1"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=p.f["annotation_size"], color=p.c["text_secondary"])
    tx = p.txt["training_curves"]
    for ax, title in ((a1, tx["loss"]), (a2, tx["val_f1"])):
        ax.set_title(title, loc="left", fontsize=p.f["label_size"], color=p.c["text_secondary"])
        ax.set_xlabel(tx["x"])
        ax.grid(axis="y", zorder=0)
        ax.xaxis.get_major_locator().set_params(integer=True)
    a2.set_ylim(0, 1)
    a2.legend(loc="lower right")
    p.header(fig, "training_curves")
    return fig


def hierarchy(p, d):
    names = [n for n in ("dinov2_logreg", "dinov2_hier_taxonomic", "dinov2_hier_cascade") if n in d["by_name"]]
    if len(names) < 2 or d["support"] is None:
        return None
    classes = list(d["support"].sort_values().index)
    vals = np.array([[d["by_name"][n]["f1_per_class"][c] for c in classes] for n in names])
    fig, ax = p.figure()
    for j in range(len(classes)):  # rango entre variantes: cuánto cambia cada clase según el esquema
        ax.plot([vals[:, j].min(), vals[:, j].max()], [j, j], color=p.c["grid"], lw=6,
                solid_capstyle="round", zorder=1)
    for off, n, row in zip(np.linspace(-0.12, 0.12, len(names)), names, vals):
        p.dot(ax, row, np.arange(len(classes)) + off, n)
    ax.legend(handles=[p.model_handle(n, f"{p.label(n)}  (F1 macro {d['by_name'][n]['macro_f1']:.3f})")
                       for n in names], loc="upper left")  # esquina vacía: las clases grandes están a la derecha
    ax.set_yticks(range(len(classes)), [f"{p.cls(c)}  (n = {d['support'][c]:,})" for c in classes])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(p.txt["hierarchy"]["x"])
    p.clean_barh(ax)
    p.header(fig, "hierarchy")
    return fig


def detector_iou(p, d):
    det = d["detector"]
    if det is None:
        return None
    fig, ax = p.figure()
    ax.hist(det["iou"], bins=np.linspace(0, 1, 41), color=p.c["categorical"][0],
            edgecolor=p.c["surface"], linewidth=2, zorder=2)
    ok = (det["iou"] >= 0.75).mean()
    ax.axvline(0.75, color=p.c["highlight"], lw=2, zorder=3)
    tx = p.txt["detector_iou"]
    ax.text(0.73, ax.get_ylim()[1] * 0.95, tx["threshold_label"].format(pct=f"{ok:.1%}"), ha="right", va="top",
            fontsize=p.f["annotation_size"], color=p.c["text_primary"])
    ax.set_xlabel(tx["x"])
    ax.set_ylabel(tx["y"])
    ax.grid(axis="y", zorder=0)
    p.header(fig, "detector_iou")
    return fig


def gpu_benchmark(p, d):
    if d["bench"] is None:
        return None
    b = d["bench"].iloc[::-1].reset_index(drop=True)
    cfg = np.where(b["amp"], "fp16", "fp32") + np.where(b["channels_last"], " + channels_last", "")
    names = b["arch"].map(p.label) + " · " + cfg
    best_of_arch = b.groupby("arch")["img_per_s"].transform("max")
    slow = b["img_per_s"] < 0.5 * best_of_arch  # configuraciones que desperdician más de la mitad
    fig, ax = p.figure()
    ax.barh(b.index, b["img_per_s"], height=p.layout["bar_height"], zorder=2,
            color=[p.c["highlight"] if s else p.color(a) for s, a in zip(slow, b["arch"])])
    for i, v in zip(b.index, b["img_per_s"]):
        ax.text(v + b["img_per_s"].max() * 0.01, i, f"{v:.0f}", va="center",
                fontsize=p.f["annotation_size"], color=p.c["text_primary"])
    ax.set_yticks(b.index, names)
    ax.set_xlabel(p.txt["gpu_benchmark"]["x"])
    p.clean_barh(ax)
    p.header(fig, "gpu_benchmark")
    return fig


FIGURES = {
    "01_distribucion_clases": class_distribution,
    "02_ranking_modelos": ranking,
    "03_f1_por_clase": f1_by_class,
    "04_recall_vs_soporte": recall_vs_support,
    "05_confusion_mejor_modelo": confusion,
    "06_costo_vs_f1": cost_vs_f1,
    "07_curvas_entrenamiento": training_curves,
    "08_plano_vs_jerarquico": hierarchy,
    "09_iou_detector": detector_iou,
    "10_benchmark_gpu": gpu_benchmark,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default=DEFAULT_THEME)
    ap.add_argument("--only", help="nombres separados por coma (basta una parte: ranking,confusion)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(FIGURES))
        return

    p = Painter(load_theme(args.theme))
    d = load_data()
    wanted = [w.strip() for w in args.only.split(",")] if args.only else None
    print(f"Tema: {args.theme} -> {p.t['output']['dir']}")
    for name, fn in FIGURES.items():
        if wanted and not any(w in name for w in wanted):
            continue
        fig = fn(p, d)
        if fig is None:
            print("   (sin datos)", name)
        else:
            p.save(fig, name)


if __name__ == "__main__":
    main()
