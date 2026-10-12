"""App local para clasificar mosquitos con los modelos entrenados. Desde la raíz del repo:
    conda activate mosquito-ml
    streamlit run app/streamlit_app.py
"""
import json
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from PIL import ImageDraw

import inference as inf
from common import CLASSES

ROOT = inf.ROOT
NAMES = {"aegypti": "Aedes aegypti", "albopictus": "Aedes albopictus", "anopheles": "Anopheles",
         "culex": "Culex", "culiseta": "Culiseta", "japonicus/koreicus": "Aedes japonicus/koreicus"}
DISEASES = {"aegypti": "dengue, zika, chikungunya, fiebre amarilla", "albopictus": "dengue, zika, chikungunya",
            "anopheles": "malaria, filariasis linfática", "culex": "virus del Nilo Occidental, encefalitis japonesa",
            "culiseta": "arbovirus, filariasis linfática", "japonicus/koreicus": "virus del Nilo Occidental, La Crosse"}
# Paleta categórica validada para daltonismo (orden fijo; el color sigue a la clase, nunca al rango)
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # secuencial: un solo tono, claro -> oscuro
ACCENT, MUTED = "#2a78d6", "#c3c2b7"

pio.templates["mosquito"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color="#0b0b0b", size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", barcornerradius=4,
    xaxis=dict(gridcolor="#e8e7e3", zeroline=False), yaxis=dict(gridcolor="#e8e7e3", zeroline=False),
    margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(bgcolor="white")))
pio.templates.default = "plotly_white+mosquito"

st.set_page_config(page_title="Identificador de mosquitos", page_icon="🦟", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos…")
def pipeline():
    return inf.Pipeline()


@st.cache_data
def dataset():
    df = pd.read_csv(ROOT / "data" / "processed" / "splits.csv")
    df["stem"] = df["img_fName"].str.replace(r"\.[^.]+$", "", regex=True)
    df["bbx_w"], df["bbx_h"] = df.bbx_xbr - df.bbx_xtl, df.bbx_ybr - df.bbx_ytl
    df["area_ratio"] = df.bbx_w * df.bbx_h / (df.img_w * df.img_h)
    df["aspect_ratio"] = df.img_w / df.img_h
    df["clase"] = df["class_label"].map(NAMES)
    return df


def label(name):
    return inf.LABELS.get(name, name)


pipe = pipeline()
df = dataset()
metrics = pipe.metrics
ALL = list(pipe.models)

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.title("🦟 Identificador de mosquitos")
    st.caption("Proyecto 2 · CC3084 Data Science · UVG 2026 · MosquitoAlert Challenge 2023")
    mode = st.radio("Modelos a usar", ["Todos", "Elegir"], horizontal=True)
    selected = ALL if mode == "Todos" else st.multiselect("Modelos", ALL, default=ALL[:1], format_func=label)
    show_perf = st.toggle("Mostrar rendimiento de los modelos", value=True)
    st.divider()
    st.caption(f"Detector YOLOv8n: {'activo' if pipe.detector else 'no disponible, se usa la imagen completa'}")
    st.caption(f"{len(ALL)} clasificadores cargados · inferencia en CPU")

tab_pred, tab_data, tab_perf = st.tabs(["Clasificar una imagen", "Explorar los datos", "Rendimiento de los modelos"])

# ---------------- 1. Clasificar ----------------
with tab_pred:
    st.subheader("¿Qué mosquito es?")
    st.write("Sube una fotografía (como las que envían los ciudadanos a Mosquito Alert). La app localiza el "
             "mosquito, lo recorta y lo clasifica con los modelos seleccionados.")
    c1, c2 = st.columns([3, 2])
    upload = c1.file_uploader("Fotografía", type=["jpg", "jpeg", "png", "webp"])
    with c2:
        st.write("¿No tienes una foto? Usa una del conjunto de prueba (nunca vista en entrenamiento).")
        cls = st.selectbox("Clase de la imagen de prueba", CLASSES, format_func=NAMES.get)
        if st.button("Imagen de prueba aleatoria", use_container_width=True):
            pool = df[(df.split == "test") & (df.class_label == cls)]
            st.session_state.sample = pool.iloc[random.randrange(len(pool))].to_dict()
    sample = None if upload else st.session_state.get("sample")
    source = upload or (ROOT / "data" / "raw" / "images" / sample["img_fName"] if sample else None)

    if source is None:
        st.info("Sube una imagen o elige una de prueba para empezar.")
    elif not selected:
        st.warning("Elige al menos un modelo en la barra lateral.")
    else:
        im, box, conf, crop = pipe.preprocess(source)
        results = pipe.predict(crop, selected)
        votes = pd.Series([CLASSES[int(p.argmax())] for p, _ in results.values()]).value_counts()
        winner = votes.index[0]

        left, right = st.columns([2, 3])
        with left:
            shown = im.copy()
            ImageDraw.Draw(shown).rectangle(box, outline="#eb6834", width=max(3, im.width // 150))
            st.image(shown, caption="Mosquito localizado" + (f" (confianza {conf:.0%})" if conf else ""))
            with st.expander("Ver la entrada del modelo"):
                st.image(crop, width=200, caption="Recorte cuadrado de 256 px con 15 % de margen")
        with right:
            if len(results) == 1:
                name, (proba, _) = next(iter(results.items()))
                st.metric(label(name), NAMES[winner], f"{proba.max():.0%} de probabilidad", delta_color="off")
            else:
                st.metric("Consenso de los modelos", NAMES[winner],
                          f"{votes.iloc[0]} de {len(results)} modelos coinciden", delta_color="off")
            if sample:
                ok = sample["class_label"] == winner
                st.markdown(f"Etiqueta real (entomólogos): **{NAMES[sample['class_label']]}** {'✅' if ok else '❌'}")
            st.caption(f"Enfermedades asociadas a *{NAMES[winner]}*: {DISEASES[winner]}.")
            best = next(n for n in metrics if n in results)
            st.caption(f"Qué tan fiable es: en el conjunto de prueba, {label(best)} reconoce el "
                       f"{metrics[best]['recall_per_class'][winner]:.0%} de los *{NAMES[winner]}* reales.")

            if len(results) == 1:
                s = pd.Series(proba, index=CLASSES).sort_values()
                fig = go.Figure(go.Bar(x=s.values, y=[NAMES[c] for c in s.index], orientation="h",
                                       marker_color=[CLASS_COLORS[c] for c in s.index],
                                       text=[f"{v:.1%}" for v in s.values], textposition="outside",
                                       hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                fig.update_layout(title="Probabilidad por clase", xaxis=dict(range=[0, 1.15], tickformat=".0%"), height=330)
            else:
                probs = pd.DataFrame({label(n): p for n, (p, _) in results.items()}, index=[NAMES[c] for c in CLASSES]).T
                fig = px.imshow(probs, text_auto=".0%", color_continuous_scale=BLUES, zmin=0, zmax=1, aspect="auto")
                fig.update_layout(title="Probabilidad que asigna cada modelo a cada clase", coloraxis_showscale=False,
                                  height=120 + 45 * len(probs))
                fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        table = pd.DataFrame([{
            "Modelo": label(n), "Predicción": NAMES[CLASSES[int(p.argmax())]], "Confianza": float(p.max()),
            "Tiempo (ms)": round(ms), **({"F1 macro en test": metrics[n]["macro_f1"]} if show_perf else {}),
        } for n, (p, ms) in results.items()])
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "Confianza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "F1 macro en test": st.column_config.NumberColumn(format="%.3f")})

# ---------------- 2. Explorar datos ----------------
with tab_data:
    st.subheader("El conjunto de datos después de la limpieza")
    counts = df.class_label.value_counts()
    k = st.columns(4)
    k[0].metric("Imágenes", f"{len(df):,}")
    k[1].metric("Clases", len(CLASSES))
    k[2].metric("Desbalance mayor/menor", f"{counts.max() / counts.min():.0f} : 1")
    k[3].metric("Área mediana del mosquito", f"{df.area_ratio.median():.1%} de la foto")
    st.write("Las fotos vienen de ciencia ciudadana y cada una tiene una caja anotada por entomólogos. "
             "Se descartaron las cajas geométricamente inválidas y se partió 70/15/15 estratificado por clase.")

    c1, c2 = st.columns(2)
    with c1:
        log = st.toggle("Escala logarítmica", value=False)
        cd = counts.rename_axis("class_label").reset_index(name="imágenes")
        cd["clase"] = cd.class_label.map(NAMES)
        fig = px.bar(cd, x="imágenes", y="clase", color="class_label", color_discrete_map=CLASS_COLORS,
                     orientation="h", log_x=log, text="imágenes")
        fig.update_layout(title=f"Dos clases concentran el {counts.iloc[:2].sum() / len(df):.0%} de las fotos",
                          showlegend=False, yaxis=dict(categoryorder="total ascending", title=""), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"*A. aegypti* ({counts['aegypti']}) y *Anopheles* ({counts['anopheles']}) tienen tan pocas fotos "
                   "que la exactitud global engaña: por eso los modelos se comparan con F1 macro, que pesa igual a cada clase.")
    with c2:
        var = st.selectbox("Variable por clase", ["area_ratio", "aspect_ratio", "img_w", "img_h"], format_func={
            "area_ratio": "Proporción de la foto que ocupa el mosquito", "aspect_ratio": "Relación de aspecto (ancho/alto)",
            "img_w": "Ancho de la imagen (px)", "img_h": "Alto de la imagen (px)"}.get)
        fig = px.box(df, x="clase", y=var, color="class_label", color_discrete_map=CLASS_COLORS,
                     category_orders={"clase": [NAMES[c] for c in CLASSES]}, points=False)
        fig.update_layout(title="Distribución por clase", showlegend=False, xaxis_title="", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"El mosquito suele ser un objeto pequeño (mediana {df.area_ratio.median():.0%} de la foto): por eso "
                   "primero se detecta y recorta. *A. aegypti* aparece mucho más grande: varias fotos se tomaron con lupa.")

    st.markdown("##### Así ve el modelo cada clase")
    gcls = st.radio("Clase", CLASSES, format_func=NAMES.get, horizontal=True, label_visibility="collapsed")
    stems = df[df.class_label == gcls].stem.sample(6, random_state=1)
    for col, s in zip(st.columns(6), stems):
        col.image(str(ROOT / "data" / "processed" / "crops" / f"{s}.jpg"), use_container_width=True)
    st.caption("Recortes de entrada (caja anotada + 15 % de margen, 256 px). Fondo, luz y pose cambian mucho entre fotos.")

    with st.expander("Tabla de la partición entrenamiento / validación / prueba"):
        st.dataframe(pd.crosstab(df.clase, df.split, margins=True, margins_name="Total")[["train", "val", "test", "Total"]],
                     use_container_width=True)

# ---------------- 3. Rendimiento ----------------
with tab_perf:
    if not show_perf:
        st.info("El rendimiento está oculto. Actívalo en la barra lateral con «Mostrar rendimiento de los modelos».")
    else:
        st.subheader("¿Qué tan bien funciona cada modelo en el conjunto de prueba?")
        n_test = int(np.sum(next(iter(metrics.values()))["confusion_matrix"]))
        st.write(f"Evaluación sobre {n_test:,} imágenes de prueba que ningún modelo vio al entrenar.")
        rows = pd.DataFrame([{"modelo": label(n), "F1 macro": m["macro_f1"], "Exactitud balanceada": m["balanced_accuracy"],
                              "Exactitud": m["accuracy"], "ms por imagen": m.get("infer_ms_per_img"),
                              "familia": m.get("family", "")} for n, m in metrics.items()])
        met = st.radio("Métrica", ["F1 macro", "Exactitud balanceada", "Exactitud"], horizontal=True)
        r = rows.sort_values(met)
        fig = go.Figure(go.Bar(x=r[met], y=r.modelo, orientation="h", text=[f"{v:.3f}" for v in r[met]],
                               textposition="outside", marker_color=[ACCENT if v == r[met].max() else MUTED for v in r[met]],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title=f"{met} por modelo (azul: el mejor)", xaxis=dict(range=[0, 1.08]), height=100 + 45 * len(r))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Con la exactitud todos parecen excelentes porque aciertan las dos clases grandes; "
                   "el F1 macro revela la diferencia real en las clases raras.")

        c1, c2 = st.columns(2)
        with c1:
            f1 = pd.DataFrame({label(n): m["f1_per_class"] for n, m in metrics.items()}).T[CLASSES]
            f1.columns = [NAMES[c] for c in CLASSES]
            fig = px.imshow(f1, text_auto=".2f", color_continuous_scale=BLUES, zmin=0, zmax=1, aspect="auto")
            fig.update_layout(title="F1 por clase: dónde falla cada modelo", coloraxis_showscale=False, height=440)
            fig.update_traces(hovertemplate="%{y}<br>%{x}: F1 %{z:.2f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cm_model = st.selectbox("Matriz de confusión de", list(metrics), format_func=label)
            norm = st.toggle("Normalizar por fila (recall)", value=True)
            cm = np.array(metrics[cm_model]["confusion_matrix"], dtype=float)
            z = cm / cm.sum(1, keepdims=True) if norm else cm
            fig = px.imshow(z, x=[NAMES[c] for c in CLASSES], y=[NAMES[c] for c in CLASSES],
                            text_auto=".0%" if norm else ".0f", color_continuous_scale=BLUES, aspect="auto",
                            labels=dict(x="Predicción", y="Real", color="valor"))
            fig.update_layout(coloraxis_showscale=False, height=360, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("*A. aegypti* casi siempre se confunde con *A. albopictus* (mismo género, y la segunda tiene "
                   "~100 veces más fotos); *japonicus/koreicus* también cae a menudo en *albopictus*.")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(rows.dropna(subset=["ms por imagen"]), x="ms por imagen", y="F1 macro", text="modelo",
                             symbol="familia", log_x=True)
            fig.update_traces(marker=dict(size=13, color=ACCENT, line=dict(width=2, color="white")),
                              textposition="top center")
            fig.update_layout(title="Desempeño vs costo de inferencia (GTX 1650)", height=420,
                              legend=dict(orientation="h", y=-0.25, title=""))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            hist = {n: pd.read_csv(p) for n in metrics if (p := ROOT / "results" / f"history_{n}.csv").exists()}
            if hist:
                fig = go.Figure()
                for (n, h), color in zip(hist.items(), [ACCENT, "#eb6834"]):
                    fig.add_scatter(x=h.epoch, y=h.val_macro_f1, name=label(n), mode="lines+markers",
                                    line=dict(width=2, color=color), marker=dict(size=8))
                fig.update_layout(title="CNN: F1 macro en validación por época", yaxis=dict(range=[0, 1]),
                                  xaxis_title="época", height=420, legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig, use_container_width=True)

        det_json, det_csv = ROOT / "results" / "detector_summary.json", ROOT / "models" / "detector" / "det_test_boxes.csv"
        if det_json.exists() and det_csv.exists():
            st.markdown("##### Detector YOLOv8n (localización del mosquito)")
            d, boxes = json.loads(det_json.read_text()), pd.read_csv(det_csv)
            k = st.columns(4)
            k[0].metric("mAP@50", f"{d.get('map50', float('nan')):.3f}")
            k[1].metric("IoU medio", f"{d['mean_iou']:.3f}")
            k[2].metric("Fotos con IoU ≥ 0.75", f"{d['iou_ge_075']:.1%}")
            k[3].metric("ms por imagen", f"{d.get('infer_ms_per_img', float('nan')):.0f}")
            fig = px.histogram(boxes, x="iou", nbins=40, color_discrete_sequence=[ACCENT])
            fig.add_vline(x=0.75, line_dash="dash", line_color="#0b0b0b", annotation_text="umbral del reto 0.75")
            fig.update_layout(title="IoU entre la caja predicha y la anotada (test)", height=330, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
